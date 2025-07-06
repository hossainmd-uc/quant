#!/usr/bin/env python3
"""
Chunked PDF extractor that splits large PDFs into 2-page chunks
to work within LandingAI API limits
"""

import os
import json
import base64
import requests
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import shutil

try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    try:
        import pypdf as PyPDF2
        PDF_LIBRARY = "pypdf"
    except ImportError:
        print("❌ Need PyPDF2 or pypdf: pip install PyPDF2")
        PyPDF2 = None
        PDF_LIBRARY = None

@dataclass
class ChunkResult:
    """Result from processing a single chunk"""
    chunk_id: int
    pages: List[int]
    chunks: List[Dict[str, Any]]
    markdown: str
    success: bool
    error: Optional[str] = None

@dataclass
class UsageTracker:
    """Track API usage and costs"""
    api_key: str
    pages_processed: int = 0
    chunks_processed: int = 0
    cost_per_page: float = 0.03
    total_cost: float = 0.0
    free_credits: float = 10.0
    max_pages_per_key: int = 333
    start_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def remaining_credits(self) -> float:
        return max(0, self.free_credits - self.total_cost)
    
    @property
    def remaining_pages(self) -> int:
        return max(0, self.max_pages_per_key - self.pages_processed)
    
    @property
    def is_exhausted(self) -> bool:
        return self.remaining_pages <= 0
    
    def add_pages(self, pages: int) -> None:
        self.pages_processed += pages
        self.chunks_processed += 1
        self.total_cost = self.pages_processed * self.cost_per_page
        print(f"Processed {pages} pages (chunk {self.chunks_processed}). Total: {self.pages_processed} pages, ${self.total_cost:.2f}")

class ChunkedExtractor:
    """Extract documents by splitting into 2-page chunks"""
    
    def __init__(self, api_keys: List[str], pages_per_chunk: int = 2):
        """
        Initialize ChunkedExtractor
        
        Args:
            api_keys: List of API keys for rotation
            pages_per_chunk: Pages per chunk (max 2 for LandingAI)
        """
        if not api_keys:
            raise ValueError("At least one API key required")
        
        if pages_per_chunk > 2:
            print("⚠️  LandingAI API has 2-page limit. Setting pages_per_chunk to 2.")
            pages_per_chunk = 2
        
        self.api_keys = api_keys
        self.pages_per_chunk = pages_per_chunk
        self.usage_trackers = [UsageTracker(api_key=key) for key in api_keys]
        self.current_key_index = 0
        
        if not PDF_LIBRARY:
            raise ImportError("PDF library required: pip install PyPDF2")
        
        print(f"ChunkedExtractor initialized with {len(api_keys)} API keys")
        print(f"Processing {pages_per_chunk} pages per chunk")
    
    def _encode_api_key(self, api_key: str) -> str:
        """Encode API key for Basic Auth"""
        return base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
    
    def _split_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """Split PDF into chunks"""
        if not PDF_LIBRARY:
            raise ImportError("PDF library required")
        
        chunk_paths = []
        
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(reader.pages)
            
            print(f"Splitting {total_pages} pages into {self.pages_per_chunk}-page chunks...")
            
            for i in range(0, total_pages, self.pages_per_chunk):
                writer = PyPDF2.PdfWriter()
                
                # Add pages to this chunk
                pages_in_chunk = 0
                for j in range(i, min(i + self.pages_per_chunk, total_pages)):
                    writer.add_page(reader.pages[j])
                    pages_in_chunk += 1
                
                # Save chunk
                chunk_path = os.path.join(output_dir, f"chunk_{i//self.pages_per_chunk + 1:04d}.pdf")
                with open(chunk_path, 'wb') as chunk_file:
                    writer.write(chunk_file)
                
                chunk_paths.append(chunk_path)
                print(f"Created chunk {len(chunk_paths)}/{(total_pages + self.pages_per_chunk - 1) // self.pages_per_chunk}: {pages_in_chunk} pages")
        
        return chunk_paths
    
    def _process_chunk(self, chunk_path: str, chunk_id: int) -> ChunkResult:
        """Process a single PDF chunk"""
        url = "https://api.va.landing.ai/v1/tools/document-analysis"
        
        current_api_key = self.api_keys[self.current_key_index]
        encoded_key = self._encode_api_key(current_api_key)
        
        headers = {
            "Authorization": f"Basic {encoded_key}"
        }
        
        data = {
            "parse_text": True,
            "parse_tables": True,
            "parse_figures": True,
            "summary_verbosity": "none",
            "caption_format": "json",
            "response_format": "json",
            "return_chunk_crops": False,
            "return_page_crops": False,
        }
        
        try:
            with open(chunk_path, 'rb') as chunk_file:
                files = {"pdf": (f"chunk_{chunk_id}.pdf", chunk_file, "application/pdf")}
                
                print(f"Processing chunk {chunk_id} with API key {self.current_key_index + 1}/{len(self.api_keys)}")
                
                response = requests.post(url, files=files, data=data, headers=headers, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Count pages in this chunk
                    with open(chunk_path, 'rb') as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        pages_in_chunk = len(reader.pages)
                    
                    # Update usage
                    self.usage_trackers[self.current_key_index].add_pages(pages_in_chunk)
                    
                    # Extract content from the correct API response format
                    extracted_chunks = []
                    markdown_parts = []
                    
                    if "data" in result and "pages" in result["data"]:
                        for page in result["data"]["pages"]:
                            if "chunks" in page:
                                for chunk in page["chunks"]:
                                    if "caption" in chunk and chunk["caption"]:
                                        extracted_chunks.append(chunk)
                                        # Ensure caption is a string
                                        caption = chunk["caption"]
                                        if isinstance(caption, str):
                                            markdown_parts.append(caption)
                                        else:
                                            # Convert to string if it's not already
                                            markdown_parts.append(str(caption))
                    
                    combined_markdown = "\n".join(markdown_parts)
                    
                    return ChunkResult(
                        chunk_id=chunk_id,
                        pages=list(range(pages_in_chunk)),
                        chunks=extracted_chunks,
                        markdown=combined_markdown,
                        success=True
                    )
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    print(f"❌ Chunk {chunk_id} failed: {error_msg}")
                    
                    # Try rotating API key if it's a quota/auth issue
                    if response.status_code in [401, 429]:
                        if self.rotate_api_key():
                            print(f"Retrying chunk {chunk_id} with new API key...")
                            return self._process_chunk(chunk_path, chunk_id)
                    
                    return ChunkResult(
                        chunk_id=chunk_id,
                        pages=[],
                        chunks=[],
                        markdown="",
                        success=False,
                        error=error_msg
                    )
                    
        except Exception as e:
            error_msg = f"Exception processing chunk {chunk_id}: {e}"
            print(f"❌ {error_msg}")
            return ChunkResult(
                chunk_id=chunk_id,
                pages=[],
                chunks=[],
                markdown="",
                success=False,
                error=error_msg
            )
    
    def rotate_api_key(self) -> bool:
        """Rotate to next available API key"""
        for i in range(len(self.api_keys)):
            next_index = (self.current_key_index + 1 + i) % len(self.api_keys)
            if not self.usage_trackers[next_index].is_exhausted:
                self.current_key_index = next_index
                print(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
                return True
        
        print("❌ All API keys exhausted!")
        return False
    
    def extract_pdf(self, pdf_path: str, output_dir: str = "extracted_knowledge") -> Dict[str, Any]:
        """Extract knowledge from PDF by processing chunks"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp(prefix="pdf_chunks_")
        
        try:
            # Split PDF into chunks
            chunk_paths = self._split_pdf(pdf_path, temp_dir)
            
            print(f"Processing {len(chunk_paths)} chunks...")
            
            # Process each chunk
            results = []
            failed_chunks = []
            
            for i, chunk_path in enumerate(chunk_paths):
                chunk_result = self._process_chunk(chunk_path, i + 1)
                
                if chunk_result.success:
                    results.append(chunk_result)
                    print(f"✅ Chunk {i+1}/{len(chunk_paths)}: {len(chunk_result.chunks)} chunks, {len(chunk_result.markdown)} chars")
                else:
                    failed_chunks.append(chunk_result)
                    print(f"❌ Chunk {i+1}/{len(chunk_paths)}: {chunk_result.error}")
                
                # Small delay to be nice to the API
                time.sleep(1)
            
            # Combine results
            all_chunks = []
            all_markdown = []
            
            for result in results:
                all_chunks.extend(result.chunks)
                if result.markdown:
                    all_markdown.append(result.markdown)
            
            combined_markdown = "\n\n".join(all_markdown)
            
            # Save results
            final_result = {
                "pdf_path": pdf_path,
                "total_chunks_processed": len(results),
                "failed_chunks": len(failed_chunks),
                "total_content_chunks": len(all_chunks),
                "markdown": combined_markdown,
                "chunks": all_chunks,
                "usage_summary": self.get_usage_summary(),
                "extraction_date": datetime.now().isoformat()
            }
            
            # Save to files
            knowledge_file = output_path / "knowledge.json"
            with open(knowledge_file, 'w') as f:
                json.dump(final_result, f, indent=2)
            
            print(f"\n🎉 Extraction complete!")
            print(f"   Processed chunks: {len(results)}/{len(chunk_paths)}")
            print(f"   Content chunks: {len(all_chunks)}")
            print(f"   Markdown length: {len(combined_markdown):,} characters")
            print(f"   Results saved to: {knowledge_file}")
            
            self.print_usage_summary()
            
            return final_result
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary across all API keys"""
        total_pages = sum(t.pages_processed for t in self.usage_trackers)
        total_cost = sum(t.total_cost for t in self.usage_trackers)
        remaining_credits = sum(t.remaining_credits for t in self.usage_trackers)
        
        return {
            "total_pages_processed": total_pages,
            "total_cost": total_cost,
            "remaining_credits": remaining_credits,
            "active_keys": len([t for t in self.usage_trackers if not t.is_exhausted]),
            "exhausted_keys": len([t for t in self.usage_trackers if t.is_exhausted]),
            "trackers": [asdict(t) for t in self.usage_trackers]
        }
    
    def print_usage_summary(self):
        """Print usage summary"""
        print("\n📊 Usage Summary:")
        total_pages = sum(t.pages_processed for t in self.usage_trackers)
        total_cost = sum(t.total_cost for t in self.usage_trackers)
        remaining_credits = sum(t.remaining_credits for t in self.usage_trackers)
        
        print(f"   Total pages processed: {total_pages}")
        print(f"   Total cost: ${total_cost:.2f}")
        print(f"   Remaining credits: ${remaining_credits:.2f}")
        
        for i, tracker in enumerate(self.usage_trackers):
            status = "EXHAUSTED" if tracker.is_exhausted else "ACTIVE"
            print(f"   Key {i+1}: {tracker.pages_processed} pages, ${tracker.total_cost:.2f} - {status}")

def load_api_keys(file_path: str = "api_keys.txt") -> List[str]:
    """Load API keys from file"""
    try:
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        api_keys = []
        for line in lines:
            if ':' in line:
                api_keys.append(line)
                print(f"✅ Loaded API key: {line[:20]}...")
            else:
                print(f"❌ Invalid API key format: {line[:20]}...")
        
        return api_keys
    except FileNotFoundError:
        print(f"❌ API keys file not found: {file_path}")
        return []

def main():
    """Main extraction function"""
    # Load API keys
    api_keys = load_api_keys()
    
    if not api_keys:
        print("❌ No API keys found!")
        return
    
    # Filter out exhausted API keys (the first one is exhausted)
    active_keys = []
    for key in api_keys:
        if not key.startswith('umm6ncm'):  # Skip the exhausted key
            active_keys.append(key)
            print(f"✅ Using active API key: {key[:20]}...")
        else:
            print(f"⚠️  Skipping exhausted API key: {key[:20]}...")
    
    if not active_keys:
        print("❌ No active API keys found!")
        return
    
    print(f"📊 Using {len(active_keys)} active API keys out of {len(api_keys)} total")
    
    # Find PDF
    pdf_path = "Python for Finance Mastering Data-Driven Finance (Yves Hilpisch) (Z-Library).pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    # Create extractor with active keys only
    extractor = ChunkedExtractor(active_keys)
    
    # Extract
    try:
        result = extractor.extract_pdf(pdf_path)
        print(f"\n🎉 Success! Extracted {len(result['chunks'])} content chunks")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    main() 