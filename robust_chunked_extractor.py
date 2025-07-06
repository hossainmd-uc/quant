#!/usr/bin/env python3

import os
import json
import time
import base64
import requests
import tempfile
import shutil
import signal
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import PyPDF2

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

class RobustChunkedExtractor:
    """Enhanced extractor with checkpoint saving and all API key usage"""
    
    def __init__(self, api_keys: List[str], pages_per_chunk: int = 2, checkpoint_interval: int = 10):
        self.api_keys = api_keys
        self.pages_per_chunk = pages_per_chunk
        self.checkpoint_interval = checkpoint_interval
        self.current_key_index = 0
        self.usage_trackers = [UsageTracker(key) for key in api_keys]
        self.results = []
        self.failed_chunks = []
        self.checkpoint_file = "extraction_checkpoint.json"
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"🚀 Initialized with {len(api_keys)} API keys, checkpoint every {checkpoint_interval} chunks")
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully"""
        print(f"\n⚠️  Received signal {signum}, saving progress...")
        self._save_checkpoint()
        print("💾 Progress saved! You can resume later.")
        sys.exit(0)
    
    def _encode_api_key(self, api_key: str) -> str:
        """Encode API key for Basic Auth"""
        return base64.b64encode(api_key.encode()).decode()
    
    def _save_checkpoint(self):
        """Save current progress to checkpoint file"""
        checkpoint_data = {
            "results": [asdict(result) for result in self.results],
            "failed_chunks": [asdict(chunk) for chunk in self.failed_chunks],
            "usage_trackers": [self._serialize_tracker(tracker) for tracker in self.usage_trackers],
            "current_key_index": self.current_key_index,
            "checkpoint_time": datetime.now().isoformat(),
            "next_chunk_to_process": len(self.results) + len(self.failed_chunks) + 1
        }
    
    def _serialize_tracker(self, tracker):
        """Serialize tracker with datetime handling"""
        data = asdict(tracker)
        if 'start_time' in data and data['start_time']:
            data['start_time'] = data['start_time'].isoformat()
        return data
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"💾 Checkpoint saved: {len(self.results)} successful chunks, {len(self.failed_chunks)} failed")
    
    def _load_checkpoint(self) -> bool:
        """Load progress from checkpoint file"""
        if not os.path.exists(self.checkpoint_file):
            return False
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Restore results
            self.results = []
            for result_data in checkpoint_data.get("results", []):
                result = ChunkResult(**result_data)
                self.results.append(result)
            
            # Restore failed chunks
            self.failed_chunks = []
            for failed_data in checkpoint_data.get("failed_chunks", []):
                failed = ChunkResult(**failed_data)
                self.failed_chunks.append(failed)
            
            # Restore usage trackers
            for i, tracker_data in enumerate(checkpoint_data.get("usage_trackers", [])):
                if i < len(self.usage_trackers):
                    # Update existing tracker with saved data
                    for key, value in tracker_data.items():
                        if key != 'start_time':  # Keep original start time
                            setattr(self.usage_trackers[i], key, value)
            
            self.current_key_index = checkpoint_data.get("current_key_index", 0)
            next_chunk = checkpoint_data.get("next_chunk_to_process", 1)
            
            print(f"📂 Loaded checkpoint: {len(self.results)} chunks completed, resuming from chunk {next_chunk}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def _split_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """Split PDF into smaller chunks"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        chunk_paths = []
        
        print(f"Splitting {total_pages} pages into chunks of {self.pages_per_chunk} pages...")
        
        for i in range(0, total_pages, self.pages_per_chunk):
            # Create new PDF for this chunk
            chunk_doc = fitz.open()
            
            # Add pages to chunk
            end_page = min(i + self.pages_per_chunk, total_pages)
            chunk_doc.insert_pdf(doc, from_page=i, to_page=end_page-1)
            
            # Save chunk
            chunk_path = os.path.join(output_dir, f"chunk_{i//self.pages_per_chunk + 1:04d}.pdf")
            chunk_doc.save(chunk_path)
            chunk_doc.close()
            
            chunk_paths.append(chunk_path)
            print(f"Created chunk {i//self.pages_per_chunk + 1}/{(total_pages + self.pages_per_chunk - 1) // self.pages_per_chunk}: {end_page - i} pages")
        
        doc.close()
        print(f"✅ Created {len(chunk_paths)} chunks")
        return chunk_paths
    
    def _process_chunk(self, chunk_path: str, chunk_id: int) -> ChunkResult:
        """Process a single PDF chunk"""
        try:
            current_tracker = self.usage_trackers[self.current_key_index]
            
            # Check if current key is exhausted
            if current_tracker.is_exhausted:
                if not self.rotate_api_key():
                    return ChunkResult(
                        chunk_id=chunk_id,
                        pages=[],
                        chunks=[],
                        markdown="",
                        success=False,
                        error="All API keys exhausted"
                    )
                current_tracker = self.usage_trackers[self.current_key_index]
            
            # Encode API key
            encoded_key = self._encode_api_key(current_tracker.api_key)
            
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
            
            # Make API call with file upload (same as working original)
            with open(chunk_path, 'rb') as chunk_file:
                files = {"pdf": (f"chunk_{chunk_id}.pdf", chunk_file, "application/pdf")}
                
                response = requests.post(
                    'https://api.va.landing.ai/v1/tools/document-analysis',
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=120
                )
            
                            if response.status_code == 200:
                result = response.json()
                
                # Count pages in this chunk (same as original)
                with open(chunk_path, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    pages_in_chunk = len(reader.pages)
                
                # Update usage
                current_tracker.add_pages(pages_in_chunk)
                
                # Extract content from the correct API response format (same as original)
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
                
            elif response.status_code == 429:
                # Rate limit - try to rotate key
                error_msg = f"Rate limit exceeded for key {self.current_key_index + 1}"
                print(f"❌ Chunk {chunk_id} failed: API Error 429: Rate limit exceeded")
                
                if self.rotate_api_key():
                    print(f"Retrying chunk {chunk_id} with new API key...")
                    return self._process_chunk(chunk_path, chunk_id)  # Retry with new key
                else:
                    return ChunkResult(
                        chunk_id=chunk_id,
                        pages=[],
                        chunks=[],
                        markdown="",
                        success=False,
                        error="All API keys exhausted after rate limit"
                    )
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                current_tracker.errors.append(error_msg)
                print(f"❌ Chunk {chunk_id} failed: {error_msg}")
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
        """Extract knowledge from PDF by processing chunks with checkpoint support"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Try to load checkpoint
        resume_from_checkpoint = self._load_checkpoint()
        
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp(prefix="pdf_chunks_")
        
        try:
            # Split PDF into chunks
            chunk_paths = self._split_pdf(pdf_path, temp_dir)
            
            print(f"Processing {len(chunk_paths)} chunks...")
            
            # Determine where to start
            start_chunk = len(self.results) + len(self.failed_chunks) if resume_from_checkpoint else 0
            
            if start_chunk > 0:
                print(f"🔄 Resuming from chunk {start_chunk + 1}")
            
            # Process remaining chunks
            for i in range(start_chunk, len(chunk_paths)):
                chunk_path = chunk_paths[i]
                chunk_result = self._process_chunk(chunk_path, i + 1)
                
                if chunk_result.success:
                    self.results.append(chunk_result)
                    print(f"✅ Chunk {i+1}/{len(chunk_paths)}: {len(chunk_result.chunks)} chunks, {len(chunk_result.markdown)} chars")
                else:
                    self.failed_chunks.append(chunk_result)
                    print(f"❌ Chunk {i+1}/{len(chunk_paths)}: {chunk_result.error}")
                
                # Save checkpoint every N chunks
                if (len(self.results) + len(self.failed_chunks)) % self.checkpoint_interval == 0:
                    self._save_checkpoint()
                
                # Small delay to be nice to the API
                time.sleep(1)
            
            # Final save
            return self._save_final_results(pdf_path, output_path)
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _save_final_results(self, pdf_path: str, output_path: Path) -> Dict[str, Any]:
        """Save final consolidated results"""
        # Combine results
        all_chunks = []
        all_markdown = []
        
        for result in self.results:
            all_chunks.extend(result.chunks)
            if result.markdown:
                all_markdown.append(result.markdown)
        
        combined_markdown = "\n\n".join(all_markdown)
        
        # Save results
        final_result = {
            "pdf_path": pdf_path,
            "total_chunks_processed": len(self.results),
            "failed_chunks": len(self.failed_chunks),
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
        
        # Clean up checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        
        print(f"\n🎉 Extraction complete!")
        print(f"   Processed chunks: {len(self.results)}/{len(self.results) + len(self.failed_chunks)}")
        print(f"   Content chunks: {len(all_chunks)}")
        print(f"   Markdown length: {len(combined_markdown):,} characters")
        print(f"   Results saved to: {knowledge_file}")
        
        self.print_usage_summary()
        
        return final_result
    
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
            "current_key_index": self.current_key_index,
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
            current = " ← CURRENT" if i == self.current_key_index else ""
            print(f"   Key {i+1}: {tracker.pages_processed} pages, ${tracker.total_cost:.2f} - {status}{current}")

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
    
    print(f"📊 Using ALL {len(api_keys)} API keys for maximum efficiency")
    
    # Find PDF
    pdf_path = "Python for Finance Mastering Data-Driven Finance (Yves Hilpisch) (Z-Library).pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    # Create robust extractor
    extractor = RobustChunkedExtractor(api_keys, checkpoint_interval=20)
    
    # Extract
    try:
        result = extractor.extract_pdf(pdf_path)
        print(f"\n🎉 Success! Extracted {len(result['chunks'])} content chunks")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Extraction interrupted by user")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")

if __name__ == "__main__":
    main() 