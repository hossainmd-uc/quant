#!/usr/bin/env python3
"""
Standalone Document Extraction Module using LandingAI agentic-doc with usage tracking and API key rotation

This is a standalone version that doesn't depend on other project modules.
"""

import os
import json
import base64
import requests
import pickle
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import time
from dataclasses import dataclass, asdict, field
import re

# Direct API implementation to bypass agentic-doc library bugs
AGENTIC_DOC_AVAILABLE = True

@dataclass
class ParsedDocument:
    """Simplified version of ParsedDocument for direct API usage"""
    chunks: List[Dict[str, Any]]
    markdown: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageTracker:
    """Track API usage and costs"""
    api_key: str
    pages_processed: int = 0
    cost_per_page: float = 0.03
    total_cost: float = 0.0
    free_credits: float = 10.0
    max_pages_per_key: int = 333  # $10 / $0.03 per page
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def remaining_credits(self) -> float:
        """Calculate remaining credits"""
        return max(0, self.free_credits - self.total_cost)
    
    @property
    def remaining_pages(self) -> int:
        """Calculate remaining pages this key can process"""
        return max(0, self.max_pages_per_key - self.pages_processed)
    
    @property
    def is_exhausted(self) -> bool:
        """Check if this API key is exhausted"""
        return self.remaining_pages <= 0
    
    def add_pages(self, pages: int) -> None:
        """Add processed pages and calculate cost"""
        self.pages_processed += pages
        self.total_cost = self.pages_processed * self.cost_per_page
        print(f"Added {pages} pages. Total: {self.pages_processed} pages, ${self.total_cost:.2f}")


@dataclass
class ExtractionProgress:
    """Track extraction progress for resuming"""
    pdf_path: str
    total_pages: int
    processed_pages: int = 0
    current_api_key_index: int = 0
    extraction_id: str = ""
    start_time: Optional[datetime] = None
    last_saved: Optional[datetime] = None
    completed: bool = False
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
        if not self.extraction_id:
            self.extraction_id = f"extraction_{int(time.time())}"
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage"""
        if self.total_pages == 0:
            return 0.0
        return (self.processed_pages / self.total_pages) * 100


@dataclass
class ExtractedConcept:
    """Financial concept extracted from document"""
    title: str
    content: str
    category: str
    page_number: Optional[int] = None
    confidence: float = 0.0
    code_examples: List[str] = field(default_factory=list)
    formulas: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)


@dataclass
class BookKnowledge:
    """Complete extracted knowledge from book"""
    title: str
    author: str
    extraction_date: datetime
    concepts: List[ExtractedConcept]
    chapters: Dict[str, List[ExtractedConcept]]
    code_snippets: List[str]
    formulas: List[str]
    key_terms: List[str]
    metadata: Dict[str, Any]


class DocumentExtractor:
    """Extract structured knowledge from financial documents with usage tracking"""
    
    def __init__(self, api_keys: Optional[List[str]] = None, cost_per_page: float = 0.03):
        """
        Initialize DocumentExtractor with multiple API keys for rotation
        
        Args:
            api_keys: List of API keys for rotation. If None, uses LANDINGAI_API_KEY env var
            cost_per_page: Cost per page processed (default: $0.03)
        """
        if api_keys is None:
            api_key = os.getenv("LANDINGAI_API_KEY")
            if not api_key:
                raise ValueError("LandingAI API key required. Set LANDINGAI_API_KEY environment variable or provide api_keys list")
            api_keys = [api_key]
        
        self.api_keys = api_keys
        self.cost_per_page = cost_per_page
        self.usage_trackers = [UsageTracker(api_key=key, cost_per_page=cost_per_page) for key in api_keys]
        self.current_key_index = 0
        self.progress_dir = Path("extraction_progress")
        self.progress_dir.mkdir(exist_ok=True)
        
        print(f"DocumentExtractor initialized with {len(api_keys)} API keys")
    
    def _encode_api_key(self, api_key: str) -> str:
        """Properly encode API key for Basic Auth"""
        # API key is already in username:password format, encode it
        return base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
    
    def _call_api_direct(self, pdf_path: str) -> Dict[str, Any]:
        """Make direct API call to LandingAI, bypassing buggy agentic-doc library"""
        url = "https://api.va.landing.ai/v1/tools/document-analysis"
        
        # Get current API key and encode it properly
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
        
        # Open PDF file
        with open(pdf_path, 'rb') as pdf_file:
            files = {"pdf": ("document.pdf", pdf_file, "application/pdf")}
            
            print(f"Making API call with key {self.current_key_index + 1}/{len(self.api_keys)}")
            print(f"API Key (first 20 chars): {current_api_key[:20]}...")
            
            try:
                response = requests.post(
                    url, 
                    files=files, 
                    data=data, 
                    headers=headers,
                    timeout=600
                )
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"API Error: {response.status_code}")
                    print(f"Response: {response.text}")
                    return {"error": f"API returned {response.status_code}", "response": response.text}
                    
            except Exception as e:
                print(f"Request failed: {e}")
                return {"error": str(e)}
    
    def get_total_usage_summary(self) -> Dict[str, Any]:
        """Get summary of usage across all API keys"""
        total_pages = sum(tracker.pages_processed for tracker in self.usage_trackers)
        total_cost = sum(tracker.total_cost for tracker in self.usage_trackers)
        remaining_credits = sum(tracker.remaining_credits for tracker in self.usage_trackers)
        
        return {
            "total_pages_processed": total_pages,
            "total_cost": total_cost,
            "remaining_credits": remaining_credits,
            "active_keys": len([t for t in self.usage_trackers if not t.is_exhausted]),
            "exhausted_keys": len([t for t in self.usage_trackers if t.is_exhausted]),
            "current_key_index": self.current_key_index,
            "trackers": [asdict(tracker) for tracker in self.usage_trackers]
        }
    
    def rotate_api_key(self) -> bool:
        """Rotate to next available API key"""
        # Find next non-exhausted key
        for i in range(len(self.api_keys)):
            next_index = (self.current_key_index + 1 + i) % len(self.api_keys)
            if not self.usage_trackers[next_index].is_exhausted:
                self.current_key_index = next_index
                os.environ["LANDINGAI_API_KEY"] = self.api_keys[self.current_key_index]
                print(f"Rotated to API key {self.current_key_index + 1}/{len(self.api_keys)}")
                return True
        
        print("❌ All API keys exhausted!")
        return False
    
    def _estimate_pdf_pages(self, pdf_path: str) -> int:
        """Estimate PDF page count"""
        try:
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    return len(reader.pages)
            except ImportError:
                # Try pypdf as alternative
                import pypdf
                with open(pdf_path, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    return len(reader.pages)
        except ImportError:
            print("PyPDF2/pypdf not available for page counting. Using estimate of 500 pages.")
            return 500
        except Exception as e:
            print(f"Could not count PDF pages: {e}. Using estimate of 500 pages.")
            return 500
    
    def extract_from_pdf(self, pdf_path: str, output_dir: str = "extracted_knowledge", 
                        extraction_id: Optional[str] = None, resume: bool = True) -> BookKnowledge:
        """Extract knowledge from PDF with usage tracking"""
        if not AGENTIC_DOC_AVAILABLE:
            raise ImportError("Install agentic-doc: pip install agentic-doc")
        
        # Start new extraction
        print(f"Starting extraction from {pdf_path}")
        
        # Get PDF page count
        pdf_pages = self._estimate_pdf_pages(pdf_path)
        print(f"PDF has approximately {pdf_pages} pages")
        
        # Check if we have enough credits
        available_pages = sum(tracker.remaining_pages for tracker in self.usage_trackers)
        if available_pages < pdf_pages:
            print(f"⚠️  PDF has ~{pdf_pages} pages but only {available_pages} pages available across all API keys")
            if available_pages == 0:
                raise ValueError("No API key credits available. Add more API keys or wait for credits to refresh.")
        
        return self._extract_with_tracking(pdf_path, pdf_pages, output_dir, extraction_id or f"extraction_{int(time.time())}")
    
    def _extract_with_tracking(self, pdf_path: str, pdf_pages: int, output_dir: str, extraction_id: str) -> BookKnowledge:
        """Extract with usage tracking using direct API calls"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Extract using direct API call
            print(f"Processing with API key {self.current_key_index + 1}/{len(self.api_keys)}")
            
            api_result = self._call_api_direct(pdf_path)
            
            if "error" in api_result:
                # Try rotating API key if we got an error
                if self.rotate_api_key():
                    print("Retrying with rotated API key...")
                    api_result = self._call_api_direct(pdf_path)
                    
                if "error" in api_result:
                    raise ValueError(f"API extraction failed: {api_result['error']}")
            
            # Convert API result to ParsedDocument format
            parsed_doc = ParsedDocument(
                chunks=api_result.get("chunks", []),
                markdown=api_result.get("markdown", ""),
                metadata=api_result.get("metadata", {})
            )
            
            pages_used = pdf_pages  # Assume all pages were processed
            
            # Update usage tracking
            current_tracker = self.usage_trackers[self.current_key_index]
            current_tracker.add_pages(pages_used)
            
            print(f"Parsed {len(parsed_doc.chunks)} chunks from {pages_used} pages")
            print(f"Markdown length: {len(parsed_doc.markdown)} characters")
            
            # Process content
            book_knowledge = self._process_content(parsed_doc, pdf_path)
            
            # Add usage metadata
            book_knowledge.metadata.update({
                "pages_processed": pages_used,
                "cost": current_tracker.total_cost,
                "api_key_used": self.current_key_index,
                "extraction_id": extraction_id,
                "usage_summary": self.get_total_usage_summary()
            })
            
            # Save results
            self._save_knowledge(book_knowledge, output_path)
            self._print_usage_summary()
            
            return book_knowledge
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            raise
    
    def _process_content(self, parsed_doc: ParsedDocument, pdf_path: str) -> BookKnowledge:
        """Process extracted content into structured knowledge"""
        concepts = []
        chapters = {}
        code_snippets = []
        formulas = []
        key_terms = []
        current_chapter = "Introduction"
        
        # Process markdown content directly since agentic-doc gives us markdown
        markdown_content = parsed_doc.markdown
        if markdown_content:
            # Split content into sections
            sections = markdown_content.split('\n\n')
            for section in sections:
                if section.strip():
                    # Check if it's a chapter title
                    if self._is_chapter_title(section):
                        current_chapter = section.strip()
                        chapters[current_chapter] = []
                    else:
                        # Extract concept from section
                        concept = self._extract_concept_from_text(section, current_chapter)
                        if concept:
                            concepts.append(concept)
                            if current_chapter not in chapters:
                                chapters[current_chapter] = []
                            chapters[current_chapter].append(concept)
                    
                    # Extract code, formulas, terms from all sections
                    code_snippets.extend(self._extract_code(section))
                    formulas.extend(self._extract_formulas(section))
                    key_terms.extend(self._extract_terms(section))
        
        return BookKnowledge(
            title="Python for Finance: Mastering Data-Driven Finance",
            author="Yves Hilpisch",
            extraction_date=datetime.now(),
            concepts=concepts,
            chapters=chapters,
            code_snippets=list(set(code_snippets)),
            formulas=list(set(formulas)),
            key_terms=list(set(key_terms)),
            metadata={
                "source_file": pdf_path,
                "markdown_length": len(parsed_doc.markdown) if parsed_doc.markdown else 0,
                "extraction_method": "direct API"
            }
        )
    
    def _is_chapter_title(self, text: str) -> bool:
        """Check if text is a chapter title"""
        text = text.strip()
        patterns = [r'^Chapter\s+\d+', r'^\d+\.\s+[A-Z]', r'^[A-Z][A-Za-z\s]+$']
        return any(re.match(p, text) for p in patterns) and 5 <= len(text) <= 100
    
    def _extract_concept_from_text(self, content: str, chapter: str) -> Optional[ExtractedConcept]:
        """Extract financial concept from text content"""
        content = content.strip()
        if len(content) < 50:
            return None
        
        category = self._categorize_content(content)
        title = content.split('.')[0][:100] + ("..." if len(content) > 100 else "")
        
        return ExtractedConcept(
            title=title,
            content=content,
            category=category,
            page_number=None,
            confidence=self._calculate_confidence(content, category),
            code_examples=self._extract_code(content),
            formulas=self._extract_formulas(content),
            key_terms=self._extract_terms(content)
        )
    
    def _extract_concept(self, chunk, chapter: str) -> Optional[ExtractedConcept]:
        """Extract financial concept from text chunk"""
        content = chunk.content.strip()
        if len(content) < 50:
            return None
        
        category = self._categorize_content(content)
        title = content.split('.')[0][:100] + ("..." if len(content) > 100 else "")
        
        return ExtractedConcept(
            title=title,
            content=content,
            category=category,
            page_number=getattr(chunk, 'page_number', None),
            confidence=self._calculate_confidence(content, category),
            code_examples=self._extract_code(content),
            formulas=self._extract_formulas(content),
            key_terms=self._extract_terms(content)
        )
    
    def _categorize_content(self, content: str) -> str:
        """Categorize content by financial keywords"""
        content_lower = content.lower()
        categories = {
            "risk_management": ["risk", "var", "volatility", "drawdown", "sharpe"],
            "options": ["option", "call", "put", "strike", "black-scholes"],
            "derivatives": ["derivative", "future", "forward", "swap"],
            "portfolio_theory": ["portfolio", "markowitz", "efficient frontier"],
            "time_series": ["time series", "autoregressive", "moving average"],
            "machine_learning": ["machine learning", "neural network", "regression"],
            "trading": ["trading", "strategy", "backtest", "execution"],
            "data_analysis": ["pandas", "numpy", "matplotlib", "data"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        return "general"
    
    def _extract_code(self, content: str) -> List[str]:
        """Extract Python code snippets"""
        patterns = [
            r'```python\n(.*?)\n```',
            r'import\s+[\w.]+',
            r'from\s+[\w.]+\s+import',
            r'def\s+\w+\([^)]*\):'
        ]
        
        code = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            code.extend(matches)
        
        return [c.strip() for c in code if c.strip()]
    
    def _extract_formulas(self, content: str) -> List[str]:
        """Extract mathematical formulas"""
        patterns = [
            r'\$([^$]+)\$',
            r'[A-Z]\s*=\s*[^,.\n]+',
            r'σ|μ|π|α|β|γ|θ|λ|Δ|∑|∏'
        ]
        
        formulas = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            formulas.extend(matches)
        
        return [f.strip() for f in formulas if f.strip()]
    
    def _extract_terms(self, content: str) -> List[str]:
        """Extract key financial terms"""
        terms = {
            'volatility', 'sharpe ratio', 'beta', 'alpha', 'var', 'correlation',
            'portfolio', 'diversification', 'hedge', 'arbitrage', 'derivative',
            'option', 'monte carlo', 'black-scholes', 'markowitz', 'capm'
        }
        
        content_lower = content.lower()
        return [term for term in terms if term in content_lower]
    
    def _calculate_confidence(self, content: str, category: str) -> float:
        """Calculate confidence score"""
        score = 0.5
        if len(content) > 100: score += 0.1
        if len(content) > 500: score += 0.1
        if category != "general": score += 0.2
        if any(kw in content.lower() for kw in ['import', 'def', 'class']): score += 0.1
        return min(score, 1.0)
    
    def _save_knowledge(self, knowledge: BookKnowledge, output_path: Path):
        """Save extracted knowledge"""
        # JSON format
        with open(output_path / "knowledge.json", 'w', encoding='utf-8') as f:
            json.dump(asdict(knowledge), f, indent=2, default=str)
        
        # Pickle format
        with open(output_path / "knowledge.pkl", 'wb') as f:
            pickle.dump(knowledge, f)
        
        print(f"Knowledge saved to {output_path}")
    
    def _print_usage_summary(self) -> None:
        """Print usage summary"""
        summary = self.get_total_usage_summary()
        
        print(f"\n{'='*50}")
        print(f"EXTRACTION USAGE SUMMARY")
        print(f"{'='*50}")
        print(f"Total Pages Processed: {summary['total_pages_processed']}")
        print(f"Total Cost: ${summary['total_cost']:.2f}")
        print(f"Remaining Credits: ${summary['remaining_credits']:.2f}")
        print(f"Active API Keys: {summary['active_keys']}/{len(self.api_keys)}")
        
        print(f"\nPER-KEY BREAKDOWN:")
        for i, tracker_data in enumerate(summary['trackers']):
            status = "EXHAUSTED" if tracker_data['is_exhausted'] else "ACTIVE"
            print(f"  Key {i+1}: {tracker_data['pages_processed']} pages, "
                  f"${tracker_data['total_cost']:.2f}, "
                  f"${tracker_data['remaining_credits']:.2f} remaining [{status}]")
        
        print(f"{'='*50}\n")


def main():
    """Main test function"""
    print("🔧 STANDALONE DOCUMENT EXTRACTION TEST")
    print("=" * 50)
    
    # Load API keys
    try:
        with open('api_keys.txt', 'r') as f:
            api_keys = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    api_keys.append(line)
    except FileNotFoundError:
        print("❌ api_keys.txt not found!")
        return
    
    print(f"Loaded {len(api_keys)} API keys")
    
    # Initialize extractor
    try:
        extractor = DocumentExtractor(api_keys=api_keys)
        print("✅ DocumentExtractor initialized successfully!")
        
        # Show usage summary
        summary = extractor.get_total_usage_summary()
        print(f"\n📊 INITIAL USAGE SUMMARY:")
        print(f"  Total Available Credits: ${summary['remaining_credits']:.2f}")
        total_pages = sum(t.get('remaining_pages', 333) for t in summary['trackers'])
        print(f"  Total Available Pages: {total_pages}")
        print(f"  Active Keys: {summary['active_keys']}/{len(api_keys)}")
        
        # Ask user if they want to proceed
        print(f"\n⚠️  COST WARNING:")
        print(f"   The Python for Finance book is ~500 pages")
        print(f"   At $0.03 per page, this will cost ~$15.00")
        print(f"   You have {len(api_keys)} API keys with ${summary['remaining_credits']:.2f} total credits")
        
        response = input("\nDo you want to proceed with the extraction? (y/n): ")
        if response.lower() in ['y', 'yes']:
            pdf_path = "Python for Finance Mastering Data-Driven Finance (Yves Hilpisch) (Z-Library).pdf"
            
            if not os.path.exists(pdf_path):
                print(f"❌ PDF file not found: {pdf_path}")
                return
            
            print(f"\n🚀 Starting extraction...")
            knowledge = extractor.extract_from_pdf(pdf_path)
            
            print(f"\n✅ EXTRACTION COMPLETED!")
            print(f"📖 Extracted {len(knowledge.concepts)} concepts")
            print(f"📝 Found {len(knowledge.code_snippets)} code snippets")
            print(f"🔢 Identified {len(knowledge.formulas)} formulas")
            print(f"💡 Extracted {len(knowledge.key_terms)} key terms")
            print(f"📊 Processed {len(knowledge.chapters)} chapters")
        else:
            print("Test completed. System is ready for extraction!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 