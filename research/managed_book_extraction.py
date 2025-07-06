#!/usr/bin/env python3
"""
Managed Book Extraction with API Key Rotation

This script demonstrates the enhanced document extraction system with:
- Multiple API key management
- Usage tracking and cost monitoring
- Automatic key rotation when limits are reached
- Progress saving and resuming
- Detailed reporting

Usage:
    python managed_book_extraction.py --help
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
from loguru import logger

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from core.data.document_extractor import DocumentExtractor, ExtractionProgress


def load_api_keys_from_file(file_path: str) -> List[str]:
    """Load API keys from a text file (one per line)"""
    api_keys = []
    with open(file_path, 'r') as f:
        for line in f:
            key = line.strip()
            if key and not key.startswith('#'):  # Skip comments
                api_keys.append(key)
    return api_keys


def create_api_keys_template(file_path: str) -> None:
    """Create a template API keys file"""
    template = """# LandingAI API Keys (one per line)
# Get your API keys from: https://platform.landing.ai/
# Each key gives you $10 in free credits (333 pages at $0.03/page)

# your_first_api_key_here
# your_second_api_key_here
# your_third_api_key_here

# Example:
# sk-1234567890abcdef1234567890abcdef12345678
# sk-abcdef1234567890abcdef1234567890abcdef12
"""
    with open(file_path, 'w') as f:
        f.write(template)
    print(f"Created API keys template at: {file_path}")
    print("Please edit this file and add your API keys, then run the script again.")


def extract_with_management(
    pdf_path: str,
    api_keys: List[str],
    output_dir: str = "extracted_knowledge",
    extraction_id: Optional[str] = None,
    resume: bool = True
) -> None:
    """Extract book knowledge with full management"""
    
    print(f"\n{'='*60}")
    print(f"MANAGED BOOK EXTRACTION")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"API Keys: {len(api_keys)} keys loaded")
    print(f"Output Directory: {output_dir}")
    print(f"Extraction ID: {extraction_id or 'Auto-generated'}")
    print(f"Resume: {'Yes' if resume else 'No'}")
    print(f"{'='*60}\n")
    
    # Initialize extractor
    extractor = DocumentExtractor(api_keys=api_keys)
    
    # Show initial usage summary
    print("INITIAL API KEY STATUS:")
    summary = extractor.get_total_usage_summary()
    print(f"  Total Available Credits: ${summary['remaining_credits']:.2f}")
    print(f"  Total Available Pages: {sum(t['remaining_pages'] for t in summary['trackers'])}")
    print(f"  Active Keys: {summary['active_keys']}/{len(api_keys)}\n")
    
    try:
        # Extract with tracking
        knowledge = extractor.extract_from_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            extraction_id=extraction_id,
            resume=resume
        )
        
        print(f"\n✅ EXTRACTION COMPLETED SUCCESSFULLY!")
        print(f"📖 Extracted {len(knowledge.concepts)} concepts")
        print(f"📝 Found {len(knowledge.code_snippets)} code snippets")
        print(f"🔢 Identified {len(knowledge.formulas)} formulas")
        print(f"💡 Extracted {len(knowledge.key_terms)} key terms")
        print(f"📊 Processed {len(knowledge.chapters)} chapters")
        
        # Show final usage summary
        final_summary = extractor.get_total_usage_summary()
        print(f"\n💰 FINAL COST SUMMARY:")
        print(f"  Pages Processed: {final_summary['total_pages_processed']}")
        print(f"  Total Cost: ${final_summary['total_cost']:.2f}")
        print(f"  Remaining Credits: ${final_summary['remaining_credits']:.2f}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        print(f"\n❌ EXTRACTION FAILED: {e}")
        
        # Show what we can do with remaining credits
        summary = extractor.get_total_usage_summary()
        remaining_pages = sum(t['remaining_pages'] for t in summary['trackers'])
        if remaining_pages > 0:
            print(f"\n💡 You still have {remaining_pages} pages available across your API keys.")
            print(f"   Consider adding more API keys or waiting for credits to refresh.")


def list_extractions(extractor: DocumentExtractor) -> None:
    """List all previous extractions"""
    extractions = extractor.list_extractions()
    
    if not extractions:
        print("No previous extractions found.")
        return
    
    print(f"\n{'='*60}")
    print(f"PREVIOUS EXTRACTIONS")
    print(f"{'='*60}")
    
    for i, extraction in enumerate(extractions, 1):
        print(f"{i}. Total Pages: {extraction['total_pages_processed']}")
        print(f"   Cost: ${extraction['total_cost']:.2f}")
        print(f"   Active Keys: {extraction['active_keys']}")
        print(f"   Exhausted Keys: {extraction['exhausted_keys']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract knowledge from PDF with API key management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract with single API key from environment
  python managed_book_extraction.py book.pdf
  
  # Extract with multiple API keys from file
  python managed_book_extraction.py book.pdf --api-keys-file keys.txt
  
  # Resume a previous extraction
  python managed_book_extraction.py book.pdf --extraction-id extraction_1234567890 --resume
  
  # List previous extractions
  python managed_book_extraction.py --list-extractions
  
  # Create API keys template file
  python managed_book_extraction.py --create-template keys.txt
        """
    )
    
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to PDF file to extract"
    )
    
    parser.add_argument(
        "--api-keys-file",
        type=str,
        help="File containing API keys (one per line)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted_knowledge",
        help="Output directory for extracted knowledge (default: extracted_knowledge)"
    )
    
    parser.add_argument(
        "--extraction-id",
        type=str,
        help="Extraction ID for resuming or tracking"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume previous extraction if available (default: True)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume previous extraction"
    )
    
    parser.add_argument(
        "--list-extractions",
        action="store_true",
        help="List all previous extractions"
    )
    
    parser.add_argument(
        "--create-template",
        type=str,
        help="Create API keys template file"
    )
    
    args = parser.parse_args()
    
    # Handle template creation
    if args.create_template:
        create_api_keys_template(args.create_template)
        return
    
    # Handle listing extractions
    if args.list_extractions:
        extractor = DocumentExtractor()
        list_extractions(extractor)
        return
    
    # Validate PDF path
    if not args.pdf_path:
        parser.error("PDF path is required unless using --list-extractions or --create-template")
    
    if not os.path.exists(args.pdf_path):
        print(f"❌ Error: PDF file not found: {args.pdf_path}")
        return
    
    # Load API keys
    api_keys = []
    
    if args.api_keys_file:
        if not os.path.exists(args.api_keys_file):
            print(f"❌ Error: API keys file not found: {args.api_keys_file}")
            print(f"Use --create-template {args.api_keys_file} to create a template")
            return
        api_keys = load_api_keys_from_file(args.api_keys_file)
    else:
        # Use environment variable
        api_key = os.getenv("LANDINGAI_API_KEY")
        if not api_key:
            print("❌ Error: No API key found!")
            print("Either set LANDINGAI_API_KEY environment variable or use --api-keys-file")
            print("Use --create-template keys.txt to create a template file")
            return
        api_keys = [api_key]
    
    if not api_keys:
        print("❌ Error: No valid API keys found in file")
        return
    
    # Handle resume flag
    resume = args.resume and not args.no_resume
    
    # Extract with management
    extract_with_management(
        pdf_path=args.pdf_path,
        api_keys=api_keys,
        output_dir=args.output_dir,
        extraction_id=args.extraction_id,
        resume=resume
    )


if __name__ == "__main__":
    main() 