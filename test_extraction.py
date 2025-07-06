#!/usr/bin/env python3
"""
Simple test script for document extraction
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import directly from the document extractor module
from core.data.document_extractor import DocumentExtractor, ExtractionProgress

def test_api_keys():
    """Test API keys initialization"""
    print("Testing API Keys...")
    
    # Load API keys from file
    with open('api_keys.txt', 'r') as f:
        api_keys = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                api_keys.append(line)
    
    print(f"Loaded {len(api_keys)} API keys")
    
    # Initialize extractor
    try:
        extractor = DocumentExtractor(api_keys=api_keys)
        print("✅ DocumentExtractor initialized successfully!")
        
        # Show usage summary
        summary = extractor.get_total_usage_summary()
        print(f"\n📊 INITIAL USAGE SUMMARY:")
        print(f"  Total Available Credits: ${summary['remaining_credits']:.2f}")
        print(f"  Total Available Pages: {sum(t['remaining_pages'] for t in summary['trackers'])}")
        print(f"  Active Keys: {summary['active_keys']}/{len(api_keys)}")
        
        return extractor
        
    except Exception as e:
        print(f"❌ Error initializing DocumentExtractor: {e}")
        return None


def test_extraction(extractor):
    """Test document extraction"""
    pdf_path = "Python for Finance Mastering Data-Driven Finance (Yves Hilpisch) (Z-Library).pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"\n🚀 Starting extraction test...")
    print(f"PDF: {pdf_path}")
    
    try:
        # Extract with tracking
        knowledge = extractor.extract_from_pdf(
            pdf_path=pdf_path,
            output_dir="test_extraction_output",
            extraction_id="test_extraction"
        )
        
        print(f"\n✅ EXTRACTION COMPLETED!")
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
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("🔧 DOCUMENT EXTRACTION TEST")
    print("=" * 50)
    
    # Test API keys
    extractor = test_api_keys()
    if not extractor:
        return
    
    # Ask user if they want to proceed with full extraction
    print(f"\n⚠️  COST WARNING:")
    print(f"   The Python for Finance book is ~500 pages")
    print(f"   At $0.03 per page, this will cost ~$15.00")
    print(f"   You have 3 API keys with $30.00 total credits")
    
    response = input("\nDo you want to proceed with the full extraction? (y/n): ")
    if response.lower() in ['y', 'yes']:
        test_extraction(extractor)
    else:
        print("Test completed. Ready for full extraction when you're ready!")


if __name__ == "__main__":
    main() 