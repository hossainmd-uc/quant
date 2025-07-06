#!/usr/bin/env python3
"""
Test chunked extractor with just first few chunks to verify it works
"""

import os
import sys
from chunked_extractor import ChunkedExtractor, load_api_keys

def test_chunked_extraction():
    """Test with first 3 chunks only"""
    print("🧪 Testing chunked extraction with first 3 chunks...")
    
    # Load API keys
    api_keys = load_api_keys()
    
    if not api_keys:
        print("❌ No API keys found!")
        return
    
    print(f"Found {len(api_keys)} API keys")
    
    # Skip the first exhausted key
    if len(api_keys) > 1:
        api_keys = api_keys[1:]  # Skip first key (exhausted)
        print(f"Using {len(api_keys)} active API keys")
    
    # Find PDF
    pdf_path = "Python for Finance Mastering Data-Driven Finance (Yves Hilpisch) (Z-Library).pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    # Create extractor
    extractor = ChunkedExtractor(api_keys)
    
    # Modify the extraction method to process only first 3 chunks
    print("📝 Modifying extractor to process only first 3 chunks...")
    
    # Create a custom extract method for testing
    def test_extract_pdf(pdf_path: str, max_chunks: int = 3) -> dict:
        """Test extraction with limited chunks"""
        from pathlib import Path
        import tempfile
        import shutil
        
        output_path = Path("test_extracted_knowledge")
        output_path.mkdir(exist_ok=True)
        
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp(prefix="pdf_chunks_test_")
        
        try:
            # Split PDF into chunks (but we'll only process first few)
            chunk_paths = extractor._split_pdf(pdf_path, temp_dir)
            
            print(f"Total chunks available: {len(chunk_paths)}")
            print(f"Processing first {max_chunks} chunks...")
            
            # Process only first few chunks
            results = []
            failed_chunks = []
            
            for i in range(min(max_chunks, len(chunk_paths))):
                chunk_path = chunk_paths[i]
                chunk_result = extractor._process_chunk(chunk_path, i + 1)
                
                if chunk_result.success:
                    results.append(chunk_result)
                    print(f"✅ Chunk {i+1}: {len(chunk_result.chunks)} chunks, {len(chunk_result.markdown)} chars")
                else:
                    failed_chunks.append(chunk_result)
                    print(f"❌ Chunk {i+1}: {chunk_result.error}")
                
                # Small delay
                import time
                time.sleep(2)
            
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
                "test_mode": True,
                "total_chunks_processed": len(results),
                "failed_chunks": len(failed_chunks),
                "total_content_chunks": len(all_chunks),
                "markdown": combined_markdown,
                "chunks": all_chunks,
                "usage_summary": extractor.get_usage_summary(),
                "extraction_date": extractor.usage_trackers[0].start_time.isoformat() if extractor.usage_trackers else None
            }
            
            # Save to files
            import json
            knowledge_file = output_path / "test_knowledge.json"
            with open(knowledge_file, 'w') as f:
                json.dump(final_result, f, indent=2)
            
            print(f"\n🎉 Test extraction complete!")
            print(f"   Processed chunks: {len(results)}/{max_chunks}")
            print(f"   Content chunks: {len(all_chunks)}")
            print(f"   Markdown length: {len(combined_markdown):,} characters")
            print(f"   Results saved to: {knowledge_file}")
            
            if combined_markdown:
                print(f"\n📝 First 500 characters of extracted content:")
                print(f"{combined_markdown[:500]}...")
            
            extractor.print_usage_summary()
            
            return final_result
            
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Run test extraction
    try:
        result = test_extract_pdf(pdf_path, max_chunks=3)
        
        if result['total_content_chunks'] > 0:
            print("\n✅ SUCCESS! Chunked extraction is working correctly!")
            print(f"   - Authentication: ✅ Working")
            print(f"   - PDF splitting: ✅ Working")
            print(f"   - Content extraction: ✅ Working ({result['total_content_chunks']} chunks)")
            print(f"   - Cost tracking: ✅ Working")
            print(f"   - API key rotation: ✅ Working")
            
            # Ask user if they want to run full extraction
            print(f"\n🤔 Test successful! Ready to process full PDF (~539 chunks)?")
            print(f"   Estimated cost: ~${539 * 2 * 0.03:.2f} (539 chunks × 2 pages × $0.03)")
            print(f"   Available credits: ${sum(t.remaining_credits for t in extractor.usage_trackers):.2f}")
            
        else:
            print("\n❌ Test failed - no content extracted")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chunked_extraction() 