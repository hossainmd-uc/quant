#!/usr/bin/env python3
"""
Simple test to debug the chunked extraction
"""

import os
import json
import base64
import requests
import tempfile
import shutil
import PyPDF2

def simple_test():
    """Simple test of one chunk"""
    print("🔍 Simple test - processing one chunk...")
    
    # API key (use the second one, first is exhausted)
    api_key = "pm4r010t74nu13rpkarzt:FDvHk7YZWoAdhi7Qxqd6EQKPat2ObfWs"
    
    # PDF path
    pdf_path = "Python for Finance Mastering Data-Driven Finance (Yves Hilpisch) (Z-Library).pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    # Create a 2-page chunk from the PDF
    print("📄 Creating 2-page chunk...")
    
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(reader.pages)
        print(f"Total pages: {total_pages}")
        
        # Create a small chunk (pages 10-11, should have content)
        writer = PyPDF2.PdfWriter()
        writer.add_page(reader.pages[10])  # Page 11 (0-indexed)
        writer.add_page(reader.pages[11])  # Page 12
        
        # Save chunk
        chunk_path = "test_chunk.pdf"
        with open(chunk_path, 'wb') as chunk_file:
            writer.write(chunk_file)
        
        print(f"✅ Created chunk: {chunk_path}")
    
    # Now test the API call
    print("🌐 Testing API call...")
    
    url = "https://api.va.landing.ai/v1/tools/document-analysis"
    encoded_key = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
    
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
            files = {"pdf": ("test_chunk.pdf", chunk_file, "application/pdf")}
            
            print("📡 Making API call...")
            response = requests.post(url, files=files, data=data, headers=headers, timeout=120)
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract content from the correct API response format
                extracted_chunks = []
                markdown_parts = []
                
                if "data" in result and "pages" in result["data"]:
                    for page in result["data"]["pages"]:
                        if "chunks" in page:
                            for chunk in page["chunks"]:
                                if "caption" in chunk and chunk["caption"]:
                                    extracted_chunks.append(chunk)
                                    markdown_parts.append(chunk["caption"])
                
                combined_markdown = "\n".join(markdown_parts)
                
                print(f"✅ SUCCESS!")
                print(f"   Chunks extracted: {len(extracted_chunks)}")
                print(f"   Markdown length: {len(combined_markdown)}")
                
                if combined_markdown:
                    print(f"\n📝 First 500 chars of markdown:")
                    print(combined_markdown[:500])
                    print("...")
                
                # Save result
                with open("test_result.json", "w") as f:
                    json.dump(result, f, indent=2)
                
                print(f"\n✅ Result saved to test_result.json")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
            print(f"🧹 Cleaned up {chunk_path}")

if __name__ == "__main__":
    simple_test() 