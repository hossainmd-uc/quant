#!/usr/bin/env python3
"""
Test script to verify LandingAI API authentication works correctly
"""

import os
import json
import base64
import requests
from pathlib import Path

def load_api_keys():
    """Load API keys from file"""
    try:
        with open("api_keys.txt", "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        api_keys = []
        for line in lines:
            if ':' in line:  # Valid API key format (username:password)
                api_keys.append(line)
                print(f"✅ Loaded API key: {line[:20]}...")
            else:
                print(f"❌ Invalid API key format: {line[:20]}...")
        
        return api_keys
    except FileNotFoundError:
        print("❌ api_keys.txt not found")
        return []

def test_api_authentication(api_key: str, pdf_path: str):
    """Test API authentication with a single API key"""
    url = "https://api.va.landing.ai/v1/tools/document-analysis"
    
    # Encode API key properly for Basic Auth
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
    
    print(f"Testing API key: {api_key[:20]}...")
    print(f"Encoded key: {encoded_key[:50]}...")
    
    try:
        with open(pdf_path, 'rb') as pdf_file:
            files = {"pdf": ("test.pdf", pdf_file, "application/pdf")}
            
            response = requests.post(
                url, 
                files=files, 
                data=data, 
                headers=headers,
                timeout=60  # Short timeout for test
            )
            
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                chunks = result.get("chunks", [])
                markdown = result.get("markdown", "")
                
                print(f"✅ SUCCESS!")
                print(f"   Chunks extracted: {len(chunks)}")
                print(f"   Markdown length: {len(markdown)}")
                print(f"   First 200 chars: {markdown[:200]}...")
                
                return True, result
            else:
                print(f"❌ FAILED - Status: {response.status_code}")
                print(f"   Response: {response.text}")
                return False, None
                
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, None

def main():
    """Test authentication with all API keys"""
    api_keys = load_api_keys()
    
    if not api_keys:
        print("❌ No API keys found!")
        return
    
    # Find the PDF file
    pdf_path = "Python for Finance Mastering Data-Driven Finance (Yves Hilpisch) (Z-Library).pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"Testing {len(api_keys)} API keys...")
    
    # Test each API key
    for i, api_key in enumerate(api_keys):
        print(f"\n--- Testing API Key {i+1}/{len(api_keys)} ---")
        success, result = test_api_authentication(api_key, pdf_path)
        
        if success:
            print(f"✅ API Key {i+1} works perfectly!")
            
            # Save sample result
            with open(f"test_result_key_{i+1}.json", "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"   Sample result saved to test_result_key_{i+1}.json")
            break
        else:
            print(f"❌ API Key {i+1} failed")
    
    print("\n🎯 Authentication test complete!")

if __name__ == "__main__":
    main() 