# API Key Management Guide for Document Extraction

This guide explains how to use the enhanced document extraction system with cost tracking, API key rotation, and progress management.

## Overview

The enhanced DocumentExtractor now supports:
- **Multiple API Key Management**: Rotate between multiple API keys automatically
- **Cost Tracking**: Monitor usage and costs in real-time
- **Progress Saving**: Save and resume extractions
- **Automatic Key Rotation**: Switch to new keys when limits are reached
- **Detailed Reporting**: Comprehensive usage summaries

## API Key Economics

Based on the LandingAI pricing model:
- **Cost per page**: $0.03
- **Free credits per key**: $10.00
- **Pages per key**: ~333 pages (10 ÷ 0.03)
- **Python for Finance book**: ~500 pages (estimated)

**To extract the full book, you'll need at least 2 API keys.**

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

1. Go to [LandingAI Platform](https://platform.landing.ai/)
2. Create an account and get your API keys
3. Each new key comes with $10 in free credits

### 3. Create API Keys File

```bash
# Create a template file
python research/managed_book_extraction.py --create-template api_keys.txt

# Edit the file and add your keys
nano api_keys.txt
```

Example `api_keys.txt`:
```
# LandingAI API Keys (one per line)
# Get your API keys from: https://platform.landing.ai/
# Each key gives you $10 in free credits (333 pages at $0.03/page)

sk-1234567890abcdef1234567890abcdef12345678
sk-abcdef1234567890abcdef1234567890abcdef12
sk-fedcba0987654321fedcba0987654321fedcba09
```

## Usage

### Basic Usage

```bash
# Extract with multiple API keys
python research/managed_book_extraction.py "Python for Finance.pdf" --api-keys-file api_keys.txt

# Extract with single API key from environment
export LANDINGAI_API_KEY="your_api_key_here"
python research/managed_book_extraction.py "Python for Finance.pdf"
```

### Advanced Usage

```bash
# Resume a previous extraction
python research/managed_book_extraction.py "Python for Finance.pdf" \
    --api-keys-file api_keys.txt \
    --extraction-id extraction_1234567890 \
    --resume

# Start fresh extraction (don't resume)
python research/managed_book_extraction.py "Python for Finance.pdf" \
    --api-keys-file api_keys.txt \
    --no-resume

# List all previous extractions
python research/managed_book_extraction.py --list-extractions

# Extract to specific directory
python research/managed_book_extraction.py "Python for Finance.pdf" \
    --api-keys-file api_keys.txt \
    --output-dir my_extraction_results
```

## Programmatic Usage

### Basic Example

```python
from core.data.document_extractor import DocumentExtractor

# Initialize with multiple API keys
api_keys = [
    "sk-1234567890abcdef1234567890abcdef12345678",
    "sk-abcdef1234567890abcdef1234567890abcdef12",
    "sk-fedcba0987654321fedcba0987654321fedcba09"
]

extractor = DocumentExtractor(api_keys=api_keys)

# Extract with automatic key rotation
knowledge = extractor.extract_from_pdf("Python for Finance.pdf")

# Print results
print(f"Extracted {len(knowledge.concepts)} concepts")
print(f"Total cost: ${knowledge.metadata['cost']:.2f}")
```

### Advanced Example with Progress Tracking

```python
from core.data.document_extractor import DocumentExtractor, ExtractionProgress

# Initialize extractor
extractor = DocumentExtractor(api_keys=api_keys)

# Check usage before starting
summary = extractor.get_total_usage_summary()
print(f"Available credits: ${summary['remaining_credits']:.2f}")
print(f"Available pages: {sum(t['remaining_pages'] for t in summary['trackers'])}")

try:
    # Extract with custom extraction ID
    knowledge = extractor.extract_from_pdf(
        pdf_path="Python for Finance.pdf",
        extraction_id="python_finance_book",
        resume=True
    )
    
    # Access detailed metadata
    print(f"Pages processed: {knowledge.metadata['pages_processed']}")
    print(f"API key used: {knowledge.metadata['api_key_used']}")
    print(f"Extraction ID: {knowledge.metadata['extraction_id']}")
    
except Exception as e:
    print(f"Extraction failed: {e}")
    
    # Check remaining credits
    final_summary = extractor.get_total_usage_summary()
    remaining = sum(t['remaining_pages'] for t in final_summary['trackers'])
    print(f"Remaining pages across all keys: {remaining}")
```

## Usage Tracking

### Real-time Monitoring

The system automatically tracks:
- Pages processed per API key
- Cost per API key
- Remaining credits per API key
- Overall usage summary

### Progress Files

Progress is saved to `extraction_progress/` directory:
- `extraction_1234567890.json`: Extraction progress
- `extraction_1234567890_usage.json`: Usage summary

### Usage Summary Example

```json
{
  "total_pages_processed": 500,
  "total_cost": 15.00,
  "remaining_credits": 15.00,
  "active_keys": 1,
  "exhausted_keys": 2,
  "current_key_index": 2,
  "trackers": [
    {
      "api_key": "sk-1234...",
      "pages_processed": 333,
      "total_cost": 10.00,
      "remaining_credits": 0.00,
      "is_exhausted": true
    },
    {
      "api_key": "sk-abcd...",
      "pages_processed": 167,
      "total_cost": 5.00,
      "remaining_credits": 5.00,
      "is_exhausted": false
    }
  ]
}
```

## Cost Management Strategies

### 1. Multi-Key Strategy
- Get 2-3 API keys for full book extraction
- System automatically rotates when limits reached
- Total cost: ~$15 for 500-page book

### 2. Incremental Extraction
- Extract specific chapters/sections
- Monitor costs per extraction
- Resume from where you left off

### 3. Budget Control
- Set up alerts when approaching limits
- Monitor usage in real-time
- Plan extractions based on available credits

## Error Handling

### Common Issues and Solutions

1. **"No API key credits available"**
   - Add more API keys to your file
   - Check usage summary for remaining credits
   - Wait for credits to refresh (if applicable)

2. **"All API keys exhausted"**
   - Get additional API keys
   - The system automatically tries key rotation
   - Check progress files to resume later

3. **"Extraction failed"**
   - Check API key validity
   - Verify PDF file exists and is readable
   - Check internet connection
   - Review error logs for specific issues

## Best Practices

### 1. API Key Management
- Keep API keys secure and private
- Use separate keys for different projects
- Monitor usage regularly
- Set up key rotation before starting large extractions

### 2. Cost Optimization
- Estimate PDF pages before extraction
- Use multiple keys for large documents
- Monitor real-time usage
- Save progress frequently

### 3. Error Recovery
- Always use extraction IDs for tracking
- Enable resume functionality
- Keep progress files for recovery
- Monitor API key status

## Troubleshooting

### Check System Status
```bash
# List all extractions and their status
python research/managed_book_extraction.py --list-extractions

# Check available credits
python -c "
from core.data.document_extractor import DocumentExtractor
extractor = DocumentExtractor()
print(extractor.get_total_usage_summary())
"
```

### Resume Failed Extraction
```bash
# Find your extraction ID from list-extractions
python research/managed_book_extraction.py --list-extractions

# Resume with the ID
python research/managed_book_extraction.py "Python for Finance.pdf" \
    --extraction-id extraction_1234567890 \
    --resume
```

### Reset and Start Fresh
```bash
# Start completely fresh (ignore previous progress)
python research/managed_book_extraction.py "Python for Finance.pdf" \
    --api-keys-file api_keys.txt \
    --no-resume
```

## Integration with Trading System

Once you've extracted the book knowledge, you can integrate it with your trading system:

```python
# Load extracted knowledge
from core.data.document_extractor import DocumentExtractor
import pickle

# Load knowledge from file
with open('extracted_knowledge/knowledge.pkl', 'rb') as f:
    knowledge = pickle.load(f)

# Search for specific concepts
extractor = DocumentExtractor()
risk_concepts = extractor.search_knowledge(knowledge, "risk management")
options_concepts = extractor.search_knowledge(knowledge, "options", category="options")

# Integrate with your trading strategies
for concept in risk_concepts:
    print(f"Risk concept: {concept.title}")
    print(f"Code examples: {len(concept.code_examples)}")
    print(f"Formulas: {len(concept.formulas)}")
```

## Support

For issues or questions:
1. Check the error logs in `logs/` directory
2. Review the progress files in `extraction_progress/`
3. Use the `--list-extractions` command to check status
4. Monitor API key usage with usage summary reports

The enhanced system provides comprehensive cost management and progress tracking to help you efficiently extract knowledge from your financial documents while managing API costs effectively. 