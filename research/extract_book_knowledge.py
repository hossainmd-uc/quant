#!/usr/bin/env python3
"""
Extract Knowledge from Python for Finance Book

This script uses the LandingAI agentic-doc API to extract structured knowledge
from the Python for Finance book and integrate it into our trading system.
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data.document_extractor import DocumentExtractor, extract_python_finance_book
from core.data.features import FeatureEngineer
from core.models.transformers import TransformerPredictor


def main():
    """Main extraction and integration process"""
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )
    
    logger.info("🚀 Starting Python for Finance Book Knowledge Extraction")
    logger.info("=" * 60)
    
    # Check if API key is set
    api_key = os.getenv("LANDINGAI_API_KEY")
    if not api_key:
        logger.error("❌ LANDINGAI_API_KEY not set!")
        logger.info("Get your API key from: https://landing.ai/agentic-document-extraction")
        logger.info("Then set it: export LANDINGAI_API_KEY=your_api_key")
        return False
    
    # Define file paths
    book_path = "Python for Finance Mastering Data-Driven Finance (Yves Hilpisch) (Z-Library).pdf"
    output_dir = "extracted_knowledge"
    
    # Check if book file exists
    if not Path(book_path).exists():
        logger.error(f"❌ Book file not found: {book_path}")
        return False
    
    try:
        # Extract knowledge from the book
        logger.info("📖 Extracting knowledge from Python for Finance book...")
        extractor = DocumentExtractor(api_key)
        
        knowledge = extractor.extract_from_pdf(
            pdf_path=book_path,
            output_dir=output_dir
        )
        
        logger.info(f"✅ Successfully extracted knowledge!")
        logger.info(f"   📚 Total concepts: {len(knowledge.concepts)}")
        logger.info(f"   📑 Total chapters: {len(knowledge.chapters)}")
        logger.info(f"   💻 Code snippets: {len(knowledge.code_snippets)}")
        logger.info(f"   📐 Formulas: {len(knowledge.formulas)}")
        logger.info(f"   🔑 Key terms: {len(knowledge.key_terms)}")
        
        # Analyze extracted content
        analyze_extracted_content(knowledge, extractor)
        
        # Generate integration recommendations
        generate_integration_recommendations(knowledge)
        
        # Create summary report
        create_summary_report(knowledge, output_dir)
        
        logger.info("🎉 Knowledge extraction completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}/")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during extraction: {e}")
        return False


def analyze_extracted_content(knowledge, extractor):
    """Analyze the extracted content"""
    logger.info("🔍 Analyzing extracted content...")
    
    # Categorize concepts
    categories = {}
    for concept in knowledge.concepts:
        category = concept.category
        if category not in categories:
            categories[category] = []
        categories[category].append(concept)
    
    logger.info("📊 Content Categories:")
    for category, concepts in categories.items():
        avg_confidence = sum(c.confidence for c in concepts) / len(concepts)
        logger.info(f"   {category}: {len(concepts)} concepts (avg confidence: {avg_confidence:.2f})")
    
    # Find high-confidence concepts
    high_confidence = [c for c in knowledge.concepts if c.confidence > 0.8]
    logger.info(f"⭐ High-confidence concepts: {len(high_confidence)}")
    
    # Search for specific trading concepts
    trading_concepts = extractor.search_knowledge(knowledge, "trading", "trading")
    risk_concepts = extractor.search_knowledge(knowledge, "risk", "risk_management")
    ml_concepts = extractor.search_knowledge(knowledge, "machine learning", "machine_learning")
    
    logger.info(f"📈 Trading concepts found: {len(trading_concepts)}")
    logger.info(f"⚠️  Risk management concepts: {len(risk_concepts)}")
    logger.info(f"🤖 Machine learning concepts: {len(ml_concepts)}")


def generate_integration_recommendations(knowledge):
    """Generate recommendations for integrating book knowledge"""
    logger.info("💡 Generating integration recommendations...")
    
    recommendations = []
    
    # Check for specific concepts that can enhance our system
    concept_categories = {}
    for concept in knowledge.concepts:
        category = concept.category
        if category not in concept_categories:
            concept_categories[category] = []
        concept_categories[category].append(concept)
    
    # Risk management enhancements
    if "risk_management" in concept_categories:
        recommendations.append({
            "area": "Risk Management",
            "description": "Integrate advanced risk metrics from the book",
            "concepts": len(concept_categories["risk_management"]),
            "implementation": "Add new risk calculation methods to core.risk module"
        })
    
    # Trading strategy enhancements
    if "trading" in concept_categories:
        recommendations.append({
            "area": "Trading Strategies",
            "description": "Implement trading strategies from the book",
            "concepts": len(concept_categories["trading"]),
            "implementation": "Create new strategy classes in core.strategies module"
        })
    
    # Portfolio theory enhancements
    if "portfolio_theory" in concept_categories:
        recommendations.append({
            "area": "Portfolio Optimization",
            "description": "Add advanced portfolio optimization techniques",
            "concepts": len(concept_categories["portfolio_theory"]),
            "implementation": "Enhance portfolio optimization in backtesting engine"
        })
    
    # Machine learning enhancements
    if "machine_learning" in concept_categories:
        recommendations.append({
            "area": "ML Models",
            "description": "Integrate ML techniques from the book",
            "concepts": len(concept_categories["machine_learning"]),
            "implementation": "Add new model architectures to core.models module"
        })
    
    # Data analysis enhancements
    if "data_analysis" in concept_categories:
        recommendations.append({
            "area": "Data Analysis",
            "description": "Improve data processing with book techniques",
            "concepts": len(concept_categories["data_analysis"]),
            "implementation": "Enhance feature engineering and data processing"
        })
    
    logger.info(f"📋 Generated {len(recommendations)} integration recommendations:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"   {i}. {rec['area']}: {rec['description']} ({rec['concepts']} concepts)")


def create_summary_report(knowledge, output_dir):
    """Create a comprehensive summary report"""
    logger.info("📝 Creating summary report...")
    
    report = {
        "extraction_summary": {
            "title": knowledge.title,
            "author": knowledge.author,
            "extraction_date": knowledge.extraction_date.isoformat(),
            "total_concepts": len(knowledge.concepts),
            "total_chapters": len(knowledge.chapters),
            "total_code_snippets": len(knowledge.code_snippets),
            "total_formulas": len(knowledge.formulas),
            "total_key_terms": len(knowledge.key_terms)
        },
        "category_breakdown": {},
        "high_confidence_concepts": [],
        "code_examples": knowledge.code_snippets[:10],  # First 10 code examples
        "key_formulas": knowledge.formulas[:10],  # First 10 formulas
        "integration_opportunities": []
    }
    
    # Category breakdown
    for concept in knowledge.concepts:
        category = concept.category
        if category not in report["category_breakdown"]:
            report["category_breakdown"][category] = {
                "count": 0,
                "avg_confidence": 0.0,
                "examples": []
            }
        
        report["category_breakdown"][category]["count"] += 1
        report["category_breakdown"][category]["examples"].append(concept.title)
    
    # Calculate average confidence for each category
    for category, data in report["category_breakdown"].items():
        category_concepts = [c for c in knowledge.concepts if c.category == category]
        data["avg_confidence"] = sum(c.confidence for c in category_concepts) / len(category_concepts)
        data["examples"] = data["examples"][:3]  # Keep only first 3 examples
    
    # High confidence concepts
    high_confidence = [c for c in knowledge.concepts if c.confidence > 0.8]
    report["high_confidence_concepts"] = [
        {
            "title": c.title,
            "category": c.category,
            "confidence": c.confidence,
            "page_number": c.page_number
        }
        for c in high_confidence[:10]  # Top 10
    ]
    
    # Save report
    report_path = Path(output_dir) / "extraction_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📊 Summary report saved to: {report_path}")


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("🎉 BOOK KNOWLEDGE EXTRACTION COMPLETED!")
        print("="*60)
        print("\nNext Steps:")
        print("1. Review extracted knowledge in extracted_knowledge/ directory")
        print("2. Examine extraction_report.json for detailed analysis")
        print("3. Use the extracted concepts to enhance your trading strategies")
        print("4. Integrate specific techniques into your ML models")
        print("5. Apply risk management insights to your portfolio optimization")
        print("\nThe extracted knowledge is now ready for integration into your trading system!")
    else:
        print("\n❌ Extraction failed. Please check the error messages above.")
        sys.exit(1) 