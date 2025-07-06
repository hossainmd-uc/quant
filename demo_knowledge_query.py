#!/usr/bin/env python3
"""
Demo: Financial Knowledge Query System

This script demonstrates how to use the knowledge query system to search and explain
financial concepts from the extracted book knowledge.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.data.knowledge_query_system import FinancialKnowledgeQuerySystem, search_knowledge


def demo_basic_search():
    """Demonstrate basic search functionality"""
    print("🔍 DEMO: Basic Knowledge Search")
    print("=" * 60)
    
    try:
        # Initialize the knowledge system
        knowledge_system = FinancialKnowledgeQuerySystem()
        
        if not knowledge_system.knowledge:
            print("❌ No knowledge loaded!")
            return
            
        print(f"📚 Loaded {len(knowledge_system.knowledge.concepts)} concepts")
        print(f"📂 Categories available: {list(knowledge_system.get_categories().keys())}")
        print()
        
        # Demo searches
        demo_queries = [
            "volatility",
            "risk management", 
            "Black-Scholes",
            "machine learning",
            "portfolio optimization"
        ]
        
        for query in demo_queries:
            print(f"\n🔍 Searching for: '{query}'")
            print("-" * 40)
            
            results = knowledge_system.search(query, max_results=2)
            
            if results.concepts:
                print(f"✅ Found {results.total_matches} matches")
                print(f"📊 Categories: {results.categories}")
                print(f"⭐ Confidence range: {results.confidence_stats['min']:.2f} - {results.confidence_stats['max']:.2f}")
                
                # Show first result
                concept = results.concepts[0]
                print(f"\n📖 Top Result: {concept.title}")
                print(f"📂 Category: {concept.category}")
                print(f"📝 Summary: {concept.content[:200]}...")
                
                if concept.code_examples:
                    print(f"💻 Has {len(concept.code_examples)} code examples")
                if concept.formulas:
                    print(f"🔢 Has {len(concept.formulas)} formulas")
            else:
                print("❌ No results found")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have extracted knowledge available at 'extracted_knowledge/knowledge.pkl'")


def demo_detailed_explanation():
    """Demonstrate detailed concept explanation"""
    print("\n\n📚 DEMO: Detailed Concept Explanation")
    print("=" * 60)
    
    try:
        knowledge_system = FinancialKnowledgeQuerySystem()
        
        # Search for a specific concept
        results = knowledge_system.search("volatility", max_results=1)
        
        if results.concepts:
            concept = results.concepts[0]
            print("🎯 Generating detailed explanation...")
            print()
            
            detailed_explanation = knowledge_system.explain_concept_detailed(concept)
            print(detailed_explanation)
        else:
            print("❌ No volatility concepts found")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_category_search():
    """Demonstrate searching by category"""
    print("\n\n📂 DEMO: Search by Category")
    print("=" * 60)
    
    try:
        knowledge_system = FinancialKnowledgeQuerySystem()
        
        categories = knowledge_system.get_categories()
        print("Available categories:")
        for cat, count in categories.items():
            print(f"  • {cat.replace('_', ' ').title()}: {count} concepts")
        
        # Search within specific category
        if 'risk_management' in categories:
            print(f"\n🎯 Searching within 'risk_management' category...")
            results = knowledge_system.search("", category="risk_management", max_results=3)
            
            print(f"Found {len(results.concepts)} risk management concepts:")
            for i, concept in enumerate(results.concepts, 1):
                print(f"  {i}. {concept.title} (confidence: {concept.confidence:.2f})")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def interactive_search():
    """Interactive search mode"""
    print("\n\n💬 INTERACTIVE SEARCH MODE")
    print("=" * 60)
    print("Enter search queries or 'quit' to exit")
    print("Examples: 'volatility', 'options pricing', 'machine learning'")
    print()
    
    try:
        knowledge_system = FinancialKnowledgeQuerySystem()
        
        while True:
            query = input("🔍 Search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"\n🔍 Searching for: '{query}'...")
            explanation = knowledge_system.search_and_explain(query, max_results=1)
            print(explanation)
            print("\n" + "="*80 + "\n")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")


def knowledge_stats():
    """Show knowledge base statistics"""
    print("\n\n📊 KNOWLEDGE BASE STATISTICS")
    print("=" * 60)
    
    try:
        knowledge_system = FinancialKnowledgeQuerySystem()
        knowledge = knowledge_system.knowledge
        
        if not knowledge:
            print("❌ No knowledge loaded!")
            return
        
        print(f"📚 Book: {knowledge.title}")
        print(f"✍️  Author: {knowledge.author}")
        print(f"📅 Extracted: {knowledge.extraction_date}")
        print()
        
        print(f"📝 Total Concepts: {len(knowledge.concepts)}")
        print(f"📂 Total Chapters: {len(knowledge.chapters)}")
        print(f"💻 Code Snippets: {len(knowledge.code_snippets)}")
        print(f"🔢 Formulas: {len(knowledge.formulas)}")
        print(f"🔑 Key Terms: {len(knowledge.key_terms)}")
        print()
        
        # Category breakdown
        categories = knowledge_system.get_categories()
        print("📊 Concepts by Category:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {cat.replace('_', ' ').title()}: {count}")
        
        # Confidence distribution
        confidences = [c.confidence for c in knowledge.concepts]
        if confidences:
            print(f"\n⭐ Confidence Statistics:")
            print(f"  • Average: {sum(confidences) / len(confidences):.3f}")
            print(f"  • Range: {min(confidences):.3f} - {max(confidences):.3f}")
            
            # High confidence concepts
            high_confidence = [c for c in knowledge.concepts if c.confidence > 0.8]
            print(f"  • High confidence (>0.8): {len(high_confidence)} concepts")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main demo function"""
    print("🚀 FINANCIAL KNOWLEDGE QUERY SYSTEM DEMO")
    print("=" * 80)
    
    # Check if knowledge file exists
    if not Path("extracted_knowledge/knowledge.pkl").exists():
        print("❌ Knowledge file not found!")
        print("Please run the knowledge extraction first:")
        print("  python research/extract_book_knowledge.py")
        return
    
    print("Choose a demo:")
    print("1. Basic Search Demo")
    print("2. Detailed Explanation Demo")
    print("3. Category Search Demo")
    print("4. Interactive Search")
    print("5. Knowledge Statistics")
    print("6. Run All Demos")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        demo_basic_search()
    elif choice == "2":
        demo_detailed_explanation()
    elif choice == "3":
        demo_category_search()
    elif choice == "4":
        interactive_search()
    elif choice == "5":
        knowledge_stats()
    elif choice == "6":
        demo_basic_search()
        demo_detailed_explanation()
        demo_category_search()
        knowledge_stats()
    else:
        print("Invalid choice. Running basic demo...")
        demo_basic_search()


if __name__ == "__main__":
    main() 