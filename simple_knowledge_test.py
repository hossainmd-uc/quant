#!/usr/bin/env python3
"""
Simple test of the Knowledge Query System

Tests the financial knowledge query functionality directly.
"""

import sys
import pickle
from pathlib import Path

# Simple test to see if we can load the knowledge
def test_knowledge_loading():
    """Test loading extracted knowledge"""
    print("🔍 Testing Knowledge Loading...")
    
    knowledge_path = Path("extracted_knowledge/knowledge.pkl")
    
    if not knowledge_path.exists():
        print("❌ Knowledge file not found!")
        print("Available files in extracted_knowledge/:")
        if Path("extracted_knowledge").exists():
            for file in Path("extracted_knowledge").iterdir():
                print(f"  - {file.name}")
        return False
    
    try:
        with open(knowledge_path, 'rb') as f:
            knowledge = pickle.load(f)
        
        print(f"✅ Successfully loaded knowledge!")
        print(f"📚 Title: {knowledge.title}")
        print(f"✍️  Author: {knowledge.author}")
        print(f"📝 Concepts: {len(knowledge.concepts)}")
        print(f"📂 Chapters: {len(knowledge.chapters)}")
        print(f"💻 Code snippets: {len(knowledge.code_snippets)}")
        print(f"🔢 Formulas: {len(knowledge.formulas)}")
        print(f"🔑 Key terms: {len(knowledge.key_terms)}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading knowledge: {e}")
        return False

def simple_search_demo(knowledge):
    """Demo simple search without the full query system"""
    print("\n🔍 Simple Search Demo...")
    
    search_term = "volatility"
    print(f"Searching for: '{search_term}'")
    
    matching_concepts = []
    for concept in knowledge.concepts:
        if search_term.lower() in concept.title.lower() or search_term.lower() in concept.content.lower():
            matching_concepts.append(concept)
    
    print(f"Found {len(matching_concepts)} concepts mentioning '{search_term}':")
    
    for i, concept in enumerate(matching_concepts[:3], 1):  # Show top 3
        print(f"\n{i}. 📖 {concept.title}")
        print(f"   📂 Category: {concept.category}")
        print(f"   ⭐ Confidence: {concept.confidence:.2f}")
        print(f"   📝 Preview: {concept.content[:150]}...")
        
        if concept.code_examples:
            print(f"   💻 Code examples: {len(concept.code_examples)}")
        if concept.formulas:
            print(f"   🔢 Formulas: {len(concept.formulas)}")

def show_categories(knowledge):
    """Show available categories"""
    print("\n📂 Available Categories:")
    
    categories = {}
    for concept in knowledge.concepts:
        category = concept.category
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {category.replace('_', ' ').title()}: {count} concepts")

def show_sample_concepts(knowledge):
    """Show some sample concepts"""
    print("\n📚 Sample Concepts:")
    
    # Show highest confidence concepts
    high_confidence = sorted(knowledge.concepts, key=lambda x: x.confidence, reverse=True)[:5]
    
    for i, concept in enumerate(high_confidence, 1):
        print(f"\n{i}. 📖 {concept.title}")
        print(f"   📂 {concept.category} | ⭐ {concept.confidence:.2f}")
        print(f"   📝 {concept.content[:100]}...")

def main():
    """Main test function"""
    print("🚀 Simple Knowledge Query Test")
    print("=" * 50)
    
    # Test loading
    if not test_knowledge_loading():
        return
    
    # Load knowledge for demos
    with open("extracted_knowledge/knowledge.pkl", 'rb') as f:
        knowledge = pickle.load(f)
    
    # Run demos
    show_categories(knowledge)
    show_sample_concepts(knowledge)
    simple_search_demo(knowledge)
    
    print("\n✅ Knowledge query test completed!")
    print("\nThe knowledge base is ready for integration!")

if __name__ == "__main__":
    main() 