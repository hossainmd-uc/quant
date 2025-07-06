#!/usr/bin/env python3
"""
Generate Complete Knowledge Catalog

This script creates a comprehensive table of contents document showing all
extracted financial concepts organized by category, with descriptions and
sample questions to help users know what they can ask about.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from core.data.robust_knowledge_system import RobustFinancialKnowledgeSystem
except ImportError:
    print("❌ Could not import robust knowledge system")
    print("Creating a standalone version...")
    
    # If import fails, let's create a simpler standalone version
    import json
    import re
    from collections import defaultdict
    
    def create_simple_catalog():
        """Create a simple catalog directly from the JSON file"""
        knowledge_path = "extracted_knowledge/knowledge.json"
        
        if not Path(knowledge_path).exists():
            return "❌ Knowledge file not found!"
        
        with open(knowledge_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        
        # Categorize chunks
        categories = defaultdict(list)
        
        for chunk in chunks:
            # Handle different content formats
            content = chunk.get('caption', '')
            if isinstance(content, dict):
                # If caption is a dict, extract text from it
                content = str(content)
            elif isinstance(content, list):
                # If caption is a list, join the elements
                content = ' '.join(str(item) for item in content)
            elif content is None:
                content = ''
            else:
                content = str(content)
            
            # Skip empty content
            if not content or len(content.strip()) < 10:
                continue
                
            category = categorize_content(content)
            categories[category].append({
                'title': content[:100] + '...' if len(content) > 100 else content,
                'content': content,
                'order': chunk.get('order', 0),
                'label': chunk.get('label', 'unknown')
            })
        
        # Generate catalog
        catalog = []
        catalog.append("# 📚 Python for Finance - Complete Knowledge Catalog")
        catalog.append("=" * 80)
        catalog.append("")
        catalog.append(f"**Total Chunks**: {len(chunks)}")
        catalog.append(f"**Categories Found**: {len(categories)}")
        catalog.append("")
        
        for category, items in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
            catalog.append(f"## 📂 {category.replace('_', ' ').title()}")
            catalog.append(f"**Concepts**: {len(items)}")
            catalog.append("")
            
            # Show top items
            catalog.append("**Key Topics:**")
            for item in sorted(items, key=lambda x: len(x['content']), reverse=True)[:10]:
                label = item.get('label', 'text')
                catalog.append(f"  • {item['title']} [{label}]")
            catalog.append("")
            
            # Show content types
            labels = [item.get('label', 'unknown') for item in items]
            label_counts = {label: labels.count(label) for label in set(labels)}
            catalog.append("**Content Types:**")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                catalog.append(f"  • {label}: {count}")
            catalog.append("")
            
            # Sample questions
            questions = generate_sample_questions(category)
            catalog.append("**What You Can Ask:**")
            for question in questions:
                catalog.append(f"  ❓ {question}")
            catalog.append("")
            catalog.append("-" * 60)
            catalog.append("")
        
        return "\n".join(catalog)
    
    def categorize_content(content):
        """Simple categorization"""
        if not isinstance(content, str):
            content = str(content)
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['option', 'call', 'put', 'black-scholes', 'derivative']):
            return "options_derivatives"
        elif any(term in content_lower for term in ['risk', 'var', 'volatility', 'sharpe']):
            return "risk_management"
        elif any(term in content_lower for term in ['portfolio', 'markowitz', 'optimization']):
            return "portfolio_theory"
        elif any(term in content_lower for term in ['machine learning', 'neural', 'model']):
            return "machine_learning"
        elif any(term in content_lower for term in ['trading', 'strategy', 'backtest']):
            return "trading_strategies"
        elif any(term in content_lower for term in ['pandas', 'numpy', 'data', 'analysis']):
            return "data_analysis"
        elif any(term in content_lower for term in ['stochastic', 'brownian', 'monte carlo']):
            return "mathematical_finance"
        elif any(term in content_lower for term in ['bond', 'yield', 'interest rate']):
            return "fixed_income"
        else:
            return "general_concepts"
    
    def generate_sample_questions(category):
        """Generate sample questions for each category"""
        questions = {
            'options_derivatives': [
                "How does the Black-Scholes model work?",
                "What are the Greeks in options trading?",
                "How do I implement option pricing?",
                "What is implied volatility?"
            ],
            'risk_management': [
                "How do I calculate Value-at-Risk?",
                "What is the Sharpe ratio?",
                "How do I measure portfolio risk?",
                "What are risk-adjusted returns?"
            ],
            'portfolio_theory': [
                "How does Modern Portfolio Theory work?",
                "What is the efficient frontier?",
                "How do I optimize portfolios?",
                "What is asset allocation?"
            ],
            'machine_learning': [
                "How can I use ML for trading?",
                "What are the best algorithms for finance?",
                "How do I build predictive models?",
                "What is feature engineering?"
            ],
            'trading_strategies': [
                "How do I backtest strategies?",
                "What are momentum strategies?",
                "How do I implement mean reversion?",
                "What are performance metrics?"
            ],
            'data_analysis': [
                "How do I use pandas for finance?",
                "What are time series techniques?",
                "How do I visualize financial data?",
                "How do I clean market data?"
            ],
            'mathematical_finance': [
                "What is geometric Brownian motion?",
                "How do Monte Carlo simulations work?",
                "What are stochastic processes?",
                "How do I model random variables?"
            ],
            'fixed_income': [
                "How do bond pricing models work?",
                "What is duration and convexity?",
                "How do I model yield curves?",
                "What are credit risk models?"
            ]
        }
        return questions.get(category, [f"What is {category.replace('_', ' ')}?"])


def main():
    """Generate the knowledge catalog"""
    print("🚀 Generating Complete Knowledge Catalog...")
    print("=" * 60)
    
    try:
        # Try to use the robust system first
        if 'RobustFinancialKnowledgeSystem' in globals():
            system = RobustFinancialKnowledgeSystem()
            print("✅ Loaded robust knowledge system")
            
            catalog_content = system.generate_knowledge_catalog()
            
            # Additional statistics
            print(f"\n📊 Knowledge Statistics:")
            print(f"  • Total Concepts: {len(system.concepts)}")
            print(f"  • Categories: {len(system.category_index)}")
            print(f"  • Total Code Examples: {sum(len(c.code_examples) for c in system.concepts)}")
            print(f"  • Total Formulas: {sum(len(c.formulas) for c in system.concepts)}")
        else:
            raise ImportError("RobustFinancialKnowledgeSystem not available")
        
    except Exception as e:
        print(f"⚠️  Robust system failed ({e}), using simple version...")
        catalog_content = create_simple_catalog()
    
    # Save catalog to file
    output_file = "KNOWLEDGE_CATALOG.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(catalog_content)
    
    print(f"\n✅ Knowledge catalog generated: {output_file}")
    print(f"📄 File size: {len(catalog_content):,} characters")
    
    # Show preview
    print("\n📋 Preview:")
    print("-" * 40)
    lines = catalog_content.split('\n')
    for line in lines[:30]:  # Show first 30 lines
        print(line)
    
    if len(lines) > 30:
        print("...")
        print(f"({len(lines) - 30} more lines)")
    
    print("\n🎉 Complete! You can now use this catalog to:")
    print("  • Discover what financial concepts are available")
    print("  • Find specific topics to ask questions about")
    print("  • Understand the scope of the extracted knowledge")
    print("  • Get sample questions for each category")
    
    return catalog_content


if __name__ == "__main__":
    main() 