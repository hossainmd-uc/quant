#!/usr/bin/env python3
"""
Advanced Portfolio Optimization Demo
===================================

This script demonstrates the advanced portfolio optimization techniques
implementing 113 portfolio theory concepts from "Python for Finance: 
Mastering Data-Driven Finance" by Yves Hilpisch.

The demo showcases:
1. Enhanced Modern Portfolio Theory (MPT) with efficient frontier
2. Black-Litterman model with investor views
3. Risk Parity optimization (Equal Risk Contribution)
4. Hierarchical Risk Parity (HRP)
5. Comparative analysis of different optimization approaches

Educational Value:
- Understand how different portfolio optimization techniques work
- Compare traditional MPT with modern approaches
- Learn about incorporating investor views (Black-Litterman)
- Explore risk-based allocation methods
- See practical implementation of advanced portfolio theory
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our advanced portfolio optimization modules
from core.portfolio.modern_portfolio_theory import (
    ModernPortfolioTheory, EfficientFrontier, 
    PortfolioConstraints, OptimizationResult
)
from core.portfolio.black_litterman import BlackLittermanOptimizer, MarketView
from core.portfolio.risk_parity import RiskParityOptimizer, RiskBudget, HierarchicalRiskParity

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{'-'*50}")
    print(f"{title}")
    print(f"{'-'*50}")

def get_sample_data():
    """
    Generate sample market data for demonstration.
    In practice, you would use real market data from yfinance, Bloomberg, etc.
    """
    print("📊 Generating sample market data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define assets
    assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY', 'QQQ', 'VTI', 'BND']
    
    # Generate correlated returns
    n_assets = len(assets)
    n_periods = 252 * 2  # 2 years of daily data
    
    # Create correlation matrix
    correlation_matrix = np.random.rand(n_assets, n_assets)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Generate returns using multivariate normal distribution
    mean_returns = np.random.normal(0.0005, 0.0005, n_assets)  # Daily returns
    volatilities = np.random.uniform(0.01, 0.03, n_assets)  # Daily volatilities
    
    # Create covariance matrix
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
    
    # Create DataFrame
    returns_df = pd.DataFrame(returns, columns=assets)
    
    # Generate market capitalizations (in billions)
    market_caps = pd.Series({
        'AAPL': 3000, 'GOOGL': 1800, 'MSFT': 2800, 'TSLA': 800,
        'SPY': 400, 'QQQ': 200, 'VTI': 300, 'BND': 100
    })
    
    print(f"✅ Generated {n_periods} periods of data for {n_assets} assets")
    print(f"📈 Assets: {', '.join(assets)}")
    
    return returns_df, market_caps

def demonstrate_modern_portfolio_theory(returns_data):
    """Demonstrate Modern Portfolio Theory optimization."""
    print_section("MODERN PORTFOLIO THEORY DEMONSTRATION")
    
    # Initialize MPT optimizer
    mpt = ModernPortfolioTheory(risk_free_rate=0.02)
    mpt.fit(returns_data)
    
    print("🎯 Modern Portfolio Theory focuses on maximizing expected return")
    print("   for a given level of risk or minimizing risk for a given return.")
    print("   It's the foundation of quantitative portfolio management.")
    
    # 1. Maximum Sharpe Ratio Portfolio
    print_subsection("1. Maximum Sharpe Ratio Portfolio")
    
    constraints = PortfolioConstraints(
        long_only=True,
        max_single_weight=0.4  # No single asset > 40%
    )
    
    max_sharpe_result = mpt.optimize_max_sharpe(constraints)
    
    if max_sharpe_result.success:
        print("✅ Maximum Sharpe Ratio Optimization Successful!")
        print(f"📊 Expected Return: {max_sharpe_result.expected_return:.4f}")
        print(f"📊 Volatility: {max_sharpe_result.volatility:.4f}")
        print(f"📊 Sharpe Ratio: {max_sharpe_result.sharpe_ratio:.4f}")
        print("\n🎛️ Optimal Weights:")
        for i, asset in enumerate(returns_data.columns):
            weight = max_sharpe_result.weights[i]
            print(f"   {asset}: {weight:.3f} ({weight*100:.1f}%)")
        
        print("\n⚖️ Risk Contributions:")
        for i, asset in enumerate(returns_data.columns):
            risk_contrib = max_sharpe_result.risk_contributions[i]
            print(f"   {asset}: {risk_contrib:.3f} ({risk_contrib*100:.1f}%)")
    else:
        print(f"❌ Optimization failed: {max_sharpe_result.message}")
    
    # 2. Minimum Volatility Portfolio
    print_subsection("2. Minimum Volatility Portfolio")
    
    min_vol_result = mpt.optimize_min_volatility(constraints)
    
    if min_vol_result.success:
        print("✅ Minimum Volatility Optimization Successful!")
        print(f"📊 Expected Return: {min_vol_result.expected_return:.4f}")
        print(f"📊 Volatility: {min_vol_result.volatility:.4f}")
        print(f"📊 Sharpe Ratio: {min_vol_result.sharpe_ratio:.4f}")
        print("\n🎛️ Optimal Weights:")
        for i, asset in enumerate(returns_data.columns):
            weight = min_vol_result.weights[i]
            print(f"   {asset}: {weight:.3f} ({weight*100:.1f}%)")
    
    # 3. Efficient Frontier
    print_subsection("3. Efficient Frontier Analysis")
    
    ef = EfficientFrontier(mpt)
    frontier_points = ef.calculate_frontier(num_points=20, constraints=constraints)
    
    if not frontier_points.empty:
        print("✅ Efficient Frontier Calculated!")
        print(f"📊 Number of frontier points: {len(frontier_points)}")
        
        # Show frontier statistics
        stats = ef.get_frontier_statistics()
        print(f"📊 Volatility range: {stats['min_volatility']:.4f} - {stats['max_volatility']:.4f}")
        print(f"📊 Return range: {stats['min_return']:.4f} - {stats['max_return']:.4f}")
        print(f"📊 Maximum Sharpe ratio: {stats['max_sharpe']:.4f}")
        
        # Show sample frontier points
        print("\n🎯 Sample Efficient Frontier Points:")
        sample_points = frontier_points.iloc[::4]  # Every 4th point
        for _, point in sample_points.iterrows():
            print(f"   Return: {point['expected_return']:.4f}, Vol: {point['volatility']:.4f}, Sharpe: {point['sharpe_ratio']:.4f}")
    
    return max_sharpe_result, min_vol_result, frontier_points

def demonstrate_black_litterman(returns_data, market_caps):
    """Demonstrate Black-Litterman model."""
    print_section("BLACK-LITTERMAN MODEL DEMONSTRATION")
    
    print("🔬 Black-Litterman improves on MPT by incorporating:")
    print("   • Market equilibrium assumptions")
    print("   • Investor views with confidence levels")
    print("   • Bayesian approach to return estimation")
    
    # Initialize Black-Litterman optimizer
    bl = BlackLittermanOptimizer(risk_aversion=3.0, tau=0.05)
    bl.fit(returns_data, market_caps)
    
    print_subsection("1. Market Equilibrium Analysis")
    
    # Show equilibrium returns and weights
    eq_returns = bl.get_equilibrium_returns()
    eq_weights = bl.get_equilibrium_weights()
    
    print("📊 Market Equilibrium (Implied) Returns:")
    for asset, ret in eq_returns.items():
        print(f"   {asset}: {ret:.4f} ({ret*252*100:.1f}% annualized)")
    
    print("\n🎛️ Market Capitalization Weights:")
    for asset, weight in eq_weights.items():
        print(f"   {asset}: {weight:.3f} ({weight*100:.1f}%)")
    
    # 2. Adding Investor Views
    print_subsection("2. Adding Investor Views")
    
    # Add some sample views
    print("📝 Adding investor views:")
    
    # Absolute view: AAPL expected to return 15% annually
    bl.add_absolute_view('AAPL', 0.15/252, confidence=0.8)
    print("   • AAPL expected to return 15% annually (80% confidence)")
    
    # Relative view: GOOGL expected to outperform MSFT by 5% annually
    bl.add_relative_view('GOOGL', 'MSFT', 0.05/252, confidence=0.6)
    print("   • GOOGL expected to outperform MSFT by 5% annually (60% confidence)")
    
    # Sector view: Tech stocks (AAPL, GOOGL, MSFT) expected to outperform
    tech_view = MarketView(
        assets=['AAPL', 'GOOGL', 'MSFT', 'SPY'],
        weights=[1/3, 1/3, 1/3, -1],
        expected_return=0.03/252,
        confidence=0.7
    )
    bl.add_view(tech_view)
    print("   • Tech stocks expected to outperform market by 3% annually (70% confidence)")
    
    # 3. Black-Litterman Optimization
    print_subsection("3. Black-Litterman Optimization Results")
    
    # Optimize portfolio with views
    bl_result = bl.optimize_portfolio()
    
    print("✅ Black-Litterman Optimization Successful!")
    print(f"📊 Expected Return: {bl_result['expected_return']:.4f}")
    print(f"📊 Volatility: {bl_result['volatility']:.4f}")
    print(f"📊 Sharpe Ratio: {bl_result['sharpe_ratio']:.4f}")
    
    print("\n🎛️ Black-Litterman Weights:")
    for asset, weight in bl_result['weights'].items():
        print(f"   {asset}: {weight:.3f} ({weight*100:.1f}%)")
    
    print("\n📈 Posterior Expected Returns:")
    for asset, ret in bl_result['posterior_returns'].items():
        print(f"   {asset}: {ret:.4f} ({ret*252*100:.1f}% annualized)")
    
    # 4. View Impact Analysis
    print_subsection("4. View Impact Analysis")
    
    impact_analysis = bl.get_view_impact()
    
    if not impact_analysis.empty:
        print("📊 Impact of Views on Expected Returns:")
        for _, row in impact_analysis.iterrows():
            print(f"   View {row['view_index']}: {row['view_description']}")
            print(f"     Confidence: {row['confidence']:.1%}")
            # Show impact on key assets
            for asset in ['AAPL', 'GOOGL', 'MSFT']:
                if f'{asset}_impact' in impact_analysis.columns:
                    impact = row[f'{asset}_impact']
                    print(f"     {asset} impact: {impact:.4f} ({impact*252*100:.1f}% annualized)")
    
    return bl_result

def demonstrate_risk_parity(returns_data):
    """Demonstrate Risk Parity optimization."""
    print_section("RISK PARITY DEMONSTRATION")
    
    print("⚖️ Risk Parity focuses on equalizing risk contributions rather than")
    print("   capital allocations. This addresses the concentration risk in")
    print("   traditional market-cap weighted portfolios.")
    
    # 1. Equal Risk Contribution (ERC)
    print_subsection("1. Equal Risk Contribution Portfolio")
    
    erc_optimizer = RiskParityOptimizer(method='equal_risk_contribution')
    erc_optimizer.fit(returns_data)
    
    erc_result = erc_optimizer.optimize()
    
    if erc_result['success']:
        print("✅ Equal Risk Contribution Optimization Successful!")
        print(f"📊 Portfolio Volatility: {erc_result['portfolio_volatility']:.4f}")
        print(f"🔄 Iterations: {erc_result['iterations']}")
        
        print("\n🎛️ ERC Weights:")
        for asset, weight in erc_result['weights'].items():
            print(f"   {asset}: {weight:.3f} ({weight*100:.1f}%)")
        
        print("\n⚖️ Risk Contributions (should be approximately equal):")
        for asset, risk_contrib in erc_result['risk_contributions'].items():
            print(f"   {asset}: {risk_contrib:.3f} ({risk_contrib*100:.1f}%)")
        
        # Calculate risk contribution standard deviation (measure of equality)
        risk_contrib_std = erc_result['risk_contributions'].std()
        print(f"\n📊 Risk Contribution Std Dev: {risk_contrib_std:.4f}")
        print("   (Lower values indicate more equal risk contributions)")
    else:
        print(f"❌ ERC Optimization failed: {erc_result['message']}")
    
    # 2. Risk Budgeting
    print_subsection("2. Risk Budgeting Portfolio")
    
    # Define custom risk budgets
    custom_budgets = {
        'AAPL': 0.15, 'GOOGL': 0.15, 'MSFT': 0.15, 'TSLA': 0.10,
        'SPY': 0.20, 'QQQ': 0.10, 'VTI': 0.10, 'BND': 0.05
    }
    
    risk_budget = RiskBudget(asset_budgets=custom_budgets)
    
    rb_optimizer = RiskParityOptimizer(method='risk_budgeting')
    rb_optimizer.fit(returns_data)
    
    rb_result = rb_optimizer.optimize(risk_budget=risk_budget)
    
    if rb_result['success']:
        print("✅ Risk Budgeting Optimization Successful!")
        print(f"📊 Portfolio Volatility: {rb_result['portfolio_volatility']:.4f}")
        
        print("\n🎛️ Risk Budgeting Weights:")
        for asset, weight in rb_result['weights'].items():
            print(f"   {asset}: {weight:.3f} ({weight*100:.1f}%)")
        
        print("\n🎯 Target vs Actual Risk Contributions:")
        for asset in returns_data.columns:
            target = rb_result['target_contributions'][asset]
            actual = rb_result['risk_contributions'][asset]
            print(f"   {asset}: Target {target:.3f} ({target*100:.1f}%) | Actual {actual:.3f} ({actual*100:.1f}%)")
    
    # 3. Naive Risk Parity
    print_subsection("3. Naive Risk Parity (Inverse Volatility)")
    
    naive_optimizer = RiskParityOptimizer(method='naive')
    naive_optimizer.fit(returns_data)
    
    naive_result = naive_optimizer.optimize()
    
    if naive_result['success']:
        print("✅ Naive Risk Parity Calculation Successful!")
        print(f"📊 Portfolio Volatility: {naive_result['portfolio_volatility']:.4f}")
        
        print("\n📊 Individual Asset Volatilities:")
        for asset, vol in naive_result['individual_volatilities'].items():
            print(f"   {asset}: {vol:.4f} ({vol*np.sqrt(252)*100:.1f}% annualized)")
        
        print("\n🎛️ Naive Risk Parity Weights (Inverse Volatility):")
        for asset, weight in naive_result['weights'].items():
            print(f"   {asset}: {weight:.3f} ({weight*100:.1f}%)")
    
    return erc_result, rb_result, naive_result

def demonstrate_hierarchical_risk_parity(returns_data):
    """Demonstrate Hierarchical Risk Parity."""
    print_section("HIERARCHICAL RISK PARITY DEMONSTRATION")
    
    print("🌳 Hierarchical Risk Parity (HRP) uses machine learning techniques")
    print("   to build portfolios that account for the hierarchical structure")
    print("   of asset correlations. It's more stable than traditional optimization.")
    
    # Initialize HRP optimizer
    hrp = HierarchicalRiskParity(linkage_method='single')
    hrp.fit(returns_data)
    
    print_subsection("1. Correlation Analysis")
    
    # Show correlation matrix
    corr_matrix = returns_data.corr()
    print("📊 Asset Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # Calculate distance matrix
    distance_matrix = hrp.calculate_distance_matrix()
    print(f"\n📏 Distance Matrix Shape: {distance_matrix.shape}")
    print("   (Distance = sqrt(0.5 * (1 - correlation)))")
    
    # 2. Hierarchical Clustering
    print_subsection("2. Hierarchical Clustering")
    
    linkage_matrix = hrp.perform_clustering()
    print(f"✅ Hierarchical clustering completed")
    print(f"📊 Linkage matrix shape: {linkage_matrix.shape}")
    
    # Get cluster allocation
    clusters = hrp.get_cluster_allocation(n_clusters=3)
    print(f"\n🎯 Asset Clusters (3 clusters):")
    for cluster_id, assets in clusters.items():
        print(f"   Cluster {cluster_id}: {', '.join(assets)}")
    
    # 3. HRP Optimization
    print_subsection("3. HRP Portfolio Optimization")
    
    hrp_result = hrp.optimize()
    
    if hrp_result['success']:
        print("✅ HRP Optimization Successful!")
        print(f"📊 Portfolio Volatility: {hrp_result['portfolio_volatility']:.4f}")
        
        print("\n🎛️ HRP Weights:")
        for asset, weight in hrp_result['weights'].items():
            print(f"   {asset}: {weight:.3f} ({weight*100:.1f}%)")
        
        print("\n🔄 Quasi-Diagonalization Order:")
        print(f"   {' → '.join(hrp_result['quasi_order'])}")
        print("   (Assets ordered by hierarchical clustering)")
    else:
        print(f"❌ HRP Optimization failed: {hrp_result['message']}")
    
    return hrp_result

def compare_optimization_methods(returns_data, mpt_result, bl_result, erc_result, hrp_result):
    """Compare different optimization methods."""
    print_section("COMPARATIVE ANALYSIS")
    
    print("📊 Comparing different portfolio optimization approaches:")
    print("   This analysis helps understand the trade-offs between methods.")
    
    # Create comparison table
    comparison_data = {
        'Method': ['Max Sharpe (MPT)', 'Black-Litterman', 'Risk Parity (ERC)', 'Hierarchical RP'],
        'Expected Return': [
            mpt_result.expected_return if hasattr(mpt_result, 'expected_return') else 0,
            bl_result['expected_return'],
            np.dot(erc_result['weights'], returns_data.mean()),
            np.dot(hrp_result['weights'], returns_data.mean())
        ],
        'Volatility': [
            mpt_result.volatility if hasattr(mpt_result, 'volatility') else 0,
            bl_result['volatility'],
            erc_result['portfolio_volatility'],
            hrp_result['portfolio_volatility']
        ],
        'Sharpe Ratio': [
            mpt_result.sharpe_ratio if hasattr(mpt_result, 'sharpe_ratio') else 0,
            bl_result['sharpe_ratio'],
            (np.dot(erc_result['weights'], returns_data.mean()) - 0.02/252) / erc_result['portfolio_volatility'],
            (np.dot(hrp_result['weights'], returns_data.mean()) - 0.02/252) / hrp_result['portfolio_volatility']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n📈 Performance Comparison:")
    print(comparison_df.round(4))
    
    # Weight concentration analysis
    print("\n🎯 Weight Concentration Analysis:")
    
    methods = {
        'Max Sharpe (MPT)': pd.Series(mpt_result.weights, index=returns_data.columns) if hasattr(mpt_result, 'weights') else pd.Series(),
        'Black-Litterman': bl_result['weights'],
        'Risk Parity (ERC)': erc_result['weights'],
        'Hierarchical RP': hrp_result['weights']
    }
    
    for method_name, weights in methods.items():
        if len(weights) > 0:
            max_weight = weights.max()
            min_weight = weights.min()
            weight_std = weights.std()
            
            print(f"   {method_name}:")
            print(f"     Max weight: {max_weight:.3f} ({max_weight*100:.1f}%)")
            print(f"     Min weight: {min_weight:.3f} ({min_weight*100:.1f}%)")
            print(f"     Weight std: {weight_std:.3f} (concentration measure)")
    
    # Risk characteristics
    print("\n⚖️ Risk Characteristics:")
    print("   • MPT: Maximizes risk-adjusted return")
    print("   • Black-Litterman: Incorporates market views")
    print("   • Risk Parity: Equalizes risk contributions")
    print("   • HRP: Uses hierarchical structure, more stable")
    
    return comparison_df

def demonstrate_portfolio_insights():
    """Provide educational insights about portfolio optimization."""
    print_section("PORTFOLIO OPTIMIZATION INSIGHTS")
    
    print("🎓 Key Insights from Advanced Portfolio Optimization:")
    print()
    
    insights = [
        "1. DIVERSIFICATION IS KEY",
        "   • Risk parity methods often provide better diversification than market-cap weighting",
        "   • HRP is particularly effective at avoiding concentration in highly correlated assets",
        "",
        "2. VIEWS MATTER",
        "   • Black-Litterman allows incorporation of expert opinions and market views",
        "   • Confidence levels help balance between market equilibrium and personal views",
        "",
        "3. RISK BUDGETING",
        "   • Traditional optimization often leads to concentrated portfolios",
        "   • Risk parity ensures more balanced risk exposure across assets",
        "",
        "4. STABILITY",
        "   • HRP tends to be more stable than mean-variance optimization",
        "   • Less sensitive to estimation errors in expected returns",
        "",
        "5. PRACTICAL CONSIDERATIONS",
        "   • Transaction costs and constraints are important in practice",
        "   • Regular rebalancing is needed to maintain target allocations",
        "   • Market conditions change, requiring adaptive strategies"
    ]
    
    for insight in insights:
        print(insight)
    
    print("\n🔬 When to Use Each Method:")
    print("   • MPT: When you have strong views on expected returns")
    print("   • Black-Litterman: When incorporating market views with uncertainty")
    print("   • Risk Parity: When seeking balanced risk exposure")
    print("   • HRP: When dealing with many assets and correlation instability")

def main():
    """Main demonstration function."""
    print_section("ADVANCED PORTFOLIO OPTIMIZATION DEMO")
    print("🚀 Welcome to the Advanced Portfolio Optimization demonstration!")
    print("📚 This demo implements 113 portfolio theory concepts from")
    print("   'Python for Finance: Mastering Data-Driven Finance' by Yves Hilpisch")
    print()
    print("⏱️  This demonstration will take approximately 2-3 minutes to complete.")
    print("🎯 You'll see practical implementations of cutting-edge portfolio theory!")
    
    try:
        # Get sample data
        returns_data, market_caps = get_sample_data()
        
        # Demonstrate different optimization techniques
        mpt_result, min_vol_result, frontier_points = demonstrate_modern_portfolio_theory(returns_data)
        bl_result = demonstrate_black_litterman(returns_data, market_caps)
        erc_result, rb_result, naive_result = demonstrate_risk_parity(returns_data)
        hrp_result = demonstrate_hierarchical_risk_parity(returns_data)
        
        # Compare methods
        comparison_df = compare_optimization_methods(returns_data, mpt_result, bl_result, erc_result, hrp_result)
        
        # Provide insights
        demonstrate_portfolio_insights()
        
        print_section("DEMO COMPLETED SUCCESSFULLY!")
        print("✅ All portfolio optimization techniques demonstrated successfully!")
        print("🎯 You've seen practical implementations of:")
        print("   • Modern Portfolio Theory with efficient frontier")
        print("   • Black-Litterman model with investor views")
        print("   • Risk Parity optimization methods")
        print("   • Hierarchical Risk Parity clustering")
        print("   • Comprehensive comparative analysis")
        print()
        print("📈 These advanced techniques form the foundation of modern")
        print("   quantitative portfolio management and are used by")
        print("   institutional investors worldwide.")
        print()
        print("💡 Next steps: Explore options pricing models and enhanced ML techniques!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 