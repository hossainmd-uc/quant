#!/usr/bin/env python3
"""
Educational Demo: Understanding Financial Risk Metrics
=====================================================

This demo helps you understand financial concepts by running them on real data
and explaining what each metric means in practical terms.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our risk management modules
from core.risk.risk_metrics import RiskMetrics
from core.risk.var_calculator import VaRCalculator
from core.risk.monte_carlo import MonteCarloSimulator
from core.risk.portfolio_optimizer import PortfolioOptimizer
from core.risk.stress_testing import StressTesting

class FinancialEducationDemo:
    """Interactive demo to understand financial risk concepts."""
    
    def __init__(self):
        self.risk_metrics = RiskMetrics()
        self.var_calculator = VaRCalculator()
        self.monte_carlo = MonteCarloSimulator()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.stress_tester = StressTesting()
        
        # Sample tickers for demonstration
        self.demo_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
        
    def explain_concept(self, concept_name: str, explanation: str):
        """Pretty print concept explanations."""
        print(f"\n{'='*60}")
        print(f"📚 CONCEPT: {concept_name}")
        print(f"{'='*60}")
        print(explanation)
        print(f"{'='*60}")
        
    def get_sample_data(self, ticker: str = 'AAPL', days: int = 252):
        """Get sample data for demonstration."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days*2)  # Get more data to ensure we have enough
            
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if len(data) < days:
                print(f"⚠️  Warning: Only got {len(data)} days of data for {ticker}")
                
            return data['Close'].dropna()
        except Exception as e:
            print(f"❌ Error getting data for {ticker}: {e}")
            return None
    
    def demo_sharpe_ratio(self):
        """Demonstrate Sharpe ratio calculation and interpretation."""
        self.explain_concept(
            "SHARPE RATIO",
            """
The Sharpe ratio measures risk-adjusted return. It answers the question:
"How much extra return do I get for the extra risk I take?"

Formula: (Return - Risk-Free Rate) / Volatility

🎯 INTERPRETATION:
- Above 1.0: Excellent risk-adjusted performance
- 0.5-1.0: Good performance
- Below 0.5: Poor risk-adjusted performance
- Negative: You're losing money relative to safe investments

Let's compare different assets...
            """
        )
        
        print("\n🔍 SHARPE RATIO COMPARISON:")
        print("-" * 50)
        
        for ticker in self.demo_tickers:
            data = self.get_sample_data(ticker)
            if data is not None:
                returns = data.pct_change().dropna()
                
                # Calculate Sharpe ratio
                risk_free_rate = 0.02  # 2% annual risk-free rate
                excess_return = returns.mean() * 252 - risk_free_rate
                volatility = returns.std() * np.sqrt(252)
                sharpe = excess_return / volatility if volatility > 0 else 0
                
                # Interpret the result
                if sharpe > 1.0:
                    interpretation = "🔥 EXCELLENT"
                elif sharpe > 0.5:
                    interpretation = "✅ GOOD"
                elif sharpe > 0:
                    interpretation = "⚠️  POOR"
                else:
                    interpretation = "❌ LOSING MONEY"
                
                print(f"{ticker:>6}: Sharpe = {sharpe:>6.2f} | {interpretation}")
        
        input("\n📖 Press Enter to continue to the next concept...")
    
    def demo_max_drawdown(self):
        """Demonstrate maximum drawdown calculation."""
        self.explain_concept(
            "MAXIMUM DRAWDOWN",
            """
Maximum Drawdown shows the biggest peak-to-trough loss in your portfolio.
It answers: "What's the worst loss I experienced?"

This is CRUCIAL for understanding:
- Psychological stress levels
- Position sizing decisions
- Risk tolerance assessment

🎯 PRACTICAL IMPACT:
If you can't stomach a 40% loss, avoid strategies with 40%+ max drawdowns!
            """
        )
        
        print("\n🔍 MAXIMUM DRAWDOWN ANALYSIS:")
        print("-" * 50)
        
        # Let's analyze TSLA in detail (known for high volatility)
        ticker = 'TSLA'
        data = self.get_sample_data(ticker, days=500)
        
        if data is not None:
            # Calculate cumulative returns
            returns = data.pct_change().dropna()
            cumulative_returns = (1 + returns).cumprod()
            
            # Calculate rolling maximum (peak)
            rolling_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            print(f"\n📊 {ticker} Analysis:")
            print(f"Maximum Drawdown: {max_drawdown:.2%}")
            print(f"Current Drawdown: {drawdown.iloc[-1]:.2%}")
            
            # Find the worst period
            worst_day = drawdown.idxmin()
            print(f"Worst Drawdown Date: {worst_day.strftime('%Y-%m-%d')}")
            
            # Practical interpretation
            if abs(max_drawdown) > 0.5:
                print("⚠️  WARNING: This asset had extreme losses! High psychological stress.")
            elif abs(max_drawdown) > 0.3:
                print("⚠️  CAUTION: Significant losses possible. Consider position sizing.")
            else:
                print("✅ MODERATE: Relatively stable asset for long-term holding.")
        
        input("\n📖 Press Enter to continue to VaR concepts...")
    
    def demo_var_concepts(self):
        """Demonstrate different VaR calculation methods."""
        self.explain_concept(
            "VALUE-AT-RISK (VaR)",
            """
VaR answers: "There's a 95% chance I won't lose more than $X tomorrow."

We'll compare 4 different methods:
1. HISTORICAL: Look at past data, find 5th percentile
2. PARAMETRIC: Assume normal distribution
3. MONTE CARLO: Simulate thousands of scenarios
4. CORNISH-FISHER: Adjust for skewness and fat tails

Each method has trade-offs between accuracy and speed.
            """
        )
        
        print("\n🔍 VaR METHODS COMPARISON:")
        print("-" * 50)
        
        # Use SPY for demonstration (broad market)
        ticker = 'SPY'
        data = self.get_sample_data(ticker)
        
        if data is not None:
            returns = data.pct_change().dropna()
            
            # Calculate VaR using different methods
            var_results = self.var_calculator.calculate_comprehensive_var(
                returns, confidence_level=0.95, portfolio_value=100000
            )
            
            print(f"\n📊 VaR Analysis for {ticker} (95% confidence, $100,000 portfolio):")
            print(f"Historical VaR:      ${var_results['historical_var']:,.0f}")
            print(f"Parametric VaR:      ${var_results['parametric_var']:,.0f}")
            print(f"Monte Carlo VaR:     ${var_results['monte_carlo_var']:,.0f}")
            print(f"Cornish-Fisher VaR:  ${var_results['cornish_fisher_var']:,.0f}")
            print(f"Expected Shortfall:  ${var_results['expected_shortfall']:,.0f}")
            
            # Explain the differences
            print(f"\n🔍 INTERPRETATION:")
            print(f"• Historical: Based on actual past market behavior")
            print(f"• Parametric: Assumes normal distribution (often underestimates risk)")
            print(f"• Monte Carlo: Uses simulation (most flexible)")
            print(f"• Expected Shortfall: Average loss when VaR is exceeded")
            
            # Show practical application
            daily_var = var_results['historical_var']
            print(f"\n💡 PRACTICAL USE:")
            print(f"• Daily risk budget: ${daily_var:,.0f}")
            print(f"• Position sizing: If this is 2% of portfolio, you can risk ${daily_var*50:,.0f} total")
            print(f"• Stop-loss level: Consider stops at ${daily_var*1.5:,.0f} loss")
        
        input("\n📖 Press Enter to continue to portfolio optimization...")
    
    def demo_portfolio_optimization(self):
        """Demonstrate Modern Portfolio Theory optimization."""
        self.explain_concept(
            "PORTFOLIO OPTIMIZATION",
            """
Modern Portfolio Theory (MPT) shows how to build optimal portfolios.
Key insight: You can reduce risk through diversification without reducing returns!

We'll optimize a portfolio using different objectives:
1. MAXIMUM SHARPE: Best risk-adjusted returns
2. MINIMUM VOLATILITY: Lowest risk
3. MAXIMUM RETURN: Highest returns (usually not recommended alone)
            """
        )
        
        print("\n🔍 PORTFOLIO OPTIMIZATION DEMO:")
        print("-" * 50)
        
        # Get data for multiple assets
        portfolio_data = {}
        for ticker in self.demo_tickers:
            data = self.get_sample_data(ticker)
            if data is not None:
                portfolio_data[ticker] = data.pct_change().dropna()
        
        if len(portfolio_data) >= 3:
            # Create returns matrix
            returns_df = pd.DataFrame(portfolio_data)
            returns_df = returns_df.dropna()
            
            # Calculate different optimal portfolios
            try:
                # Maximum Sharpe ratio portfolio
                max_sharpe_weights = self.portfolio_optimizer.optimize_portfolio(
                    returns_df, objective='max_sharpe'
                )
                
                # Minimum volatility portfolio
                min_vol_weights = self.portfolio_optimizer.optimize_portfolio(
                    returns_df, objective='min_volatility'
                )
                
                print(f"\n📊 OPTIMAL PORTFOLIO ALLOCATIONS:")
                print(f"\n🏆 MAXIMUM SHARPE RATIO (Best Risk-Adjusted):")
                for ticker, weight in max_sharpe_weights.items():
                    print(f"  {ticker}: {weight:.1%}")
                
                print(f"\n🛡️  MINIMUM VOLATILITY (Lowest Risk):")
                for ticker, weight in min_vol_weights.items():
                    print(f"  {ticker}: {weight:.1%}")
                
                # Calculate portfolio statistics
                max_sharpe_return = np.sum(max_sharpe_weights * returns_df.mean() * 252)
                max_sharpe_vol = np.sqrt(np.dot(max_sharpe_weights, np.dot(returns_df.cov() * 252, max_sharpe_weights)))
                max_sharpe_sharpe = max_sharpe_return / max_sharpe_vol
                
                min_vol_return = np.sum(min_vol_weights * returns_df.mean() * 252)
                min_vol_vol = np.sqrt(np.dot(min_vol_weights, np.dot(returns_df.cov() * 252, min_vol_weights)))
                min_vol_sharpe = min_vol_return / min_vol_vol
                
                print(f"\n📈 PORTFOLIO STATISTICS:")
                print(f"Max Sharpe Portfolio: Return={max_sharpe_return:.1%}, Vol={max_sharpe_vol:.1%}, Sharpe={max_sharpe_sharpe:.2f}")
                print(f"Min Vol Portfolio:    Return={min_vol_return:.1%}, Vol={min_vol_vol:.1%}, Sharpe={min_vol_sharpe:.2f}")
                
                # Equal weight comparison
                equal_weights = pd.Series(1/len(returns_df.columns), index=returns_df.columns)
                equal_return = np.sum(equal_weights * returns_df.mean() * 252)
                equal_vol = np.sqrt(np.dot(equal_weights, np.dot(returns_df.cov() * 252, equal_weights)))
                equal_sharpe = equal_return / equal_vol
                
                print(f"Equal Weight:         Return={equal_return:.1%}, Vol={equal_vol:.1%}, Sharpe={equal_sharpe:.2f}")
                
                print(f"\n💡 INSIGHTS:")
                print(f"• Max Sharpe portfolio optimizes risk-adjusted returns")
                print(f"• Min Vol portfolio reduces risk but may sacrifice returns")
                print(f"• Both likely outperform equal weighting")
                
            except Exception as e:
                print(f"❌ Error in optimization: {e}")
        
        input("\n📖 Press Enter to continue to Monte Carlo simulation...")
    
    def demo_monte_carlo(self):
        """Demonstrate Monte Carlo simulation concepts."""
        self.explain_concept(
            "MONTE CARLO SIMULATION",
            """
Monte Carlo runs thousands of "what-if" scenarios to understand possible outcomes.

Process:
1. Model how prices move (random walks with drift)
2. Simulate thousands of possible futures
3. Analyze the distribution of outcomes

This gives you:
- Probability of different outcomes
- Risk measures (VaR, CVaR)
- Confidence intervals
            """
        )
        
        print("\n🔍 MONTE CARLO SIMULATION DEMO:")
        print("-" * 50)
        
        # Get data for simulation
        ticker = 'AAPL'
        data = self.get_sample_data(ticker)
        
        if data is not None:
            returns = data.pct_change().dropna()
            
            # Run Monte Carlo simulation
            num_simulations = 5000
            time_horizon = 252  # 1 year
            initial_value = 100000  # $100,000
            
            print(f"\n🎲 Running {num_simulations:,} simulations for {ticker}...")
            print(f"Initial Portfolio Value: ${initial_value:,}")
            print(f"Time Horizon: {time_horizon} days (1 year)")
            
            # Calculate statistics from returns
            mu = returns.mean()
            sigma = returns.std()
            
            print(f"\n📊 HISTORICAL STATISTICS:")
            print(f"Daily Return (μ): {mu:.4f} ({mu*252:.2%} annualized)")
            print(f"Daily Volatility (σ): {sigma:.4f} ({sigma*np.sqrt(252):.2%} annualized)")
            
            # Run simulation
            results = self.monte_carlo.simulate_portfolio_paths(
                initial_value=initial_value,
                returns=returns,
                time_horizon=time_horizon,
                num_simulations=num_simulations
            )
            
            final_values = results['final_values']
            
            print(f"\n🎯 SIMULATION RESULTS:")
            print(f"Mean Final Value: ${np.mean(final_values):,.0f}")
            print(f"Median Final Value: ${np.median(final_values):,.0f}")
            print(f"Standard Deviation: ${np.std(final_values):,.0f}")
            
            # Risk metrics
            var_95 = np.percentile(final_values, 5)
            var_99 = np.percentile(final_values, 1)
            cvar_95 = np.mean(final_values[final_values <= var_95])
            
            print(f"\n📉 RISK METRICS:")
            print(f"VaR 95%: ${var_95:,.0f} (5% chance of losing more than ${initial_value - var_95:,.0f})")
            print(f"VaR 99%: ${var_99:,.0f} (1% chance of losing more than ${initial_value - var_99:,.0f})")
            print(f"CVaR 95%: ${cvar_95:,.0f} (average loss in worst 5% of cases)")
            
            # Probability analysis
            prob_profit = np.mean(final_values > initial_value) * 100
            prob_double = np.mean(final_values > initial_value * 2) * 100
            prob_loss_50 = np.mean(final_values < initial_value * 0.5) * 100
            
            print(f"\n🎯 PROBABILITY ANALYSIS:")
            print(f"Probability of Profit: {prob_profit:.1f}%")
            print(f"Probability of Doubling: {prob_double:.1f}%")
            print(f"Probability of 50%+ Loss: {prob_loss_50:.1f}%")
            
            print(f"\n💡 PRACTICAL INSIGHTS:")
            print(f"• Monte Carlo shows the range of possible outcomes")
            print(f"• Use VaR for position sizing and risk budgeting")
            print(f"• Consider worst-case scenarios for stress testing")
        
        input("\n📖 Press Enter to continue to stress testing...")
    
    def demo_stress_testing(self):
        """Demonstrate stress testing concepts."""
        self.explain_concept(
            "STRESS TESTING",
            """
Stress testing asks: "What happens in extreme market conditions?"

Normal models assume "normal" markets, but markets can be abnormal!
We test scenarios like:
- Market crashes (like 2008, 2020)
- Volatility spikes
- Correlation breakdowns

This helps you understand portfolio behavior in crisis situations.
            """
        )
        
        print("\n🔍 STRESS TESTING DEMO:")
        print("-" * 50)
        
        # Get data for stress testing
        ticker = 'SPY'
        data = self.get_sample_data(ticker, days=500)
        
        if data is not None:
            returns = data.pct_change().dropna()
            initial_value = 100000
            
            print(f"\n🧪 STRESS TESTING {ticker} PORTFOLIO:")
            print(f"Initial Value: ${initial_value:,}")
            
            # Normal scenario (baseline)
            normal_result = self.monte_carlo.simulate_portfolio_paths(
                initial_value=initial_value,
                returns=returns,
                time_horizon=252,
                num_simulations=1000
            )
            
            # Market crash scenario
            crash_result = self.stress_tester.simulate_market_crash_scenario(
                initial_value=initial_value,
                returns=returns,
                crash_magnitude=0.3,  # 30% crash
                time_horizon=252,
                num_simulations=1000
            )
            
            # Volatility spike scenario
            vol_spike_result = self.stress_tester.simulate_volatility_spike(
                initial_value=initial_value,
                returns=returns,
                vol_multiplier=2.0,  # Double volatility
                time_horizon=252,
                num_simulations=1000
            )
            
            print(f"\n📊 SCENARIO COMPARISON:")
            print(f"{'Scenario':<20} {'Mean Final':<12} {'VaR 95%':<10} {'CVaR 95%':<10}")
            print("-" * 55)
            
            scenarios = [
                ("Normal", normal_result['final_values']),
                ("Market Crash", crash_result['final_values']),
                ("High Volatility", vol_spike_result['final_values'])
            ]
            
            for name, values in scenarios:
                mean_val = np.mean(values)
                var_95 = np.percentile(values, 5)
                cvar_95 = np.mean(values[values <= var_95])
                
                print(f"{name:<20} ${mean_val:>8,.0f}   ${var_95:>8,.0f}  ${cvar_95:>8,.0f}")
            
            # Calculate stress impact
            normal_var = np.percentile(normal_result['final_values'], 5)
            crash_var = np.percentile(crash_result['final_values'], 5)
            stress_impact = (normal_var - crash_var) / normal_var * 100
            
            print(f"\n⚠️  STRESS IMPACT ANALYSIS:")
            print(f"Market Crash increases VaR by {stress_impact:.1f}%")
            print(f"Normal VaR 95%: ${normal_var:,.0f}")
            print(f"Crash VaR 95%: ${crash_var:,.0f}")
            print(f"Additional Risk: ${normal_var - crash_var:,.0f}")
            
            print(f"\n💡 STRESS TESTING INSIGHTS:")
            print(f"• Stress tests reveal hidden risks in extreme conditions")
            print(f"• Use results for position sizing and risk budgeting")
            print(f"• Consider correlation breakdowns in portfolio construction")
            print(f"• Plan for scenarios beyond normal market conditions")
        
        input("\n📖 Press Enter to see the summary...")
    
    def demo_summary(self):
        """Provide a comprehensive summary of all concepts."""
        self.explain_concept(
            "SUMMARY: PUTTING IT ALL TOGETHER",
            """
🎯 RISK METRICS DECISION TREE:

1. PERFORMANCE EVALUATION:
   → Use Sharpe/Sortino ratio to compare strategies
   → Higher = better risk-adjusted returns

2. RISK ASSESSMENT:
   → Check Maximum Drawdown for worst-case losses
   → Use VaR for daily/weekly risk budgeting

3. PORTFOLIO CONSTRUCTION:
   → Apply Modern Portfolio Theory for optimal allocation
   → Consider correlation benefits of diversification

4. SCENARIO PLANNING:
   → Use Monte Carlo for understanding outcome distributions
   → Apply stress testing for extreme scenarios

5. PRACTICAL IMPLEMENTATION:
   → Set position sizes based on VaR
   → Use drawdown limits for stop-losses
   → Monitor correlations for diversification benefits
   → Regular rebalancing based on optimization

🔧 WORKFLOW EXAMPLE:
1. Calculate Sharpe ratios → Select best strategies
2. Optimize portfolio weights → Allocate capital
3. Calculate VaR → Size positions
4. Run stress tests → Set risk limits
5. Monitor and rebalance → Maintain optimal allocation
            """
        )
        
        print("\n🎓 CONGRATULATIONS!")
        print("You now understand the core concepts of quantitative risk management!")
        print("\n📚 KEY TAKEAWAYS:")
        print("• Risk metrics help you make informed decisions")
        print("• Different VaR methods have different use cases")
        print("• Portfolio optimization can improve risk-adjusted returns")
        print("• Monte Carlo simulation reveals possible outcomes")
        print("• Stress testing prepares you for extreme events")
        print("\n🚀 NEXT STEPS:")
        print("• Apply these concepts to your own trading strategies")
        print("• Experiment with different parameters and timeframes")
        print("• Build intuition by running scenarios regularly")
        print("• Remember: These are tools to inform decisions, not replace judgment!")

def main():
    """Run the educational demo."""
    print("=" * 60)
    print("🎓 FINANCIAL RISK MANAGEMENT EDUCATION DEMO")
    print("=" * 60)
    print("This demo will help you understand financial risk concepts")
    print("by running them on real market data with explanations.")
    print("\n📋 We'll cover:")
    print("1. Risk Metrics (Sharpe, Drawdown, etc.)")
    print("2. Value-at-Risk (VaR) Methods")
    print("3. Portfolio Optimization")
    print("4. Monte Carlo Simulation")
    print("5. Stress Testing")
    print("\n🔍 Each section includes:")
    print("• Concept explanation")
    print("• Real data examples")
    print("• Practical interpretation")
    print("• When to use each method")
    
    input("\n📖 Press Enter to start the educational journey...")
    
    # Initialize demo
    demo = FinancialEducationDemo()
    
    # Run through all concepts
    demo.demo_sharpe_ratio()
    demo.demo_max_drawdown()
    demo.demo_var_concepts()
    demo.demo_portfolio_optimization()
    demo.demo_monte_carlo()
    demo.demo_stress_testing()
    demo.demo_summary()
    
    print("\n" + "=" * 60)
    print("🎉 Educational Demo Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main() 