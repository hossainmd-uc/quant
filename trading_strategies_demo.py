#!/usr/bin/env python3
"""
Trading Strategies Comprehensive Demo
====================================

This demo showcases the complete trading strategies framework implemented based on
concepts from "Python for Finance: Mastering Data-Driven Finance".

Strategies demonstrated:
1. Momentum Strategies (Basic Momentum, RSI, Bollinger Bands)
2. Mean Reversion Strategies (Mean Reversion, Pairs Trading, Statistical Arbitrage)
3. Trend Following Strategies (Trend Following, MA Crossover, Breakout)
4. Machine Learning Strategies (Linear Regression, Random Forest)
5. Technical Indicator Strategies (MACD, Stochastic)
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import all our trading strategies
try:
    from core.strategies import (
        # Momentum strategies
        MomentumStrategy, RSIMomentumStrategy, BollingerBandsStrategy,
        # Mean reversion strategies  
        MeanReversionStrategy, PairsTrading, StatisticalArbitrage,
        # Trend following strategies
        TrendFollowingStrategy, MovingAverageCrossover, BreakoutStrategy,
        # Technical strategies
        MACDStrategy, StochasticStrategy
    )
    # ML strategies require sklearn
    try:
        from core.strategies import LinearRegressionStrategy, RandomForestStrategy
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
        print("⚠️  Machine Learning strategies unavailable (sklearn not installed)")
        
except ImportError as e:
    print(f"❌ Error importing strategies: {e}")
    print("Please make sure the core.strategies module is properly set up.")
    sys.exit(1)

class TradingStrategiesDemo:
    """Comprehensive demo of all trading strategies."""
    
    def __init__(self):
        self.demo_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
        self.data = {}
        
        # Strategy performance tracking
        self.strategy_results = {}
        
    def get_sample_data(self, days: int = 500) -> bool:
        """Get sample data for all symbols."""
        print("📥 Fetching market data for strategy testing...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        success_count = 0
        for symbol in self.demo_symbols:
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(start=start_date, end=end_date)
                
                if len(data) > 100:  # Minimum data requirement
                    self.data[symbol] = data
                    success_count += 1
                    print(f"✅ {symbol}: {len(data)} days of data")
                else:
                    print(f"⚠️  {symbol}: Insufficient data ({len(data)} days)")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error fetching data - {e}")
        
        if success_count >= 3:
            print(f"\n✅ Successfully loaded data for {success_count} symbols")
            return True
        else:
            print(f"\n❌ Only {success_count} symbols loaded. Need at least 3 for demo.")
            return False
    
    def run_strategy_backtest(self, strategy, strategy_name: str, days_back: int = 100):
        """Run backtest for a strategy."""
        try:
            # Add data to strategy
            for symbol, data in self.data.items():
                strategy.add_data(symbol, data)
            
            # Initialize strategy
            strategy.initialize()
            
            # Run backtest
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            results = strategy.run_backtest(start_date, end_date)
            
            # Store results
            self.strategy_results[strategy_name] = {
                'strategy': strategy,
                'results': results,
                'num_signals': len(results['signals']),
                'performance': results['performance']
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error running {strategy_name}: {e}")
            return None
    
    def demo_momentum_strategies(self):
        """Demonstrate momentum-based strategies."""
        print("\n" + "="*60)
        print("🚀 MOMENTUM STRATEGIES DEMO")
        print("="*60)
        print("Momentum strategies capitalize on the continuation of price trends.")
        print("They buy assets that are rising and sell those that are falling.\n")
        
        # 1. Basic Momentum Strategy
        print("1️⃣  BASIC MOMENTUM STRATEGY")
        print("-" * 30)
        momentum_strategy = MomentumStrategy(
            lookback_period=20,
            min_momentum_threshold=0.02,
            max_momentum_threshold=0.10
        )
        
        results = self.run_strategy_backtest(momentum_strategy, "Basic Momentum")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
        
        # 2. RSI Momentum Strategy
        print(f"\n2️⃣  RSI MOMENTUM STRATEGY")
        print("-" * 30)
        rsi_strategy = RSIMomentumStrategy(
            rsi_period=14,
            oversold_threshold=30,
            overbought_threshold=70
        )
        
        results = self.run_strategy_backtest(rsi_strategy, "RSI Momentum")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
        
        # 3. Bollinger Bands Strategy
        print(f"\n3️⃣  BOLLINGER BANDS STRATEGY")
        print("-" * 30)
        bb_strategy = BollingerBandsStrategy(
            bb_period=20,
            bb_std=2.0,
            breakout_mode=False  # Mean reversion mode
        )
        
        results = self.run_strategy_backtest(bb_strategy, "Bollinger Bands")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
        
        print(f"\n💡 Key Insights:")
        print(f"• Momentum strategies work best in trending markets")
        print(f"• RSI helps identify overbought/oversold conditions")
        print(f"• Bollinger Bands show when prices deviate from normal ranges")
    
    def demo_mean_reversion_strategies(self):
        """Demonstrate mean reversion strategies."""
        print("\n" + "="*60)
        print("📈 MEAN REVERSION STRATEGIES DEMO")
        print("="*60)
        print("Mean reversion strategies assume prices will return to their average.")
        print("They buy when prices are low and sell when prices are high.\n")
        
        # 1. Basic Mean Reversion
        print("1️⃣  BASIC MEAN REVERSION STRATEGY")
        print("-" * 30)
        mean_rev_strategy = MeanReversionStrategy(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        results = self.run_strategy_backtest(mean_rev_strategy, "Mean Reversion")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
        
        # 2. Pairs Trading
        print(f"\n2️⃣  PAIRS TRADING STRATEGY")
        print("-" * 30)
        
        # Only run if we have enough symbols
        available_symbols = list(self.data.keys())
        if len(available_symbols) >= 2:
            pairs = [(available_symbols[0], available_symbols[1])]
            
            pairs_strategy = PairsTrading(
                pairs=pairs,
                lookback_period=60,
                entry_threshold=2.0,
                min_correlation=0.5
            )
            
            results = self.run_strategy_backtest(pairs_strategy, "Pairs Trading")
            if results:
                perf = results['performance']
                print(f"📊 Performance Summary:")
                print(f"   Trading Pair: {pairs[0][0]} / {pairs[0][1]}")
                print(f"   Total Return: {perf.get('total_return', 0):.2%}")
                print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
                print(f"   Number of Signals: {len(results['signals'])}")
        else:
            print("⚠️  Need at least 2 symbols for pairs trading demo")
        
        # 3. Statistical Arbitrage
        print(f"\n3️⃣  STATISTICAL ARBITRAGE STRATEGY")
        print("-" * 30)
        
        if len(available_symbols) >= 3:
            stat_arb_strategy = StatisticalArbitrage(
                basket_symbols=available_symbols[:3],
                lookback_period=60,
                entry_threshold=2.0
            )
            
            results = self.run_strategy_backtest(stat_arb_strategy, "Statistical Arbitrage")
            if results:
                perf = results['performance']
                print(f"📊 Performance Summary:")
                print(f"   Basket: {', '.join(available_symbols[:3])}")
                print(f"   Total Return: {perf.get('total_return', 0):.2%}")
                print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
                print(f"   Number of Signals: {len(results['signals'])}")
        else:
            print("⚠️  Need at least 3 symbols for statistical arbitrage demo")
        
        print(f"\n💡 Key Insights:")
        print(f"• Mean reversion works best in range-bound markets")
        print(f"• Pairs trading exploits temporary price divergences")
        print(f"• Statistical arbitrage trades relative performance vs basket")
    
    def demo_trend_following_strategies(self):
        """Demonstrate trend following strategies."""
        print("\n" + "="*60)
        print("📈 TREND FOLLOWING STRATEGIES DEMO")
        print("="*60)
        print("Trend following strategies identify and trade with market trends.")
        print("They aim to catch sustained price movements in either direction.\n")
        
        # 1. Basic Trend Following
        print("1️⃣  BASIC TREND FOLLOWING STRATEGY")
        print("-" * 30)
        trend_strategy = TrendFollowingStrategy(
            short_window=20,
            long_window=50,
            trend_strength_threshold=0.02
        )
        
        results = self.run_strategy_backtest(trend_strategy, "Trend Following")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
        
        # 2. Moving Average Crossover
        print(f"\n2️⃣  MOVING AVERAGE CROSSOVER STRATEGY")
        print("-" * 30)
        ma_crossover_strategy = MovingAverageCrossover(
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        results = self.run_strategy_backtest(ma_crossover_strategy, "MA Crossover")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
        
        # 3. Breakout Strategy
        print(f"\n3️⃣  BREAKOUT STRATEGY")
        print("-" * 30)
        breakout_strategy = BreakoutStrategy(
            lookback_period=20,
            breakout_threshold=0.02,
            volume_confirmation=True
        )
        
        results = self.run_strategy_backtest(breakout_strategy, "Breakout")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
        
        print(f"\n💡 Key Insights:")
        print(f"• Trend following captures sustained price movements")
        print(f"• MA crossovers provide clear entry/exit signals")
        print(f"• Breakout strategies catch momentum from consolidation")
    
    def demo_technical_strategies(self):
        """Demonstrate technical indicator strategies."""
        print("\n" + "="*60)
        print("📊 TECHNICAL INDICATOR STRATEGIES DEMO")
        print("="*60)
        print("Technical strategies use mathematical indicators derived from price/volume.")
        print("They provide systematic signals based on market behavior patterns.\n")
        
        # 1. MACD Strategy
        print("1️⃣  MACD STRATEGY")
        print("-" * 30)
        macd_strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        results = self.run_strategy_backtest(macd_strategy, "MACD")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
        
        # 2. Stochastic Strategy
        print(f"\n2️⃣  STOCHASTIC STRATEGY")
        print("-" * 30)
        stochastic_strategy = StochasticStrategy(
            k_period=14,
            d_period=3,
            oversold_threshold=20,
            overbought_threshold=80
        )
        
        results = self.run_strategy_backtest(stochastic_strategy, "Stochastic")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
        
        print(f"\n💡 Key Insights:")
        print(f"• MACD shows momentum changes through moving average convergence")
        print(f"• Stochastic identifies overbought/oversold conditions")
        print(f"• Technical indicators provide objective, rule-based signals")
    
    def demo_ml_strategies(self):
        """Demonstrate machine learning strategies."""
        if not ML_AVAILABLE:
            print("\n⚠️  Machine Learning strategies require scikit-learn")
            print("Install with: pip install scikit-learn")
            return
        
        print("\n" + "="*60)
        print("🤖 MACHINE LEARNING STRATEGIES DEMO")
        print("="*60)
        print("ML strategies use statistical models to predict future price movements.")
        print("They learn patterns from historical data and adapt over time.\n")
        
        # 1. Linear Regression Strategy
        print("1️⃣  LINEAR REGRESSION STRATEGY")
        print("-" * 30)
        lr_strategy = LinearRegressionStrategy(
            prediction_horizon=5,
            feature_window=20,
            retrain_frequency=50
        )
        
        results = self.run_strategy_backtest(lr_strategy, "Linear Regression")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
            
            # Show feature importance if available
            if hasattr(lr_strategy, 'get_feature_importance'):
                importance = lr_strategy.get_feature_importance()
                if importance:
                    print(f"   Top Features: {', '.join(list(importance.keys())[:3])}")
        
        # 2. Random Forest Strategy
        print(f"\n2️⃣  RANDOM FOREST STRATEGY")
        print("-" * 30)
        rf_strategy = RandomForestStrategy(
            n_estimators=50,
            max_depth=10,
            prediction_horizon=5,
            feature_window=20,
            retrain_frequency=50
        )
        
        results = self.run_strategy_backtest(rf_strategy, "Random Forest")
        if results:
            perf = results['performance']
            print(f"📊 Performance Summary:")
            print(f"   Total Return: {perf.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"   Win Rate: {perf.get('win_rate', 0):.2%}")
            print(f"   Number of Signals: {len(results['signals'])}")
            
            # Show feature importance if available
            if hasattr(rf_strategy, 'get_feature_importance'):
                importance = rf_strategy.get_feature_importance()
                if importance:
                    print(f"   Top Features: {', '.join(list(importance.keys())[:3])}")
        
        print(f"\n💡 Key Insights:")
        print(f"• ML strategies adapt to changing market conditions")
        print(f"• Linear regression provides interpretable feature weights")
        print(f"• Random forests capture non-linear relationships")
        print(f"• Feature importance helps understand what drives predictions")
    
    def compare_all_strategies(self):
        """Compare performance of all strategies."""
        print("\n" + "="*60)
        print("🏆 STRATEGY PERFORMANCE COMPARISON")
        print("="*60)
        
        if not self.strategy_results:
            print("❌ No strategy results available for comparison")
            return
        
        # Create comparison table
        print(f"{'Strategy':<25} {'Return':<10} {'Sharpe':<8} {'Signals':<8} {'Win Rate':<10}")
        print("-" * 70)
        
        for name, data in self.strategy_results.items():
            perf = data['performance']
            return_val = perf.get('total_return', 0)
            sharpe_val = perf.get('sharpe_ratio', 0)
            signals_val = data['num_signals']
            win_rate_val = perf.get('win_rate', 0)
            
            print(f"{name:<25} {return_val:>8.2%} {sharpe_val:>6.2f} {signals_val:>6} {win_rate_val:>8.2%}")
        
        # Find best performers
        best_return = max(self.strategy_results.items(), 
                         key=lambda x: x[1]['performance'].get('total_return', -999))
        best_sharpe = max(self.strategy_results.items(), 
                         key=lambda x: x[1]['performance'].get('sharpe_ratio', -999))
        
        print(f"\n🏅 Best Performers:")
        print(f"   Highest Return: {best_return[0]} ({best_return[1]['performance'].get('total_return', 0):.2%})")
        print(f"   Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['performance'].get('sharpe_ratio', 0):.2f})")
        
        print(f"\n📈 Strategy Insights:")
        print(f"• Different strategies perform better in different market conditions")
        print(f"• Combining multiple strategies can improve risk-adjusted returns")
        print(f"• Consider transaction costs and slippage in real trading")
        print(f"• Regular rebalancing and parameter optimization is important")
    
    def run_complete_demo(self):
        """Run the complete trading strategies demonstration."""
        print("=" * 60)
        print("🎯 COMPREHENSIVE TRADING STRATEGIES DEMO")
        print("=" * 60)
        print("Based on concepts from 'Python for Finance: Mastering Data-Driven Finance'")
        print("Implementing 177+ trading strategy concepts from the book\n")
        
        # Load data
        if not self.get_sample_data():
            print("❌ Failed to load sufficient data for demo")
            return
        
        # Run all strategy demonstrations
        self.demo_momentum_strategies()
        self.demo_mean_reversion_strategies() 
        self.demo_trend_following_strategies()
        self.demo_technical_strategies()
        self.demo_ml_strategies()
        
        # Compare all strategies
        self.compare_all_strategies()
        
        print("\n" + "="*60)
        print("✅ TRADING STRATEGIES DEMO COMPLETE!")
        print("="*60)
        print("🎓 Key Takeaways:")
        print("• Different strategy types work in different market conditions")
        print("• Risk management is crucial for all strategies")
        print("• Combining strategies can improve performance")
        print("• Regular backtesting and optimization is essential")
        print("• Consider transaction costs in real trading")
        print("\n🚀 Next Steps:")
        print("• Experiment with different parameters")
        print("• Combine strategies into portfolios")
        print("• Add more sophisticated risk management")
        print("• Implement live trading capabilities")

def main():
    """Run the trading strategies demo."""
    try:
        demo = TradingStrategiesDemo()
        demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 