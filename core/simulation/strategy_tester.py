"""
Strategy Testing Framework

Comprehensive backtesting and strategy evaluation system that integrates:
- Portfolio simulation with realistic market conditions
- Position sizing and risk management
- Transaction costs and slippage
- Performance analysis and reporting
- Walk-forward analysis and out-of-sample testing
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    portfolio_values: pd.Series
    daily_returns: pd.Series
    trades: List[Dict]
    positions: Dict[str, Any]
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    transaction_costs: List[Dict]
    drawdown_periods: List[Dict]
    monthly_returns: pd.Series
    regime_analysis: Dict[str, Any]

class StrategyTester:
    """
    Comprehensive strategy testing framework
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 max_position_size: float = 0.2,
                 risk_free_rate: float = 0.02):
        """
        Initialize strategy tester
        
        Args:
            initial_capital: Starting capital
            commission_rate: Commission rate (0.001 = 0.1%)
            slippage_rate: Slippage rate (0.0005 = 0.05%)
            max_position_size: Maximum position size as fraction of portfolio
            risk_free_rate: Risk-free rate for performance metrics
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        
        # Initialize components
        from .portfolio_simulator import PortfolioSimulator
        from .market_simulator import MarketSimulator
        from .position_sizer import PositionSizer
        from .transaction_costs import TransactionCostModel
        from .performance_analyzer import PerformanceAnalyzer
        
        self.portfolio_simulator = PortfolioSimulator(
            initial_cash=initial_capital,
            max_position_size=max_position_size,
            transaction_cost_rate=commission_rate,
            slippage_rate=slippage_rate
        )
        
        self.market_simulator = MarketSimulator()
        self.position_sizer = PositionSizer(max_position_size=max_position_size)
        self.transaction_cost_model = TransactionCostModel()
        self.performance_analyzer = PerformanceAnalyzer(risk_free_rate=risk_free_rate)
    
    def prepare_data(self, 
                    price_data: pd.DataFrame,
                    signal_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare and validate data for backtesting
        
        Args:
            price_data: DataFrame with columns [timestamp, symbol, price, volume]
            signal_data: DataFrame with columns [timestamp, symbol, signal, confidence]
            
        Returns:
            Tuple of (cleaned_price_data, cleaned_signal_data)
        """
        # Validate required columns
        required_price_cols = ['timestamp', 'symbol', 'price', 'volume']
        required_signal_cols = ['timestamp', 'symbol', 'signal', 'confidence']
        
        for col in required_price_cols:
            if col not in price_data.columns:
                raise ValueError(f"Missing required column in price_data: {col}")
        
        for col in required_signal_cols:
            if col not in signal_data.columns:
                raise ValueError(f"Missing required column in signal_data: {col}")
        
        # Convert timestamps to datetime
        price_data = price_data.copy()
        signal_data = signal_data.copy()
        
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        signal_data['timestamp'] = pd.to_datetime(signal_data['timestamp'])
        
        # Remove missing values
        price_data = price_data.dropna()
        signal_data = signal_data.dropna()
        
        # Sort by timestamp
        price_data = price_data.sort_values(['timestamp', 'symbol'])
        signal_data = signal_data.sort_values(['timestamp', 'symbol'])
        
        return price_data, signal_data
    
    def generate_synthetic_data(self, 
                              symbols: List[str],
                              start_date: datetime,
                              end_date: datetime,
                              initial_prices: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic price and signal data for testing
        
        Args:
            symbols: List of asset symbols
            start_date: Start date for data generation
            end_date: End date for data generation
            initial_prices: Initial prices for each symbol
            
        Returns:
            Tuple of (price_data, signal_data)
        """
        if initial_prices is None:
            initial_prices = {symbol: 100.0 for symbol in symbols}
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        price_data = []
        signal_data = []
        
        for symbol in symbols:
            # Generate synthetic price path (geometric Brownian motion)
            n_days = len(date_range)
            dt = 1/252  # Daily time step
            mu = 0.1  # Expected return
            sigma = 0.2  # Volatility
            
            # Generate random returns
            returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_days)
            
            # Generate price path
            prices = [initial_prices[symbol]]
            for i in range(1, n_days):
                price = prices[-1] * (1 + returns[i])
                prices.append(max(price, 0.01))  # Ensure positive prices
            
            # Generate volume data
            base_volume = 1000000
            volumes = np.random.lognormal(np.log(base_volume), 0.5, n_days)
            
            # Generate signals (simple momentum)
            signals = []
            confidences = []
            
            for i in range(n_days):
                if i < 20:  # Need history for momentum
                    signal = 0.0
                    confidence = 0.0
                else:
                    # Simple momentum signal
                    recent_returns = returns[i-20:i]
                    momentum = np.mean(recent_returns)
                    signal = np.tanh(momentum * 10)  # Scale to [-1, 1]
                    confidence = min(abs(signal) * 2, 1.0)  # Higher confidence for stronger signals
                
                signals.append(signal)
                confidences.append(confidence)
            
            # Create DataFrames
            for i, date in enumerate(date_range):
                price_data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'price': prices[i],
                    'volume': volumes[i]
                })
                
                signal_data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'signal': signals[i],
                    'confidence': confidences[i]
                })
        
        price_df = pd.DataFrame(price_data)
        signal_df = pd.DataFrame(signal_data)
        
        return price_df, signal_df
    
    def run_backtest(self,
                    strategy_name: str,
                    price_data: pd.DataFrame,
                    signal_data: pd.DataFrame,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    benchmark_data: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        Run comprehensive backtest
        
        Args:
            strategy_name: Name of the strategy
            price_data: Price data DataFrame
            signal_data: Signal data DataFrame
            start_date: Backtest start date (optional)
            end_date: Backtest end date (optional)
            benchmark_data: Benchmark data for comparison (optional)
            
        Returns:
            BacktestResult with comprehensive results
        """
        # Prepare data
        price_data, signal_data = self.prepare_data(price_data, signal_data)
        
        # Filter by date range if specified
        if start_date:
            price_data = price_data[price_data['timestamp'] >= start_date]
            signal_data = signal_data[signal_data['timestamp'] >= start_date]
        
        if end_date:
            price_data = price_data[price_data['timestamp'] <= end_date]
            signal_data = signal_data[signal_data['timestamp'] <= end_date]
        
        if price_data.empty or signal_data.empty:
            raise ValueError("No data available for the specified date range")
        
        # Get date range
        actual_start_date = min(price_data['timestamp'].min(), signal_data['timestamp'].min())
        actual_end_date = max(price_data['timestamp'].max(), signal_data['timestamp'].max())
        
        # Run portfolio simulation
        simulation_result = self.portfolio_simulator.simulate_strategy(
            signals=signal_data,
            price_data=price_data,
            strategy_name=strategy_name
        )
        
        # Calculate comprehensive performance metrics
        performance_metrics = self.performance_analyzer.calculate_comprehensive_metrics(
            portfolio_values=simulation_result.portfolio_values,
            trades=simulation_result.trades
        )
        
        # Analyze regime performance
        regime_analysis = self.analyze_regime_performance(
            simulation_result.portfolio_values,
            price_data
        )
        
        # Analyze drawdown periods
        drawdown_periods = self.analyze_drawdown_periods(
            simulation_result.portfolio_values
        )
        
        # Calculate monthly returns
        monthly_returns = simulation_result.portfolio_values.resample('M').last().pct_change().dropna()
        
        # Create comprehensive result
        result = BacktestResult(
            strategy_name=strategy_name,
            start_date=actual_start_date,
            end_date=actual_end_date,
            initial_capital=self.initial_capital,
            final_capital=simulation_result.total_value,
            portfolio_values=simulation_result.portfolio_values,
            daily_returns=simulation_result.daily_returns,
            trades=[{
                'symbol': t.symbol,
                'quantity': t.quantity,
                'price': t.price,
                'timestamp': t.timestamp,
                'side': t.side,
                'transaction_cost': t.transaction_cost,
                'slippage': t.slippage
            } for t in simulation_result.trades],
            positions=simulation_result.positions,
            performance_metrics=simulation_result.performance_metrics,
            risk_metrics={
                'volatility': performance_metrics.volatility,
                'max_drawdown': performance_metrics.maximum_drawdown,
                'var_95': performance_metrics.var_95,
                'cvar_95': performance_metrics.cvar_95,
                'ulcer_index': performance_metrics.ulcer_index
            },
            transaction_costs=[],  # Would be populated with detailed cost analysis
            drawdown_periods=drawdown_periods,
            monthly_returns=monthly_returns,
            regime_analysis=regime_analysis
        )
        
        return result
    
    def analyze_regime_performance(self,
                                 portfolio_values: pd.Series,
                                 price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance across different market regimes"""
        # Simple regime detection based on volatility
        returns = portfolio_values.pct_change().dropna()
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Define regimes
        vol_median = rolling_vol.median()
        high_vol_periods = rolling_vol > vol_median * 1.5
        low_vol_periods = rolling_vol < vol_median * 0.5
        normal_vol_periods = ~(high_vol_periods | low_vol_periods)
        
        # Calculate performance by regime
        regime_performance = {}
        
        if high_vol_periods.any():
            high_vol_returns = returns[high_vol_periods]
            regime_performance['high_volatility'] = {
                'periods': high_vol_periods.sum(),
                'avg_return': high_vol_returns.mean() * 252,
                'volatility': high_vol_returns.std() * np.sqrt(252),
                'sharpe': (high_vol_returns.mean() - self.risk_free_rate/252) / high_vol_returns.std() * np.sqrt(252) if high_vol_returns.std() > 0 else 0
            }
        
        if low_vol_periods.any():
            low_vol_returns = returns[low_vol_periods]
            regime_performance['low_volatility'] = {
                'periods': low_vol_periods.sum(),
                'avg_return': low_vol_returns.mean() * 252,
                'volatility': low_vol_returns.std() * np.sqrt(252),
                'sharpe': (low_vol_returns.mean() - self.risk_free_rate/252) / low_vol_returns.std() * np.sqrt(252) if low_vol_returns.std() > 0 else 0
            }
        
        if normal_vol_periods.any():
            normal_vol_returns = returns[normal_vol_periods]
            regime_performance['normal_volatility'] = {
                'periods': normal_vol_periods.sum(),
                'avg_return': normal_vol_returns.mean() * 252,
                'volatility': normal_vol_returns.std() * np.sqrt(252),
                'sharpe': (normal_vol_returns.mean() - self.risk_free_rate/252) / normal_vol_returns.std() * np.sqrt(252) if normal_vol_returns.std() > 0 else 0
            }
        
        return regime_performance
    
    def analyze_drawdown_periods(self, portfolio_values: pd.Series) -> List[Dict]:
        """Analyze individual drawdown periods"""
        # Calculate drawdown
        peak = portfolio_values.cummax()
        drawdown = (portfolio_values - peak) / peak
        
        # Find drawdown periods
        in_drawdown = drawdown < -0.01  # More than 1% drawdown
        
        drawdown_periods = []
        current_period = None
        
        for i, (date, in_dd) in enumerate(in_drawdown.items()):
            if in_dd and current_period is None:
                # Start of drawdown
                current_period = {
                    'start_date': date,
                    'start_value': portfolio_values.iloc[i],
                    'peak_value': peak.iloc[i]
                }
            elif not in_dd and current_period is not None:
                # End of drawdown
                current_period.update({
                    'end_date': date,
                    'end_value': portfolio_values.iloc[i],
                    'trough_value': portfolio_values.iloc[i-1],
                    'max_drawdown': abs(drawdown.iloc[i-1]),
                    'duration_days': (date - current_period['start_date']).days,
                    'recovery_time': 0  # Would need to calculate time to recover
                })
                drawdown_periods.append(current_period)
                current_period = None
        
        # Handle case where backtest ends during drawdown
        if current_period is not None:
            current_period.update({
                'end_date': portfolio_values.index[-1],
                'end_value': portfolio_values.iloc[-1],
                'trough_value': portfolio_values.iloc[-1],
                'max_drawdown': abs(drawdown.iloc[-1]),
                'duration_days': (portfolio_values.index[-1] - current_period['start_date']).days,
                'recovery_time': None  # Still in drawdown
            })
            drawdown_periods.append(current_period)
        
        return drawdown_periods
    
    def run_walk_forward_analysis(self,
                                strategy_name: str,
                                price_data: pd.DataFrame,
                                signal_data: pd.DataFrame,
                                window_size_months: int = 12,
                                step_size_months: int = 1) -> Dict[str, Any]:
        """
        Run walk-forward analysis
        
        Args:
            strategy_name: Name of the strategy
            price_data: Price data DataFrame
            signal_data: Signal data DataFrame
            window_size_months: Size of each analysis window in months
            step_size_months: Step size between windows in months
            
        Returns:
            Dictionary with walk-forward analysis results
        """
        # Prepare data
        price_data, signal_data = self.prepare_data(price_data, signal_data)
        
        # Get date range
        start_date = min(price_data['timestamp'].min(), signal_data['timestamp'].min())
        end_date = max(price_data['timestamp'].max(), signal_data['timestamp'].max())
        
        # Generate analysis windows
        windows = []
        current_start = start_date
        
        while current_start < end_date:
            window_end = current_start + timedelta(days=window_size_months * 30)
            if window_end > end_date:
                window_end = end_date
            
            windows.append((current_start, window_end))
            current_start = current_start + timedelta(days=step_size_months * 30)
        
        # Run backtest for each window
        results = []
        for i, (window_start, window_end) in enumerate(windows):
            try:
                result = self.run_backtest(
                    strategy_name=f"{strategy_name}_Window_{i}",
                    price_data=price_data,
                    signal_data=signal_data,
                    start_date=window_start,
                    end_date=window_end
                )
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to run backtest for window {i}: {e}")
                continue
        
        # Analyze results across windows
        if not results:
            return {'error': 'No successful backtests in walk-forward analysis'}
        
        # Calculate stability metrics
        returns = [r.performance_metrics['annualized_return'] for r in results]
        sharpe_ratios = [r.performance_metrics['sharpe_ratio'] for r in results]
        max_drawdowns = [r.performance_metrics['max_drawdown'] for r in results]
        
        stability_metrics = {
            'num_windows': len(results),
            'return_stability': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'coefficient_of_variation': np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else 0
            },
            'sharpe_stability': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'min': np.min(sharpe_ratios),
                'max': np.max(sharpe_ratios),
                'positive_periods': sum(1 for s in sharpe_ratios if s > 0)
            },
            'drawdown_stability': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'min': np.min(max_drawdowns),
                'max': np.max(max_drawdowns)
            }
        }
        
        return {
            'window_results': results,
            'stability_metrics': stability_metrics,
            'overall_performance': {
                'consistent_profitability': sum(1 for r in returns if r > 0) / len(returns),
                'consistent_positive_sharpe': sum(1 for s in sharpe_ratios if s > 0) / len(sharpe_ratios),
                'max_consecutive_losses': self.calculate_max_consecutive_losses(results),
                'performance_trend': self.calculate_performance_trend(results)
            }
        }
    
    def calculate_max_consecutive_losses(self, results: List[BacktestResult]) -> int:
        """Calculate maximum consecutive losing periods"""
        consecutive_losses = 0
        max_consecutive = 0
        
        for result in results:
            if result.performance_metrics['total_return'] < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive
    
    def calculate_performance_trend(self, results: List[BacktestResult]) -> str:
        """Calculate performance trend across windows"""
        if len(results) < 3:
            return "insufficient_data"
        
        returns = [r.performance_metrics['annualized_return'] for r in results]
        
        # Simple linear regression to detect trend
        x = np.arange(len(returns))
        slope = np.polyfit(x, returns, 1)[0]
        
        if slope > 0.5:
            return "improving"
        elif slope < -0.5:
            return "deteriorating"
        else:
            return "stable"
    
    def compare_strategies(self, results: List[BacktestResult]) -> pd.DataFrame:
        """Compare multiple strategy results"""
        comparison_data = []
        
        for result in results:
            comparison_data.append({
                'Strategy': result.strategy_name,
                'Total Return (%)': result.performance_metrics['total_return'],
                'Annualized Return (%)': result.performance_metrics['annualized_return'],
                'Volatility (%)': result.performance_metrics['volatility'],
                'Sharpe Ratio': result.performance_metrics['sharpe_ratio'],
                'Max Drawdown (%)': result.performance_metrics['max_drawdown'],
                'Calmar Ratio': result.performance_metrics['calmar_ratio'],
                'Win Rate (%)': result.performance_metrics['win_rate'] * 100,
                'Profit Factor': result.performance_metrics['profit_factor'],
                'Total Trades': result.performance_metrics['total_trades'],
                'Final Capital': result.final_capital
            })
        
        return pd.DataFrame(comparison_data)
    
    def generate_backtest_report(self, result: BacktestResult) -> str:
        """Generate comprehensive backtest report"""
        report = f"""
======================================================================
                    {result.strategy_name.upper()} BACKTEST REPORT
======================================================================

BACKTEST SUMMARY:
• Strategy: {result.strategy_name}
• Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}
• Initial Capital: ${result.initial_capital:,.2f}
• Final Capital: ${result.final_capital:,.2f}
• Total Return: {result.performance_metrics['total_return']:.2f}%

PERFORMANCE METRICS:
• Annualized Return: {result.performance_metrics['annualized_return']:.2f}%
• Volatility: {result.performance_metrics['volatility']:.2f}%
• Sharpe Ratio: {result.performance_metrics['sharpe_ratio']:.2f}
• Sortino Ratio: {result.performance_metrics['sortino_ratio']:.2f}
• Calmar Ratio: {result.performance_metrics['calmar_ratio']:.2f}

RISK METRICS:
• Maximum Drawdown: {result.performance_metrics['max_drawdown']:.2f}%
• Value at Risk (95%): {result.risk_metrics['var_95']:.2f}%
• Conditional VaR (95%): {result.risk_metrics['cvar_95']:.2f}%
• Ulcer Index: {result.risk_metrics['ulcer_index']:.2f}

TRADE STATISTICS:
• Total Trades: {result.performance_metrics['total_trades']}
• Win Rate: {result.performance_metrics['win_rate']:.2%}
• Profit Factor: {result.performance_metrics['profit_factor']:.2f}
• Average Trade Return: {result.performance_metrics['avg_trade_return']:.4f}

DRAWDOWN ANALYSIS:
• Number of Drawdown Periods: {len(result.drawdown_periods)}
• Average Drawdown Duration: {np.mean([dd['duration_days'] for dd in result.drawdown_periods]):.0f} days
• Longest Drawdown: {max([dd['duration_days'] for dd in result.drawdown_periods]) if result.drawdown_periods else 0} days

REGIME ANALYSIS:
"""
        
        for regime, stats in result.regime_analysis.items():
            report += f"• {regime.replace('_', ' ').title()}: {stats['periods']} periods, Return: {stats['avg_return']:.2f}%, Sharpe: {stats['sharpe']:.2f}\n"
        
        report += "\n======================================================================\n"
        
        return report 