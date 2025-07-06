"""
Performance Analyzer - Comprehensive Trading Metrics

Calculates comprehensive performance metrics including:
- Return metrics (total, annualized, risk-adjusted)
- Risk metrics (volatility, drawdown, VaR)
- Efficiency metrics (Sharpe, Sortino, Calmar)
- Trade statistics and attribution
- Benchmark comparisons
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    compound_annual_growth_rate: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    maximum_drawdown: float
    average_drawdown: float
    drawdown_duration: int
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    payoff_ratio: float
    average_trade_return: float
    
    # Time-based metrics
    best_month: float
    worst_month: float
    positive_months: int
    negative_months: int
    
    # Additional metrics
    recovery_factor: float
    ulcer_index: float
    gain_to_pain_ratio: float
    sterling_ratio: float

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis system
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
    
    def calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate returns from portfolio values"""
        return portfolio_values.pct_change().dropna()
    
    def calculate_total_return(self, portfolio_values: pd.Series) -> float:
        """Calculate total return"""
        if len(portfolio_values) < 2:
            return 0.0
        
        return (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
    
    def calculate_annualized_return(self, portfolio_values: pd.Series) -> float:
        """Calculate annualized return"""
        if len(portfolio_values) < 2:
            return 0.0
        
        total_return = self.calculate_total_return(portfolio_values) / 100
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        
        if days <= 0:
            return 0.0
        
        annualized_return = (1 + total_return) ** (365 / days) - 1
        return annualized_return * 100
    
    def calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return 0.0
        
        return returns.std() * np.sqrt(self.trading_days_per_year) * 100
    
    def calculate_downside_volatility(self, returns: pd.Series) -> float:
        """Calculate downside volatility (volatility of negative returns)"""
        negative_returns = returns[returns < 0]
        if len(negative_returns) < 2:
            return 0.0
        
        return negative_returns.std() * np.sqrt(self.trading_days_per_year) * 100
    
    def calculate_drawdown(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        peak = portfolio_values.cummax()
        drawdown = (portfolio_values - peak) / peak * 100
        return drawdown
    
    def calculate_maximum_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        drawdown = self.calculate_drawdown(portfolio_values)
        return abs(drawdown.min())
    
    def calculate_average_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate average drawdown"""
        drawdown = self.calculate_drawdown(portfolio_values)
        negative_drawdown = drawdown[drawdown < 0]
        if len(negative_drawdown) == 0:
            return 0.0
        return abs(negative_drawdown.mean())
    
    def calculate_drawdown_duration(self, portfolio_values: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        drawdown = self.calculate_drawdown(portfolio_values)
        
        # Find periods of drawdown
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            return 0
        
        # Calculate duration of each drawdown period
        drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)
        
        max_duration = 0
        current_start = None
        
        for i, (start, end) in enumerate(zip(drawdown_starts, drawdown_ends)):
            if start:
                current_start = drawdown_starts.index[i]
            if end and current_start is not None:
                duration = (drawdown_ends.index[i] - current_start).days
                max_duration = max(max_duration, duration)
                current_start = None
        
        return max_duration
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        
        excess_return = returns.mean() - self.risk_free_rate / self.trading_days_per_year
        return (excess_return / returns.std()) * np.sqrt(self.trading_days_per_year)
    
    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_return = returns.mean() - self.risk_free_rate / self.trading_days_per_year
        downside_vol = self.calculate_downside_volatility(returns) / 100
        
        if downside_vol == 0:
            return 0.0
        
        return (excess_return * np.sqrt(self.trading_days_per_year)) / downside_vol
    
    def calculate_calmar_ratio(self, portfolio_values: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annualized_return = self.calculate_annualized_return(portfolio_values)
        max_drawdown = self.calculate_maximum_drawdown(portfolio_values)
        
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / max_drawdown
    
    def calculate_var_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        if len(returns) < 10:
            return 0.0, 0.0
        
        var = np.percentile(returns, (1 - confidence_level) * 100) * 100
        cvar = returns[returns <= var/100].mean() * 100
        
        return abs(var), abs(cvar)
    
    def calculate_ulcer_index(self, portfolio_values: pd.Series) -> float:
        """Calculate Ulcer Index"""
        drawdown = self.calculate_drawdown(portfolio_values)
        squared_drawdown = drawdown ** 2
        ulcer_index = np.sqrt(squared_drawdown.mean())
        return ulcer_index
    
    def calculate_gain_to_pain_ratio(self, returns: pd.Series) -> float:
        """Calculate Gain-to-Pain ratio"""
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        
        if negative_returns == 0:
            return float('inf') if positive_returns > 0 else 0.0
        
        return positive_returns / negative_returns
    
    def calculate_sterling_ratio(self, portfolio_values: pd.Series) -> float:
        """Calculate Sterling ratio"""
        annualized_return = self.calculate_annualized_return(portfolio_values)
        average_drawdown = self.calculate_average_drawdown(portfolio_values)
        
        if average_drawdown == 0:
            return 0.0
        
        return annualized_return / average_drawdown
    
    def analyze_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze trade statistics"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'payoff_ratio': 0.0,
                'average_trade_return': 0.0
            }
        
        # Extract trade results
        trade_results = []
        for trade in trades:
            if 'pnl' in trade:
                trade_results.append(trade['pnl'])
            elif 'profit' in trade:
                trade_results.append(trade['profit'])
        
        if not trade_results:
            return {
                'total_trades': len(trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'payoff_ratio': 0.0,
                'average_trade_return': 0.0
            }
        
        # Calculate statistics
        winning_trades = [t for t in trade_results if t > 0]
        losing_trades = [t for t in trade_results if t < 0]
        
        total_trades = len(trade_results)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = num_winning / total_trades if total_trades > 0 else 0.0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Payoff ratio
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Average trade return
        avg_trade_return = np.mean(trade_results)
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'payoff_ratio': payoff_ratio,
            'average_trade_return': avg_trade_return
        }
    
    def analyze_monthly_performance(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """Analyze monthly performance"""
        if len(portfolio_values) < 2:
            return {
                'best_month': 0.0,
                'worst_month': 0.0,
                'positive_months': 0,
                'negative_months': 0
            }
        
        # Resample to monthly
        monthly_values = portfolio_values.resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna() * 100
        
        if len(monthly_returns) == 0:
            return {
                'best_month': 0.0,
                'worst_month': 0.0,
                'positive_months': 0,
                'negative_months': 0
            }
        
        best_month = monthly_returns.max()
        worst_month = monthly_returns.min()
        positive_months = (monthly_returns > 0).sum()
        negative_months = (monthly_returns < 0).sum()
        
        return {
            'best_month': best_month,
            'worst_month': worst_month,
            'positive_months': positive_months,
            'negative_months': negative_months
        }
    
    def calculate_comprehensive_metrics(self,
                                      portfolio_values: pd.Series,
                                      trades: Optional[List[Dict]] = None,
                                      benchmark_returns: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            portfolio_values: Portfolio value time series
            trades: List of trade dictionaries
            benchmark_returns: Benchmark return series for comparison
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        returns = self.calculate_returns(portfolio_values)
        
        # Return metrics
        total_return = self.calculate_total_return(portfolio_values)
        annualized_return = self.calculate_annualized_return(portfolio_values)
        
        # Risk metrics
        volatility = self.calculate_volatility(returns)
        downside_volatility = self.calculate_downside_volatility(returns)
        maximum_drawdown = self.calculate_maximum_drawdown(portfolio_values)
        average_drawdown = self.calculate_average_drawdown(portfolio_values)
        drawdown_duration = self.calculate_drawdown_duration(portfolio_values)
        
        # Risk-adjusted metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        calmar_ratio = self.calculate_calmar_ratio(portfolio_values)
        
        # Information ratio (vs benchmark)
        information_ratio = 0.0
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            excess_returns = returns - benchmark_returns
            if len(excess_returns) > 1 and excess_returns.std() > 0:
                information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days_per_year)
        
        # Treynor ratio (simplified without beta)
        treynor_ratio = 0.0
        if len(returns) > 1:
            excess_return = returns.mean() - self.risk_free_rate / self.trading_days_per_year
            # Assuming beta = 1 for simplicity
            treynor_ratio = excess_return * self.trading_days_per_year
        
        # Distribution metrics
        skewness = returns.skew() if len(returns) > 2 else 0.0
        kurtosis = returns.kurtosis() if len(returns) > 2 else 0.0
        var_95, cvar_95 = self.calculate_var_cvar(returns)
        
        # Trade statistics
        trade_stats = self.analyze_trades(trades or [])
        
        # Monthly performance
        monthly_stats = self.analyze_monthly_performance(portfolio_values)
        
        # Additional metrics
        recovery_factor = annualized_return / maximum_drawdown if maximum_drawdown > 0 else 0.0
        ulcer_index = self.calculate_ulcer_index(portfolio_values)
        gain_to_pain_ratio = self.calculate_gain_to_pain_ratio(returns)
        sterling_ratio = self.calculate_sterling_ratio(portfolio_values)
        
        return PerformanceMetrics(
            # Return metrics
            total_return=total_return,
            annualized_return=annualized_return,
            compound_annual_growth_rate=annualized_return,
            
            # Risk metrics
            volatility=volatility,
            downside_volatility=downside_volatility,
            maximum_drawdown=maximum_drawdown,
            average_drawdown=average_drawdown,
            drawdown_duration=drawdown_duration,
            
            # Risk-adjusted metrics
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            
            # Distribution metrics
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95,
            
            # Trade statistics
            total_trades=trade_stats['total_trades'],
            winning_trades=trade_stats['winning_trades'],
            losing_trades=trade_stats['losing_trades'],
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            payoff_ratio=trade_stats['payoff_ratio'],
            average_trade_return=trade_stats['average_trade_return'],
            
            # Monthly metrics
            best_month=monthly_stats['best_month'],
            worst_month=monthly_stats['worst_month'],
            positive_months=monthly_stats['positive_months'],
            negative_months=monthly_stats['negative_months'],
            
            # Additional metrics
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            gain_to_pain_ratio=gain_to_pain_ratio,
            sterling_ratio=sterling_ratio
        )
    
    def compare_strategies(self, 
                          strategy_results: Dict[str, pd.Series]) -> pd.DataFrame:
        """Compare multiple strategies"""
        comparison_data = []
        
        for strategy_name, portfolio_values in strategy_results.items():
            metrics = self.calculate_comprehensive_metrics(portfolio_values)
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': metrics.total_return,
                'Annualized Return (%)': metrics.annualized_return,
                'Volatility (%)': metrics.volatility,
                'Sharpe Ratio': metrics.sharpe_ratio,
                'Sortino Ratio': metrics.sortino_ratio,
                'Calmar Ratio': metrics.calmar_ratio,
                'Max Drawdown (%)': metrics.maximum_drawdown,
                'Win Rate (%)': metrics.win_rate * 100,
                'Profit Factor': metrics.profit_factor
            })
        
        return pd.DataFrame(comparison_data)
    
    def generate_performance_report(self, 
                                  portfolio_values: pd.Series,
                                  trades: Optional[List[Dict]] = None,
                                  strategy_name: str = "Strategy") -> str:
        """Generate comprehensive performance report"""
        metrics = self.calculate_comprehensive_metrics(portfolio_values, trades)
        
        report = f"""
======================================================================
                    {strategy_name.upper()} PERFORMANCE REPORT
======================================================================

RETURN METRICS:
• Total Return: {metrics.total_return:.2f}%
• Annualized Return: {metrics.annualized_return:.2f}%
• Compound Annual Growth Rate: {metrics.compound_annual_growth_rate:.2f}%

RISK METRICS:
• Volatility: {metrics.volatility:.2f}%
• Downside Volatility: {metrics.downside_volatility:.2f}%
• Maximum Drawdown: {metrics.maximum_drawdown:.2f}%
• Average Drawdown: {metrics.average_drawdown:.2f}%
• Drawdown Duration: {metrics.drawdown_duration} days

RISK-ADJUSTED METRICS:
• Sharpe Ratio: {metrics.sharpe_ratio:.2f}
• Sortino Ratio: {metrics.sortino_ratio:.2f}
• Calmar Ratio: {metrics.calmar_ratio:.2f}
• Information Ratio: {metrics.information_ratio:.2f}
• Recovery Factor: {metrics.recovery_factor:.2f}

DISTRIBUTION METRICS:
• Skewness: {metrics.skewness:.2f}
• Kurtosis: {metrics.kurtosis:.2f}
• Value at Risk (95%): {metrics.var_95:.2f}%
• Conditional VaR (95%): {metrics.cvar_95:.2f}%

TRADE STATISTICS:
• Total Trades: {metrics.total_trades}
• Winning Trades: {metrics.winning_trades}
• Losing Trades: {metrics.losing_trades}
• Win Rate: {metrics.win_rate:.2%}
• Profit Factor: {metrics.profit_factor:.2f}
• Payoff Ratio: {metrics.payoff_ratio:.2f}
• Average Trade Return: {metrics.average_trade_return:.4f}

MONTHLY PERFORMANCE:
• Best Month: {metrics.best_month:.2f}%
• Worst Month: {metrics.worst_month:.2f}%
• Positive Months: {metrics.positive_months}
• Negative Months: {metrics.negative_months}

ADDITIONAL METRICS:
• Ulcer Index: {metrics.ulcer_index:.2f}
• Gain-to-Pain Ratio: {metrics.gain_to_pain_ratio:.2f}
• Sterling Ratio: {metrics.sterling_ratio:.2f}

======================================================================
        """
        
        return report 