"""
Risk Metrics Module

Comprehensive risk metrics calculations based on modern portfolio theory
and risk management concepts from "Python for Finance" by Yves Hilpisch.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
from loguru import logger


class RiskMetrics:
    """
    Comprehensive risk metrics calculator for portfolio and trading analysis.
    
    This class implements various risk measures including:
    - Sharpe Ratio and variations
    - Sortino Ratio
    - Maximum Drawdown
    - Value at Risk (VaR)
    - Conditional Value at Risk (CVaR)
    - Beta and Alpha calculations
    - Volatility measures
    - Correlation analysis
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"RiskMetrics initialized with risk-free rate: {risk_free_rate:.2%}")
    
    def calculate_sharpe_ratio(
        self, 
        returns: Union[pd.Series, np.ndarray], 
        periods: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        The Sharpe ratio measures risk-adjusted return by dividing excess return
        by the standard deviation of returns.
        
        Args:
            returns: Series of returns
            periods: Number of periods per year for annualization
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        # Calculate excess returns
        excess_returns = returns - self.risk_free_rate / periods
        
        # Calculate Sharpe ratio
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods)
        return sharpe
    
    def calculate_sortino_ratio(
        self, 
        returns: Union[pd.Series, np.ndarray], 
        periods: int = 252
    ) -> float:
        """
        Calculate Sortino ratio.
        
        The Sortino ratio is similar to the Sharpe ratio but only considers
        downside volatility, making it more appropriate for asymmetric return distributions.
        
        Args:
            returns: Series of returns
            periods: Number of periods per year for annualization
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        # Calculate excess returns
        excess_returns = returns - self.risk_free_rate / periods
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_deviation * np.sqrt(periods)
        return sortino
    
    def calculate_maximum_drawdown(
        self, 
        returns: Union[pd.Series, np.ndarray]
    ) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration.
        
        Maximum drawdown is the largest peak-to-trough decline in portfolio value.
        
        Args:
            returns: Series of returns
            
        Returns:
            Tuple of (max_drawdown, duration_days)
        """
        if len(returns) == 0:
            return 0.0, 0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Calculate duration
        drawdown_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                drawdown_duration = max(drawdown_duration, current_duration)
            else:
                current_duration = 0
        
        return abs(max_drawdown), drawdown_duration
    
    def calculate_var(
        self, 
        returns: Union[pd.Series, np.ndarray], 
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation.
        
        VaR represents the maximum expected loss over a given time period
        at a specified confidence level.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        # Calculate VaR using historical simulation
        var = returns.quantile(confidence_level)
        return abs(var)
    
    def calculate_cvar(
        self, 
        returns: Union[pd.Series, np.ndarray], 
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.
        
        CVaR is the expected loss given that the loss exceeds the VaR threshold.
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% CVaR)
            
        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        # Calculate VaR threshold
        var_threshold = returns.quantile(confidence_level)
        
        # Calculate CVaR as mean of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]
        if len(tail_returns) == 0:
            return 0.0
        
        cvar = tail_returns.mean()
        return abs(cvar)
    
    def calculate_beta(
        self, 
        asset_returns: Union[pd.Series, np.ndarray],
        market_returns: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate beta coefficient.
        
        Beta measures the sensitivity of an asset's returns to market returns.
        
        Args:
            asset_returns: Series of asset returns
            market_returns: Series of market returns
            
        Returns:
            Beta coefficient
        """
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        asset_returns = pd.Series(asset_returns) if isinstance(asset_returns, np.ndarray) else asset_returns
        market_returns = pd.Series(market_returns) if isinstance(market_returns, np.ndarray) else market_returns
        
        # Ensure same length
        min_len = min(len(asset_returns), len(market_returns))
        asset_returns = asset_returns.iloc[-min_len:]
        market_returns = market_returns.iloc[-min_len:]
        
        # Calculate covariance and variance
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 0.0
        
        beta = covariance / market_variance
        return beta
    
    def calculate_alpha(
        self, 
        asset_returns: Union[pd.Series, np.ndarray],
        market_returns: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate alpha coefficient.
        
        Alpha measures the excess return of an asset relative to what would be
        expected given its beta and the market return.
        
        Args:
            asset_returns: Series of asset returns
            market_returns: Series of market returns
            
        Returns:
            Alpha coefficient (annualized)
        """
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        asset_returns = pd.Series(asset_returns) if isinstance(asset_returns, np.ndarray) else asset_returns
        market_returns = pd.Series(market_returns) if isinstance(market_returns, np.ndarray) else market_returns
        
        # Calculate beta
        beta = self.calculate_beta(asset_returns, market_returns)
        
        # Calculate alpha using CAPM formula
        asset_return = asset_returns.mean() * 252  # Annualized
        market_return = market_returns.mean() * 252  # Annualized
        
        alpha = asset_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))
        return alpha
    
    def calculate_information_ratio(
        self, 
        asset_returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate Information Ratio.
        
        The Information Ratio measures the consistency of outperformance
        relative to a benchmark.
        
        Args:
            asset_returns: Series of asset returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Information Ratio
        """
        if len(asset_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        asset_returns = pd.Series(asset_returns) if isinstance(asset_returns, np.ndarray) else asset_returns
        benchmark_returns = pd.Series(benchmark_returns) if isinstance(benchmark_returns, np.ndarray) else benchmark_returns
        
        # Calculate excess returns
        excess_returns = asset_returns - benchmark_returns
        
        # Calculate Information Ratio
        if excess_returns.std() == 0:
            return 0.0
        
        ir = excess_returns.mean() / excess_returns.std()
        return ir
    
    def calculate_calmar_ratio(
        self, 
        returns: Union[pd.Series, np.ndarray],
        periods: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio.
        
        The Calmar Ratio is the annualized return divided by maximum drawdown.
        
        Args:
            returns: Series of returns
            periods: Number of periods per year for annualization
            
        Returns:
            Calmar Ratio
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        # Calculate annualized return
        annualized_return = returns.mean() * periods
        
        # Calculate maximum drawdown
        max_drawdown, _ = self.calculate_maximum_drawdown(returns)
        
        if max_drawdown == 0:
            return float('inf')
        
        calmar = annualized_return / max_drawdown
        return calmar
    
    def calculate_volatility(
        self, 
        returns: Union[pd.Series, np.ndarray],
        periods: int = 252
    ) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Series of returns
            periods: Number of periods per year for annualization
            
        Returns:
            Annualized volatility
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        volatility = returns.std() * np.sqrt(periods)
        return volatility
    
    def calculate_skewness(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate skewness of returns.
        
        Skewness measures the asymmetry of the return distribution.
        
        Args:
            returns: Series of returns
            
        Returns:
            Skewness
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        skewness = stats.skew(returns)
        return skewness
    
    def calculate_kurtosis(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate kurtosis of returns.
        
        Kurtosis measures the "tailedness" of the return distribution.
        
        Args:
            returns: Series of returns
            
        Returns:
            Kurtosis
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        kurtosis = stats.kurtosis(returns)
        return kurtosis
    
    def calculate_comprehensive_metrics(
        self, 
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for relative metrics
            
        Returns:
            Dictionary of risk metrics
        """
        if len(returns) == 0:
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns)
        metrics['volatility'] = self.calculate_volatility(returns)
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns)
        
        # Drawdown metrics
        max_dd, dd_duration = self.calculate_maximum_drawdown(returns)
        metrics['max_drawdown'] = max_dd
        metrics['drawdown_duration'] = dd_duration
        
        # Risk metrics
        metrics['var_95'] = self.calculate_var(returns, 0.05)
        metrics['cvar_95'] = self.calculate_cvar(returns, 0.05)
        metrics['var_99'] = self.calculate_var(returns, 0.01)
        metrics['cvar_99'] = self.calculate_cvar(returns, 0.01)
        
        # Distribution metrics
        metrics['skewness'] = self.calculate_skewness(returns)
        metrics['kurtosis'] = self.calculate_kurtosis(returns)
        
        # Relative metrics (if benchmark provided)
        if benchmark_returns is not None:
            metrics['beta'] = self.calculate_beta(returns, benchmark_returns)
            metrics['alpha'] = self.calculate_alpha(returns, benchmark_returns)
            metrics['information_ratio'] = self.calculate_information_ratio(returns, benchmark_returns)
        
        return metrics
    
    def generate_risk_report(
        self, 
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
        asset_name: str = "Asset"
    ) -> str:
        """
        Generate a comprehensive risk report.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns
            asset_name: Name of the asset for the report
            
        Returns:
            Formatted risk report string
        """
        metrics = self.calculate_comprehensive_metrics(returns, benchmark_returns)
        
        report = f"""
{'='*60}
RISK ANALYSIS REPORT: {asset_name}
{'='*60}

PERFORMANCE METRICS:
• Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.4f}
• Sortino Ratio:          {metrics.get('sortino_ratio', 0):.4f}
• Calmar Ratio:           {metrics.get('calmar_ratio', 0):.4f}
• Annualized Volatility:  {metrics.get('volatility', 0):.4f}

DRAWDOWN ANALYSIS:
• Maximum Drawdown:       {metrics.get('max_drawdown', 0):.4f} ({metrics.get('max_drawdown', 0)*100:.2f}%)
• Drawdown Duration:      {metrics.get('drawdown_duration', 0)} periods

RISK METRICS:
• Value at Risk (95%):    {metrics.get('var_95', 0):.4f}
• Conditional VaR (95%):  {metrics.get('cvar_95', 0):.4f}
• Value at Risk (99%):    {metrics.get('var_99', 0):.4f}
• Conditional VaR (99%):  {metrics.get('cvar_99', 0):.4f}

DISTRIBUTION ANALYSIS:
• Skewness:              {metrics.get('skewness', 0):.4f}
• Kurtosis:              {metrics.get('kurtosis', 0):.4f}
"""
        
        if benchmark_returns is not None:
            report += f"""
RELATIVE METRICS:
• Beta:                  {metrics.get('beta', 0):.4f}
• Alpha:                 {metrics.get('alpha', 0):.4f}
• Information Ratio:     {metrics.get('information_ratio', 0):.4f}
"""
        
        report += f"""
{'='*60}
"""
        
        return report 