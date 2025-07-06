"""
Value at Risk (VaR) Calculator Module

Comprehensive VaR calculation methods based on risk management concepts
from "Python for Finance" by Yves Hilpisch.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
from scipy import stats
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class VaRMethod(Enum):
    """VaR calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"


class VaRCalculator:
    """
    Comprehensive Value at Risk calculator.
    
    This class implements multiple VaR calculation methods:
    - Historical Simulation
    - Parametric (Normal Distribution)
    - Monte Carlo Simulation
    - Cornish-Fisher Expansion
    - Expected Shortfall (CVaR)
    """
    
    def __init__(self, confidence_levels: List[float] = None):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_levels: List of confidence levels (default: [0.01, 0.05, 0.10])
        """
        self.confidence_levels = confidence_levels or [0.01, 0.05, 0.10]
        logger.info(f"VaRCalculator initialized with confidence levels: {self.confidence_levels}")
    
    def historical_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.05,
        window_size: Optional[int] = None
    ) -> float:
        """
        Calculate VaR using Historical Simulation method.
        
        Historical VaR uses the actual historical distribution of returns
        to estimate potential losses.
        
        Args:
            returns: Historical returns data
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            window_size: Optional rolling window size
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        if window_size is not None:
            returns = returns.tail(window_size)
        
        # Calculate VaR as the percentile of the return distribution
        var = returns.quantile(confidence_level)
        
        return abs(var)
    
    def parametric_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.05,
        distribution: str = 'normal'
    ) -> float:
        """
        Calculate VaR using Parametric method.
        
        Parametric VaR assumes returns follow a specific distribution
        (typically normal) and uses distribution parameters.
        
        Args:
            returns: Historical returns data
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            distribution: Distribution type ('normal', 't', 'skewed_t')
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        if distribution == 'normal':
            # Normal distribution VaR
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(confidence_level)
            var = mean + z_score * std
            
        elif distribution == 't':
            # Student's t-distribution VaR
            params = stats.t.fit(returns)
            var = stats.t.ppf(confidence_level, *params)
            
        elif distribution == 'skewed_t':
            # Skewed t-distribution VaR
            params = stats.skewnorm.fit(returns)
            var = stats.skewnorm.ppf(confidence_level, *params)
            
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        return abs(var)
    
    def monte_carlo_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.05,
        n_simulations: int = 10000,
        forecast_horizon: int = 1
    ) -> float:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Monte Carlo VaR generates random scenarios based on estimated
        parameters and calculates VaR from the simulated distribution.
        
        Args:
            returns: Historical returns data
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            n_simulations: Number of Monte Carlo simulations
            forecast_horizon: Forecast horizon in periods
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        # Estimate parameters from historical data
        mean = returns.mean()
        std = returns.std()
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean, std, n_simulations * forecast_horizon)
        
        if forecast_horizon > 1:
            # Calculate cumulative returns for multi-period horizon
            simulated_returns = simulated_returns.reshape(n_simulations, forecast_horizon)
            cumulative_returns = np.sum(simulated_returns, axis=1)
        else:
            cumulative_returns = simulated_returns
        
        # Calculate VaR from simulated distribution
        var = np.percentile(cumulative_returns, confidence_level * 100)
        
        return abs(var)
    
    def cornish_fisher_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate VaR using Cornish-Fisher expansion.
        
        Cornish-Fisher VaR adjusts the normal VaR for skewness and kurtosis
        to better account for non-normal return distributions.
        
        Args:
            returns: Historical returns data
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        # Calculate moments
        mean = returns.mean()
        std = returns.std()
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Get normal quantile
        z = stats.norm.ppf(confidence_level)
        
        # Apply Cornish-Fisher adjustment
        cf_adjustment = (
            z + 
            (z**2 - 1) * skewness / 6 +
            (z**3 - 3*z) * kurtosis / 24 -
            (2*z**3 - 5*z) * skewness**2 / 36
        )
        
        # Calculate adjusted VaR
        var = mean + cf_adjustment * std
        
        return abs(var)
    
    def expected_shortfall(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.05,
        method: VaRMethod = VaRMethod.HISTORICAL
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Expected Shortfall is the average of all losses that exceed the VaR threshold.
        
        Args:
            returns: Historical returns data
            confidence_level: Confidence level (e.g., 0.05 for 95% ES)
            method: VaR calculation method to use
            
        Returns:
            Expected Shortfall value
        """
        if len(returns) == 0:
            return 0.0
        
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        if method == VaRMethod.HISTORICAL:
            # Historical simulation ES
            var_threshold = returns.quantile(confidence_level)
            tail_losses = returns[returns <= var_threshold]
            
            if len(tail_losses) == 0:
                return 0.0
            
            es = tail_losses.mean()
            
        elif method == VaRMethod.PARAMETRIC:
            # Parametric ES (assuming normal distribution)
            mean = returns.mean()
            std = returns.std()
            
            # Calculate ES using normal distribution formula
            z = stats.norm.ppf(confidence_level)
            es = mean - std * stats.norm.pdf(z) / confidence_level
            
        else:
            # For other methods, use historical approach on simulated data
            if method == VaRMethod.MONTE_CARLO:
                simulated_returns = np.random.normal(
                    returns.mean(), returns.std(), 10000
                )
            else:
                simulated_returns = returns.values
            
            var_threshold = np.percentile(simulated_returns, confidence_level * 100)
            tail_losses = simulated_returns[simulated_returns <= var_threshold]
            es = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
        
        return abs(es)
    
    def rolling_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        window_size: int = 250,
        confidence_level: float = 0.05,
        method: VaRMethod = VaRMethod.HISTORICAL
    ) -> pd.Series:
        """
        Calculate rolling VaR over time.
        
        Args:
            returns: Historical returns data
            window_size: Rolling window size
            confidence_level: Confidence level
            method: VaR calculation method
            
        Returns:
            Series of rolling VaR values
        """
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        
        rolling_vars = []
        
        for i in range(window_size, len(returns) + 1):
            window_returns = returns.iloc[i-window_size:i]
            
            if method == VaRMethod.HISTORICAL:
                var = self.historical_var(window_returns, confidence_level)
            elif method == VaRMethod.PARAMETRIC:
                var = self.parametric_var(window_returns, confidence_level)
            elif method == VaRMethod.MONTE_CARLO:
                var = self.monte_carlo_var(window_returns, confidence_level)
            elif method == VaRMethod.CORNISH_FISHER:
                var = self.cornish_fisher_var(window_returns, confidence_level)
            else:
                var = self.historical_var(window_returns, confidence_level)
            
            rolling_vars.append(var)
        
        # Create series with proper index
        var_series = pd.Series(
            rolling_vars,
            index=returns.index[window_size:]
        )
        
        return var_series
    
    def component_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate Component VaR for portfolio positions.
        
        Component VaR measures how much each position contributes to total portfolio VaR.
        
        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights array
            confidence_level: Confidence level
            
        Returns:
            Dictionary of component VaR values
        """
        if returns.empty:
            return {}
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate portfolio VaR
        portfolio_var = self.historical_var(portfolio_returns, confidence_level)
        
        # Calculate marginal VaR for each asset
        component_vars = {}
        
        for i, asset in enumerate(returns.columns):
            # Calculate correlation with portfolio
            correlation = portfolio_returns.corr(returns[asset])
            
            # Calculate asset volatility
            asset_volatility = returns[asset].std()
            
            # Calculate marginal VaR
            marginal_var = correlation * asset_volatility
            
            # Calculate component VaR
            component_var = weights[i] * marginal_var * portfolio_var / portfolio_returns.std()
            
            component_vars[asset] = component_var
        
        return component_vars
    
    def backtesting_exceptions(
        self,
        returns: Union[pd.Series, np.ndarray],
        var_estimates: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.05
    ) -> Dict[str, Union[int, float, bool]]:
        """
        Perform VaR backtesting to validate model accuracy.
        
        Args:
            returns: Actual returns
            var_estimates: VaR estimates
            confidence_level: Confidence level used for VaR
            
        Returns:
            Dictionary with backtesting results
        """
        returns = pd.Series(returns) if isinstance(returns, np.ndarray) else returns
        var_estimates = pd.Series(var_estimates) if isinstance(var_estimates, np.ndarray) else var_estimates
        
        # Ensure same length
        min_len = min(len(returns), len(var_estimates))
        returns = returns.iloc[-min_len:]
        var_estimates = var_estimates.iloc[-min_len:]
        
        # Count exceptions (losses exceeding VaR)
        exceptions = (returns < -var_estimates).sum()
        
        # Calculate exception rate
        exception_rate = exceptions / len(returns)
        
        # Expected exception rate
        expected_exceptions = confidence_level * len(returns)
        
        # Traffic light test
        if exceptions <= expected_exceptions:
            traffic_light = "GREEN"
        elif exceptions <= expected_exceptions * 1.5:
            traffic_light = "YELLOW"
        else:
            traffic_light = "RED"
        
        # Kupiec test statistic
        if exceptions == 0:
            kupiec_stat = 0
        else:
            kupiec_stat = -2 * np.log(
                (confidence_level**exceptions * (1-confidence_level)**(len(returns)-exceptions)) /
                ((exception_rate**exceptions) * (1-exception_rate)**(len(returns)-exceptions))
            )
        
        # Critical value at 5% significance level
        critical_value = stats.chi2.ppf(0.95, df=1)
        kupiec_reject = kupiec_stat > critical_value
        
        return {
            'exceptions': exceptions,
            'exception_rate': exception_rate,
            'expected_exceptions': expected_exceptions,
            'traffic_light': traffic_light,
            'kupiec_statistic': kupiec_stat,
            'kupiec_reject': kupiec_reject,
            'critical_value': critical_value
        }
    
    def calculate_comprehensive_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: float = 0.05
    ) -> Dict[str, float]:
        """
        Calculate VaR using all available methods.
        
        Args:
            returns: Historical returns data
            confidence_level: Confidence level
            
        Returns:
            Dictionary of VaR estimates from different methods
        """
        var_estimates = {}
        
        # Historical VaR
        var_estimates['historical'] = self.historical_var(returns, confidence_level)
        
        # Parametric VaR
        var_estimates['parametric_normal'] = self.parametric_var(returns, confidence_level, 'normal')
        var_estimates['parametric_t'] = self.parametric_var(returns, confidence_level, 't')
        
        # Monte Carlo VaR
        var_estimates['monte_carlo'] = self.monte_carlo_var(returns, confidence_level)
        
        # Cornish-Fisher VaR
        var_estimates['cornish_fisher'] = self.cornish_fisher_var(returns, confidence_level)
        
        # Expected Shortfall
        var_estimates['expected_shortfall'] = self.expected_shortfall(returns, confidence_level)
        
        return var_estimates
    
    def generate_var_report(
        self,
        returns: Union[pd.Series, np.ndarray],
        asset_name: str = "Asset"
    ) -> str:
        """
        Generate comprehensive VaR report.
        
        Args:
            returns: Historical returns data
            asset_name: Name of the asset
            
        Returns:
            Formatted VaR report
        """
        report = f"""
{'='*60}
VALUE AT RISK ANALYSIS: {asset_name}
{'='*60}

DATA SUMMARY:
• Number of observations: {len(returns)}
• Mean return:           {np.mean(returns):.4f}
• Volatility:            {np.std(returns):.4f}
• Skewness:              {stats.skew(returns):.4f}
• Kurtosis:              {stats.kurtosis(returns):.4f}

"""
        
        # Calculate VaR for different confidence levels
        for conf_level in self.confidence_levels:
            var_estimates = self.calculate_comprehensive_var(returns, conf_level)
            
            report += f"VaR ESTIMATES ({(1-conf_level)*100:.0f}% CONFIDENCE LEVEL):\n"
            for method, var_value in var_estimates.items():
                report += f"• {method:20} {var_value:.4f}\n"
            report += "\n"
        
        report += f"{'='*60}\n"
        
        return report 