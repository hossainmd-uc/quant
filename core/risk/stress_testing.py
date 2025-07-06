"""
Stress Testing Module

Advanced stress testing and scenario analysis for risk management
based on concepts from "Python for Finance" by Yves Hilpisch.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from loguru import logger


class StressTester:
    """
    Advanced stress testing for portfolio and risk analysis.
    """
    
    def __init__(self):
        """Initialize stress tester."""
        logger.info("StressTester initialized")
    
    def market_crash_scenario(
        self,
        returns: pd.DataFrame,
        crash_magnitude: float = -0.20,
        recovery_periods: int = 60
    ) -> pd.DataFrame:
        """
        Simulate market crash scenario.
        
        Args:
            returns: Historical returns
            crash_magnitude: Size of initial crash
            recovery_periods: Number of periods for recovery
            
        Returns:
            Stressed return series
        """
        stressed_returns = returns.copy()
        
        # Apply initial crash
        crash_day = len(stressed_returns) // 2
        stressed_returns.iloc[crash_day] = crash_magnitude
        
        # Apply gradual recovery
        recovery_rate = abs(crash_magnitude) / recovery_periods
        for i in range(1, recovery_periods + 1):
            if crash_day + i < len(stressed_returns):
                stressed_returns.iloc[crash_day + i] += recovery_rate * (1 - i/recovery_periods)
        
        return stressed_returns
    
    def volatility_shock(
        self,
        returns: pd.DataFrame,
        volatility_multiplier: float = 2.0
    ) -> pd.DataFrame:
        """
        Apply volatility shock to returns.
        
        Args:
            returns: Historical returns
            volatility_multiplier: Volatility multiplication factor
            
        Returns:
            Stressed return series with increased volatility
        """
        mean_returns = returns.mean()
        centered_returns = returns - mean_returns
        
        # Scale volatility
        stressed_returns = mean_returns + centered_returns * volatility_multiplier
        
        return stressed_returns 