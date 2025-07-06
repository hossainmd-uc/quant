"""
Risk Management Module

This module provides comprehensive risk management capabilities including:
- Value at Risk (VaR) calculations
- Monte Carlo simulation for risk assessment
- Portfolio optimization using Modern Portfolio Theory
- Risk metrics calculation (Sharpe ratio, Sortino ratio, etc.)
- Stress testing and scenario analysis
- Correlation analysis and diversification metrics

Based on concepts from "Python for Finance" by Yves Hilpisch.
"""

from .risk_metrics import RiskMetrics
from .var_calculator import VaRCalculator
from .monte_carlo import MonteCarloRiskSimulator
from .portfolio_optimizer import PortfolioOptimizer
from .stress_testing import StressTester

__all__ = [
    "RiskMetrics",
    "VaRCalculator", 
    "MonteCarloRiskSimulator",
    "PortfolioOptimizer",
    "StressTester"
] 