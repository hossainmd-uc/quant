"""
Trading Strategies Module
========================

This module contains various algorithmic trading strategies implemented based on
concepts from "Python for Finance: Mastering Data-Driven Finance".

The strategies include:
- Base strategy framework
- Momentum strategies
- Mean reversion strategies
- Trend following strategies
- Pairs trading strategies
- Statistical arbitrage
- Machine learning strategies
- Technical indicator strategies
"""

from .base_strategy import BaseStrategy, StrategySignal
from .momentum_strategies import MomentumStrategy, RSIMomentumStrategy, BollingerBandsStrategy
from .mean_reversion_strategies import MeanReversionStrategy, PairsTrading, StatisticalArbitrage
from .trend_following_strategies import TrendFollowingStrategy, MovingAverageCrossover, BreakoutStrategy
from .ml_strategies import MLStrategy, LinearRegressionStrategy, RandomForestStrategy
from .technical_strategies import TechnicalStrategy, MACDStrategy, StochasticStrategy

__all__ = [
    # Base classes
    'BaseStrategy',
    'StrategySignal',
    
    # Momentum strategies
    'MomentumStrategy',
    'RSIMomentumStrategy', 
    'BollingerBandsStrategy',
    
    # Mean reversion strategies
    'MeanReversionStrategy',
    'PairsTrading',
    'StatisticalArbitrage',
    
    # Trend following strategies
    'TrendFollowingStrategy',
    'MovingAverageCrossover',
    'BreakoutStrategy',
    
    # ML strategies
    'MLStrategy',
    'LinearRegressionStrategy',
    'RandomForestStrategy',
    
    # Technical strategies
    'TechnicalStrategy',
    'MACDStrategy',
    'StochasticStrategy'
] 