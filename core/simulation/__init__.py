"""
Comprehensive Trading Simulation System

This module provides a complete trading simulation framework that integrates:
- Portfolio simulation with realistic market conditions
- Position sizing and risk management
- Transaction costs and slippage modeling
- Strategy testing and backtesting
- Performance analysis and reporting
- Integration with ML models, risk management, and portfolio optimization

Key Components:
- PortfolioSimulator: Core simulation engine
- MarketSimulator: Realistic market conditions
- PositionSizer: Dynamic position sizing
- TransactionCostModel: Realistic cost modeling
- PerformanceAnalyzer: Comprehensive performance metrics
- StrategyTester: Backtesting framework
"""

from .portfolio_simulator import PortfolioSimulator, SimulationResult
from .market_simulator import MarketSimulator, MarketConditions
from .position_sizer import PositionSizer, PositionSizeResult
from .transaction_costs import TransactionCostModel, TransactionCost
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .strategy_tester import StrategyTester, BacktestResult

__all__ = [
    'PortfolioSimulator',
    'SimulationResult',
    'MarketSimulator',
    'MarketConditions',
    'PositionSizer',
    'PositionSizeResult',
    'TransactionCostModel',
    'TransactionCost',
    'PerformanceAnalyzer',
    'PerformanceMetrics',
    'StrategyTester',
    'BacktestResult'
] 