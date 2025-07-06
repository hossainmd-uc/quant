"""
Advanced Portfolio Optimization Module
=====================================

This module contains advanced portfolio optimization techniques implementing
113 portfolio theory concepts from "Python for Finance: Mastering Data-Driven Finance".

The module includes:
- Modern Portfolio Theory (MPT) enhancements
- Black-Litterman model implementation
- Risk parity optimization
- Factor-based portfolio construction
- Multi-objective optimization
- Robust optimization techniques
- Alternative risk measures
- Dynamic asset allocation
- Transaction cost optimization
- Regime-aware portfolio management
"""

from .modern_portfolio_theory import ModernPortfolioTheory, EfficientFrontier
from .black_litterman import BlackLittermanOptimizer
from .risk_parity import RiskParityOptimizer, HierarchicalRiskParity
from .factor_models import FactorBasedOptimizer, FamaFrenchOptimizer
from .multi_objective import MultiObjectiveOptimizer, ParetoBounds
from .robust_optimization import RobustOptimizer, UncertaintySetOptimizer
from .alternative_risk import AlternativeRiskOptimizer, CVaROptimizer
from .dynamic_allocation import DynamicAssetAllocator, TacticalAssetAllocation
from .transaction_costs import TransactionCostOptimizer, TurnoverOptimizer
from .regime_models import RegimeAwareOptimizer, MarkovRegimeOptimizer

__all__ = [
    # Modern Portfolio Theory
    'ModernPortfolioTheory',
    'EfficientFrontier',
    
    # Black-Litterman
    'BlackLittermanOptimizer',
    
    # Risk Parity
    'RiskParityOptimizer',
    'HierarchicalRiskParity',
    
    # Factor Models
    'FactorBasedOptimizer',
    'FamaFrenchOptimizer',
    
    # Multi-Objective
    'MultiObjectiveOptimizer',
    'ParetoBounds',
    
    # Robust Optimization
    'RobustOptimizer',
    'UncertaintySetOptimizer',
    
    # Alternative Risk
    'AlternativeRiskOptimizer',
    'CVaROptimizer',
    
    # Dynamic Allocation
    'DynamicAssetAllocator',
    'TacticalAssetAllocation',
    
    # Transaction Costs
    'TransactionCostOptimizer',
    'TurnoverOptimizer',
    
    # Regime Models
    'RegimeAwareOptimizer',
    'MarkovRegimeOptimizer'
] 