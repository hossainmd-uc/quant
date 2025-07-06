"""
Portfolio Optimizer Module

Advanced portfolio optimization based on Modern Portfolio Theory
and concepts from "Python for Finance" by Yves Hilpisch.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy.optimize import minimize
from scipy.stats import norm
from loguru import logger


class PortfolioOptimizer:
    """
    Modern Portfolio Theory optimizer with multiple optimization objectives.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"PortfolioOptimizer initialized with risk-free rate: {risk_free_rate:.2%}")
    
    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        objective: str = 'sharpe',
        constraints: Optional[Dict] = None
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize portfolio based on specified objective.
        
        Args:
            returns: DataFrame of asset returns
            objective: Optimization objective ('sharpe', 'min_vol', 'max_return')
            constraints: Optional constraints dictionary
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        n_assets = len(returns.columns)
        
        # Default constraints
        if constraints is None:
            constraints = {
                'sum_weights': 1.0,
                'weight_bounds': (0, 1),
                'max_weight': 0.4
            }
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Bounds for weights
        bounds = tuple(constraints.get('weight_bounds', (0, 1)) for _ in range(n_assets))
        
        # Constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - constraints.get('sum_weights', 1.0)}
        ]
        
        # Maximum weight constraint
        if 'max_weight' in constraints:
            max_weight = constraints['max_weight']
            for i in range(n_assets):
                constraint_list.append({
                    'type': 'ineq', 
                    'fun': lambda x, i=i: max_weight - x[i]
                })
        
        # Objective function
        if objective == 'sharpe':
            objective_func = lambda x: -self._calculate_sharpe_ratio(x, returns)
        elif objective == 'min_vol':
            objective_func = lambda x: self._calculate_portfolio_volatility(x, returns)
        elif objective == 'max_return':
            objective_func = lambda x: -self._calculate_portfolio_return(x, returns)
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Optimize
        result = minimize(
            objective_func,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
        
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = self._calculate_portfolio_return(optimal_weights, returns)
        portfolio_volatility = self._calculate_portfolio_volatility(optimal_weights, returns)
        sharpe_ratio = self._calculate_sharpe_ratio(optimal_weights, returns)
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'success': result.success
        }
    
    def _calculate_portfolio_return(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Calculate portfolio expected return"""
        return np.sum(returns.mean() * weights) * 252
    
    def _calculate_portfolio_volatility(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Calculate portfolio volatility"""
        cov_matrix = returns.cov() * 252
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def _calculate_sharpe_ratio(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Calculate Sharpe ratio"""
        portfolio_return = self._calculate_portfolio_return(weights, returns)
        portfolio_volatility = self._calculate_portfolio_volatility(weights, returns)
        
        if portfolio_volatility == 0:
            return 0
        
        return (portfolio_return - self.risk_free_rate) / portfolio_volatility 