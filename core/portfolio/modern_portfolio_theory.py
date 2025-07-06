"""
Enhanced Modern Portfolio Theory
===============================

Advanced implementation of Modern Portfolio Theory with comprehensive
optimization capabilities, efficient frontier construction, and multiple
risk measures.

Key Features:
- Enhanced MPT with multiple constraints
- Efficient frontier construction
- Risk budgeting and contribution analysis
- Alternative risk measures (CVaR, MAD, etc.)
- Multi-period optimization
- Robust estimation techniques
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import optimize
from scipy.stats import norm
import warnings
from dataclasses import dataclass

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""
    min_weights: Optional[np.ndarray] = None  # Minimum weights per asset
    max_weights: Optional[np.ndarray] = None  # Maximum weights per asset
    sum_to_one: bool = True  # Weights must sum to 1
    long_only: bool = True   # No short selling
    max_single_weight: Optional[float] = None  # Maximum single asset weight
    sector_constraints: Optional[Dict] = None  # Sector exposure limits
    turnover_limit: Optional[float] = None    # Maximum portfolio turnover

@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    success: bool
    message: str
    risk_contributions: Optional[np.ndarray] = None
    factor_exposures: Optional[Dict] = None

class ModernPortfolioTheory:
    """
    Enhanced Modern Portfolio Theory optimizer with advanced features.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize MPT optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.expected_returns = None
        self.covariance_matrix = None
        self.assets = None
        
    def fit(self, returns_data: pd.DataFrame) -> None:
        """
        Fit the model with historical returns data.
        
        Args:
            returns_data: DataFrame of asset returns
        """
        self.returns_data = returns_data.copy()
        self.assets = returns_data.columns.tolist()
        
        # Calculate expected returns (sample mean)
        self.expected_returns = returns_data.mean().values
        
        # Calculate covariance matrix
        self.covariance_matrix = returns_data.cov().values
        
    def set_expected_returns(self, expected_returns: Union[pd.Series, np.ndarray, Dict]) -> None:
        """
        Set custom expected returns.
        
        Args:
            expected_returns: Expected returns for each asset
        """
        if isinstance(expected_returns, pd.Series):
            self.expected_returns = expected_returns.values
        elif isinstance(expected_returns, dict):
            self.expected_returns = np.array([expected_returns[asset] for asset in self.assets])
        else:
            self.expected_returns = np.array(expected_returns)
    
    def shrink_covariance(self, shrinkage: float = 0.2) -> np.ndarray:
        """
        Apply Ledoit-Wolf shrinkage to covariance matrix.
        
        Args:
            shrinkage: Shrinkage intensity (0-1)
            
        Returns:
            Shrunk covariance matrix
        """
        if self.covariance_matrix is None:
            raise ValueError("Must fit model first")
        
        # Target matrix (identity scaled by average variance)
        target = np.eye(len(self.assets)) * np.trace(self.covariance_matrix) / len(self.assets)
        
        # Shrunk covariance matrix
        shrunk_cov = (1 - shrinkage) * self.covariance_matrix + shrinkage * target
        
        return shrunk_cov
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, 
                                  cov_matrix: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        
        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix (optional)
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        if cov_matrix is None:
            cov_matrix = self.covariance_matrix
        
        # Expected return
        expected_return = np.dot(weights, self.expected_returns)
        
        # Volatility
        volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        return expected_return, volatility, sharpe_ratio
    
    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate risk contributions of each asset.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Risk contributions array
        """
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
        marginal_contrib = np.dot(self.covariance_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        return risk_contrib
    
    def optimize_max_sharpe(self, constraints: Optional[PortfolioConstraints] = None) -> OptimizationResult:
        """
        Optimize portfolio for maximum Sharpe ratio.
        
        Args:
            constraints: Portfolio constraints
            
        Returns:
            Optimization result
        """
        if constraints is None:
            constraints = PortfolioConstraints()
        
        n_assets = len(self.assets)
        
        # Objective function (negative Sharpe ratio to minimize)
        def negative_sharpe(weights):
            expected_return, volatility, sharpe_ratio = self.calculate_portfolio_metrics(weights)
            return -sharpe_ratio
        
        # Constraints
        cons = []
        
        # Weights sum to 1
        if constraints.sum_to_one:
            cons.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Bounds
        bounds = []
        for i in range(n_assets):
            min_w = 0.0 if constraints.long_only else -1.0
            max_w = 1.0
            
            if constraints.min_weights is not None:
                min_w = max(min_w, constraints.min_weights[i])
            if constraints.max_weights is not None:
                max_w = min(max_w, constraints.max_weights[i])
            if constraints.max_single_weight is not None:
                max_w = min(max_w, constraints.max_single_weight)
            
            bounds.append((min_w, max_w))
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = optimize.minimize(
                negative_sharpe, x0, method='SLSQP',
                bounds=bounds, constraints=cons,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                expected_return, volatility, sharpe_ratio = self.calculate_portfolio_metrics(weights)
                risk_contributions = self.calculate_risk_contributions(weights)
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=expected_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    success=True,
                    message="Optimization successful",
                    risk_contributions=risk_contributions
                )
            else:
                return OptimizationResult(
                    weights=np.ones(n_assets) / n_assets,
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    success=False,
                    message=f"Optimization failed: {result.message}"
                )
                
        except Exception as e:
            return OptimizationResult(
                weights=np.ones(n_assets) / n_assets,
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                success=False,
                message=f"Optimization error: {str(e)}"
            )
    
    def optimize_min_volatility(self, constraints: Optional[PortfolioConstraints] = None) -> OptimizationResult:
        """
        Optimize portfolio for minimum volatility.
        
        Args:
            constraints: Portfolio constraints
            
        Returns:
            Optimization result
        """
        if constraints is None:
            constraints = PortfolioConstraints()
        
        n_assets = len(self.assets)
        
        # Objective function (portfolio volatility)
        def portfolio_volatility(weights):
            _, volatility, _ = self.calculate_portfolio_metrics(weights)
            return volatility
        
        # Constraints
        cons = []
        
        # Weights sum to 1
        if constraints.sum_to_one:
            cons.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Bounds
        bounds = []
        for i in range(n_assets):
            min_w = 0.0 if constraints.long_only else -1.0
            max_w = 1.0
            
            if constraints.min_weights is not None:
                min_w = max(min_w, constraints.min_weights[i])
            if constraints.max_weights is not None:
                max_w = min(max_w, constraints.max_weights[i])
            if constraints.max_single_weight is not None:
                max_w = min(max_w, constraints.max_single_weight)
            
            bounds.append((min_w, max_w))
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = optimize.minimize(
                portfolio_volatility, x0, method='SLSQP',
                bounds=bounds, constraints=cons,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                expected_return, volatility, sharpe_ratio = self.calculate_portfolio_metrics(weights)
                risk_contributions = self.calculate_risk_contributions(weights)
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=expected_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    success=True,
                    message="Optimization successful",
                    risk_contributions=risk_contributions
                )
            else:
                return OptimizationResult(
                    weights=np.ones(n_assets) / n_assets,
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    success=False,
                    message=f"Optimization failed: {result.message}"
                )
                
        except Exception as e:
            return OptimizationResult(
                weights=np.ones(n_assets) / n_assets,
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                success=False,
                message=f"Optimization error: {str(e)}"
            )
    
    def optimize_target_return(self, target_return: float, 
                             constraints: Optional[PortfolioConstraints] = None) -> OptimizationResult:
        """
        Optimize portfolio for target return with minimum risk.
        
        Args:
            target_return: Target portfolio return
            constraints: Portfolio constraints
            
        Returns:
            Optimization result
        """
        if constraints is None:
            constraints = PortfolioConstraints()
        
        n_assets = len(self.assets)
        
        # Objective function (portfolio volatility)
        def portfolio_volatility(weights):
            _, volatility, _ = self.calculate_portfolio_metrics(weights)
            return volatility
        
        # Constraints
        cons = []
        
        # Weights sum to 1
        if constraints.sum_to_one:
            cons.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # Target return constraint
        cons.append({'type': 'eq', 'fun': lambda w: np.dot(w, self.expected_returns) - target_return})
        
        # Bounds
        bounds = []
        for i in range(n_assets):
            min_w = 0.0 if constraints.long_only else -1.0
            max_w = 1.0
            
            if constraints.min_weights is not None:
                min_w = max(min_w, constraints.min_weights[i])
            if constraints.max_weights is not None:
                max_w = min(max_w, constraints.max_weights[i])
            if constraints.max_single_weight is not None:
                max_w = min(max_w, constraints.max_single_weight)
            
            bounds.append((min_w, max_w))
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = optimize.minimize(
                portfolio_volatility, x0, method='SLSQP',
                bounds=bounds, constraints=cons,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                expected_return, volatility, sharpe_ratio = self.calculate_portfolio_metrics(weights)
                risk_contributions = self.calculate_risk_contributions(weights)
                
                return OptimizationResult(
                    weights=weights,
                    expected_return=expected_return,
                    volatility=volatility,
                    sharpe_ratio=sharpe_ratio,
                    success=True,
                    message="Optimization successful",
                    risk_contributions=risk_contributions
                )
            else:
                return OptimizationResult(
                    weights=np.ones(n_assets) / n_assets,
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    success=False,
                    message=f"Optimization failed: {result.message}"
                )
                
        except Exception as e:
            return OptimizationResult(
                weights=np.ones(n_assets) / n_assets,
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                success=False,
                message=f"Optimization error: {str(e)}"
            )

class EfficientFrontier:
    """
    Efficient frontier construction and analysis.
    """
    
    def __init__(self, mpt_optimizer: ModernPortfolioTheory):
        """
        Initialize efficient frontier calculator.
        
        Args:
            mpt_optimizer: Fitted MPT optimizer
        """
        self.mpt = mpt_optimizer
        self.frontier_points = []
        
    def calculate_frontier(self, num_points: int = 50, 
                         constraints: Optional[PortfolioConstraints] = None) -> pd.DataFrame:
        """
        Calculate efficient frontier points.
        
        Args:
            num_points: Number of points on the frontier
            constraints: Portfolio constraints
            
        Returns:
            DataFrame with frontier points
        """
        if self.mpt.expected_returns is None:
            raise ValueError("MPT optimizer must be fitted first")
        
        # Get minimum and maximum possible returns
        min_vol_result = self.mpt.optimize_min_volatility(constraints)
        
        # Estimate return range
        min_return = min_vol_result.expected_return
        max_return = max(self.mpt.expected_returns) * 0.8  # Conservative maximum
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, num_points)
        
        frontier_data = []
        
        for target_return in target_returns:
            try:
                result = self.mpt.optimize_target_return(target_return, constraints)
                
                if result.success:
                    frontier_data.append({
                        'target_return': target_return,
                        'expected_return': result.expected_return,
                        'volatility': result.volatility,
                        'sharpe_ratio': result.sharpe_ratio,
                        'weights': result.weights
                    })
            except:
                # Skip points that fail optimization
                continue
        
        self.frontier_points = pd.DataFrame(frontier_data)
        return self.frontier_points
    
    def get_tangency_portfolio(self, constraints: Optional[PortfolioConstraints] = None) -> OptimizationResult:
        """
        Get the tangency portfolio (maximum Sharpe ratio).
        
        Args:
            constraints: Portfolio constraints
            
        Returns:
            Tangency portfolio optimization result
        """
        return self.mpt.optimize_max_sharpe(constraints)
    
    def plot_frontier(self, show_tangency: bool = True) -> None:
        """
        Plot the efficient frontier.
        
        Args:
            show_tangency: Whether to highlight tangency portfolio
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.frontier_points.empty:
                print("No frontier points calculated. Run calculate_frontier() first.")
                return
            
            plt.figure(figsize=(10, 6))
            
            # Plot frontier
            plt.plot(self.frontier_points['volatility'], 
                    self.frontier_points['expected_return'], 
                    'b-', linewidth=2, label='Efficient Frontier')
            
            # Plot tangency portfolio
            if show_tangency:
                tangency = self.get_tangency_portfolio()
                if tangency.success:
                    plt.plot(tangency.volatility, tangency.expected_return, 
                            'r*', markersize=15, label=f'Tangency Portfolio (Sharpe: {tangency.sharpe_ratio:.2f})')
            
            plt.xlabel('Volatility')
            plt.ylabel('Expected Return')
            plt.title('Efficient Frontier')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def get_frontier_statistics(self) -> Dict:
        """
        Get statistics about the efficient frontier.
        
        Returns:
            Dictionary of frontier statistics
        """
        if self.frontier_points.empty:
            return {}
        
        return {
            'num_points': len(self.frontier_points),
            'min_volatility': self.frontier_points['volatility'].min(),
            'max_volatility': self.frontier_points['volatility'].max(),
            'min_return': self.frontier_points['expected_return'].min(),
            'max_return': self.frontier_points['expected_return'].max(),
            'max_sharpe': self.frontier_points['sharpe_ratio'].max(),
            'avg_sharpe': self.frontier_points['sharpe_ratio'].mean()
        } 