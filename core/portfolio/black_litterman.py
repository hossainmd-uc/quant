"""
Black-Litterman Model
====================

Implementation of the Black-Litterman model for portfolio optimization.
This model enhances traditional Mean-Variance Optimization by incorporating
market views and their associated confidence levels.

Key Features:
- Market equilibrium implied returns
- Investor views incorporation
- Confidence-weighted optimization
- Bayesian approach to return estimation
- Robust parameter estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import linalg
import warnings
from dataclasses import dataclass

@dataclass
class MarketView:
    """
    Represents an investor's view about asset returns.
    """
    assets: List[str]  # Assets involved in the view
    weights: List[float]  # Weights for each asset in the view
    expected_return: float  # Expected return for this view
    confidence: float  # Confidence level (0-1, higher = more confident)
    
    def __post_init__(self):
        """Validate the view."""
        if len(self.assets) != len(self.weights):
            raise ValueError("Assets and weights must have the same length")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")

class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimizer.
    
    The Black-Litterman model starts with market equilibrium assumptions
    and allows investors to incorporate their views about future returns.
    """
    
    def __init__(self, risk_aversion: float = 3.0, tau: float = 0.05):
        """
        Initialize Black-Litterman optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter (higher = more risk averse)
            tau: Scaling factor for uncertainty of prior (typically 0.01-0.1)
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.returns_data = None
        self.market_caps = None
        self.covariance_matrix = None
        self.assets = None
        self.equilibrium_returns = None
        self.views = []
        
    def fit(self, returns_data: pd.DataFrame, market_caps: Optional[pd.Series] = None) -> None:
        """
        Fit the model with historical returns data.
        
        Args:
            returns_data: DataFrame of asset returns
            market_caps: Series of market capitalizations (optional)
        """
        self.returns_data = returns_data.copy()
        self.assets = returns_data.columns.tolist()
        
        # Calculate covariance matrix
        self.covariance_matrix = returns_data.cov().values
        
        # Set market caps (equal weight if not provided)
        if market_caps is not None:
            self.market_caps = market_caps.reindex(self.assets)
        else:
            self.market_caps = pd.Series(1.0, index=self.assets)
        
        # Calculate equilibrium returns
        self._calculate_equilibrium_returns()
    
    def _calculate_equilibrium_returns(self) -> None:
        """
        Calculate implied equilibrium returns using reverse optimization.
        
        These are the returns that would justify the current market capitalization
        weights as optimal under mean-variance optimization.
        """
        # Market capitalization weights
        market_weights = self.market_caps / self.market_caps.sum()
        w_market = market_weights.values
        
        # Implied equilibrium returns: π = δ * Σ * w_market
        # where δ is risk aversion, Σ is covariance matrix
        self.equilibrium_returns = self.risk_aversion * np.dot(self.covariance_matrix, w_market)
    
    def add_view(self, view: MarketView) -> None:
        """
        Add a market view.
        
        Args:
            view: Market view to add
        """
        # Validate that all assets in the view exist
        for asset in view.assets:
            if asset not in self.assets:
                raise ValueError(f"Asset {asset} not found in the universe")
        
        self.views.append(view)
    
    def add_absolute_view(self, asset: str, expected_return: float, confidence: float) -> None:
        """
        Add an absolute view (single asset expected return).
        
        Args:
            asset: Asset name
            expected_return: Expected return for the asset
            confidence: Confidence level (0-1)
        """
        view = MarketView(
            assets=[asset],
            weights=[1.0],
            expected_return=expected_return,
            confidence=confidence
        )
        self.add_view(view)
    
    def add_relative_view(self, asset1: str, asset2: str, 
                         expected_outperformance: float, confidence: float) -> None:
        """
        Add a relative view (asset1 vs asset2).
        
        Args:
            asset1: First asset
            asset2: Second asset
            expected_outperformance: Expected outperformance of asset1 vs asset2
            confidence: Confidence level (0-1)
        """
        view = MarketView(
            assets=[asset1, asset2],
            weights=[1.0, -1.0],
            expected_return=expected_outperformance,
            confidence=confidence
        )
        self.add_view(view)
    
    def _construct_view_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct the P, Q, and Ω matrices for Black-Litterman.
        
        Returns:
            Tuple of (P, Q, Omega) matrices
        """
        if not self.views:
            # No views - return empty matrices
            return np.zeros((0, len(self.assets))), np.zeros(0), np.zeros((0, 0))
        
        n_views = len(self.views)
        n_assets = len(self.assets)
        
        # P matrix: links views to assets
        P = np.zeros((n_views, n_assets))
        
        # Q vector: view expected returns
        Q = np.zeros(n_views)
        
        # Omega matrix: view uncertainty
        Omega = np.zeros((n_views, n_views))
        
        for i, view in enumerate(self.views):
            # Fill P matrix
            for j, asset in enumerate(view.assets):
                asset_idx = self.assets.index(asset)
                P[i, asset_idx] = view.weights[j]
            
            # Fill Q vector
            Q[i] = view.expected_return
            
            # Fill Omega matrix (diagonal with view variances)
            # Higher confidence = lower variance
            view_variance = (1 - view.confidence) * np.dot(P[i], np.dot(self.covariance_matrix, P[i]))
            Omega[i, i] = view_variance
        
        return P, Q, Omega
    
    def calculate_posterior_returns(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate posterior expected returns and covariance matrix.
        
        Returns:
            Tuple of (posterior_returns, posterior_covariance)
        """
        if self.equilibrium_returns is None:
            raise ValueError("Must fit model first")
        
        # Prior parameters
        mu_prior = self.equilibrium_returns
        sigma_prior = self.tau * self.covariance_matrix
        
        # If no views, return prior
        if not self.views:
            return mu_prior, sigma_prior
        
        # Construct view matrices
        P, Q, Omega = self._construct_view_matrices()
        
        # Black-Litterman formula
        # Posterior precision = Prior precision + View precision
        prior_precision = linalg.inv(sigma_prior)
        view_precision = np.dot(P.T, np.dot(linalg.inv(Omega), P))
        posterior_precision = prior_precision + view_precision
        
        # Posterior covariance
        posterior_covariance = linalg.inv(posterior_precision)
        
        # Posterior mean
        prior_term = np.dot(prior_precision, mu_prior)
        view_term = np.dot(P.T, np.dot(linalg.inv(Omega), Q))
        posterior_returns = np.dot(posterior_covariance, prior_term + view_term)
        
        return posterior_returns, posterior_covariance
    
    def optimize_portfolio(self, target_volatility: Optional[float] = None,
                         target_return: Optional[float] = None,
                         constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio using Black-Litterman inputs.
        
        Args:
            target_volatility: Target portfolio volatility (optional)
            target_return: Target portfolio return (optional)
            constraints: Additional constraints (optional)
            
        Returns:
            Dictionary with optimization results
        """
        # Calculate posterior parameters
        mu_bl, sigma_bl = self.calculate_posterior_returns()
        
        # Optimize portfolio
        if target_volatility is not None:
            # Minimize risk for target return
            weights = self._optimize_target_volatility(mu_bl, sigma_bl, target_volatility)
        elif target_return is not None:
            # Minimize risk for target return
            weights = self._optimize_target_return(mu_bl, sigma_bl, target_return)
        else:
            # Maximize utility (mean-variance optimization)
            weights = self._optimize_utility(mu_bl, sigma_bl)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, mu_bl)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(sigma_bl, weights)))
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': pd.Series(weights, index=self.assets),
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'posterior_returns': pd.Series(mu_bl, index=self.assets),
            'posterior_covariance': pd.DataFrame(sigma_bl, index=self.assets, columns=self.assets)
        }
    
    def _optimize_utility(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Optimize utility function (mean-variance).
        
        Args:
            mu: Expected returns
            sigma: Covariance matrix
            
        Returns:
            Optimal weights
        """
        # Analytical solution for mean-variance optimization
        # w = (1/δ) * Σ^(-1) * μ / (1^T * Σ^(-1) * μ)
        
        sigma_inv = linalg.inv(sigma)
        numerator = np.dot(sigma_inv, mu)
        denominator = np.dot(np.ones(len(mu)), numerator)
        
        weights = numerator / denominator
        return weights
    
    def _optimize_target_return(self, mu: np.ndarray, sigma: np.ndarray, 
                              target_return: float) -> np.ndarray:
        """
        Optimize for target return (minimize risk).
        
        Args:
            mu: Expected returns
            sigma: Covariance matrix
            target_return: Target portfolio return
            
        Returns:
            Optimal weights
        """
        from scipy.optimize import minimize
        
        n_assets = len(mu)
        
        # Objective function (portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(sigma, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target_return}  # Target return
        ]
        
        # Bounds (long only)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(portfolio_variance, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def _optimize_target_volatility(self, mu: np.ndarray, sigma: np.ndarray, 
                                  target_volatility: float) -> np.ndarray:
        """
        Optimize for target volatility (maximize return).
        
        Args:
            mu: Expected returns
            sigma: Covariance matrix
            target_volatility: Target portfolio volatility
            
        Returns:
            Optimal weights
        """
        from scipy.optimize import minimize
        
        n_assets = len(mu)
        
        # Objective function (negative portfolio return)
        def negative_return(weights):
            return -np.dot(weights, mu)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.sqrt(np.dot(w, np.dot(sigma, w))) - target_volatility}  # Target volatility
        ]
        
        # Bounds (long only)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(negative_return, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x if result.success else x0
    
    def get_view_impact(self) -> pd.DataFrame:
        """
        Analyze the impact of each view on the portfolio.
        
        Returns:
            DataFrame with view impact analysis
        """
        if not self.views:
            return pd.DataFrame()
        
        # Calculate returns without views
        mu_no_views = self.equilibrium_returns
        
        # Calculate returns with all views
        mu_with_views, _ = self.calculate_posterior_returns()
        
        # Impact of all views
        total_impact = mu_with_views - mu_no_views
        
        # Try to isolate individual view impacts (approximate)
        view_impacts = []
        
        for i, view in enumerate(self.views):
            # Create temporary optimizer with only this view
            temp_optimizer = BlackLittermanOptimizer(self.risk_aversion, self.tau)
            temp_optimizer.returns_data = self.returns_data
            temp_optimizer.market_caps = self.market_caps
            temp_optimizer.covariance_matrix = self.covariance_matrix
            temp_optimizer.assets = self.assets
            temp_optimizer.equilibrium_returns = self.equilibrium_returns
            temp_optimizer.views = [view]
            
            mu_single_view, _ = temp_optimizer.calculate_posterior_returns()
            single_view_impact = mu_single_view - mu_no_views
            
            view_impacts.append({
                'view_index': i,
                'view_description': f"{view.assets} -> {view.expected_return:.3f}",
                'confidence': view.confidence,
                'impact': single_view_impact
            })
        
        # Create DataFrame
        impact_df = pd.DataFrame(view_impacts)
        
        # Add asset-level impacts
        for j, asset in enumerate(self.assets):
            impact_df[f'{asset}_impact'] = [impact['impact'][j] for impact in view_impacts]
        
        return impact_df
    
    def clear_views(self) -> None:
        """Clear all views."""
        self.views = []
    
    def get_equilibrium_weights(self) -> pd.Series:
        """
        Get market equilibrium weights.
        
        Returns:
            Series of equilibrium weights
        """
        if self.market_caps is None:
            raise ValueError("Must fit model first")
        
        weights = self.market_caps / self.market_caps.sum()
        return weights
    
    def get_equilibrium_returns(self) -> pd.Series:
        """
        Get implied equilibrium returns.
        
        Returns:
            Series of equilibrium returns
        """
        if self.equilibrium_returns is None:
            raise ValueError("Must fit model first")
        
        return pd.Series(self.equilibrium_returns, index=self.assets) 