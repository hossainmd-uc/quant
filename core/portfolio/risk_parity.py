"""
Risk Parity Optimization
=======================

Implementation of risk parity and risk budgeting portfolio optimization techniques.
These methods focus on equalizing risk contributions rather than capital allocations.

Key Features:
- Equal Risk Contribution (ERC) portfolios
- Hierarchical Risk Parity (HRP)
- Risk budgeting optimization
- Naive risk parity
- Volatility targeting
- Risk factor decomposition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import optimize
from scipy.cluster.hierarchy import dendrogram, linkage, ClusterWarning
from scipy.spatial.distance import squareform
import warnings
from dataclasses import dataclass

@dataclass
class RiskBudget:
    """
    Risk budget specification for portfolio optimization.
    """
    asset_budgets: Optional[Dict[str, float]] = None  # Asset-level risk budgets
    group_budgets: Optional[Dict[str, float]] = None  # Group-level risk budgets
    asset_groups: Optional[Dict[str, str]] = None     # Asset to group mapping
    
    def __post_init__(self):
        """Validate risk budgets."""
        if self.asset_budgets is not None:
            if not np.isclose(sum(self.asset_budgets.values()), 1.0):
                raise ValueError("Asset risk budgets must sum to 1")
        
        if self.group_budgets is not None:
            if not np.isclose(sum(self.group_budgets.values()), 1.0):
                raise ValueError("Group risk budgets must sum to 1")

class RiskParityOptimizer:
    """
    Risk parity portfolio optimizer.
    
    Implements various risk parity techniques that focus on equalizing
    risk contributions rather than capital allocations.
    """
    
    def __init__(self, method: str = 'equal_risk_contribution'):
        """
        Initialize risk parity optimizer.
        
        Args:
            method: Risk parity method ('equal_risk_contribution', 'risk_budgeting', 'naive')
        """
        self.method = method
        self.returns_data = None
        self.covariance_matrix = None
        self.assets = None
        
        if method not in ['equal_risk_contribution', 'risk_budgeting', 'naive']:
            raise ValueError("Method must be 'equal_risk_contribution', 'risk_budgeting', or 'naive'")
    
    def fit(self, returns_data: pd.DataFrame) -> None:
        """
        Fit the model with historical returns data.
        
        Args:
            returns_data: DataFrame of asset returns
        """
        self.returns_data = returns_data.copy()
        self.assets = returns_data.columns.tolist()
        self.covariance_matrix = returns_data.cov().values
    
    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate risk contributions for given weights.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Array of risk contributions
        """
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
        marginal_contrib = np.dot(self.covariance_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        return risk_contrib / np.sum(risk_contrib)  # Normalize to sum to 1
    
    def optimize_equal_risk_contribution(self, max_iter: int = 1000, 
                                       tolerance: float = 1e-8) -> Dict:
        """
        Optimize for equal risk contribution (ERC) portfolio.
        
        Args:
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(self.assets)
        
        # Objective function: sum of squared deviations from equal risk contribution
        def objective(weights):
            risk_contrib = self.calculate_risk_contributions(weights)
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraint: weights sum to 1
        constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Bounds: long only
        bounds = [(0.001, 1.0) for _ in range(n_assets)]  # Small lower bound to avoid division by zero
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraint,
            options={'maxiter': max_iter, 'ftol': tolerance}
        )
        
        if result.success:
            weights = result.x
            risk_contrib = self.calculate_risk_contributions(weights)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
            
            return {
                'weights': pd.Series(weights, index=self.assets),
                'risk_contributions': pd.Series(risk_contrib, index=self.assets),
                'portfolio_volatility': portfolio_vol,
                'success': True,
                'message': 'Optimization successful',
                'iterations': result.nit
            }
        else:
            return {
                'weights': pd.Series(np.ones(n_assets) / n_assets, index=self.assets),
                'risk_contributions': pd.Series(np.ones(n_assets) / n_assets, index=self.assets),
                'portfolio_volatility': 0.0,
                'success': False,
                'message': f'Optimization failed: {result.message}',
                'iterations': result.nit
            }
    
    def optimize_risk_budgeting(self, risk_budget: RiskBudget, 
                              max_iter: int = 1000, tolerance: float = 1e-8) -> Dict:
        """
        Optimize for risk budgeting portfolio.
        
        Args:
            risk_budget: Risk budget specification
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(self.assets)
        
        # Get target risk contributions
        if risk_budget.asset_budgets is not None:
            target_contrib = np.array([risk_budget.asset_budgets.get(asset, 0.0) 
                                     for asset in self.assets])
        else:
            # Equal risk contribution as default
            target_contrib = np.ones(n_assets) / n_assets
        
        # Objective function: sum of squared deviations from target risk contributions
        def objective(weights):
            risk_contrib = self.calculate_risk_contributions(weights)
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraint: weights sum to 1
        constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Bounds: long only
        bounds = [(0.001, 1.0) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraint,
            options={'maxiter': max_iter, 'ftol': tolerance}
        )
        
        if result.success:
            weights = result.x
            risk_contrib = self.calculate_risk_contributions(weights)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
            
            return {
                'weights': pd.Series(weights, index=self.assets),
                'risk_contributions': pd.Series(risk_contrib, index=self.assets),
                'target_contributions': pd.Series(target_contrib, index=self.assets),
                'portfolio_volatility': portfolio_vol,
                'success': True,
                'message': 'Optimization successful',
                'iterations': result.nit
            }
        else:
            return {
                'weights': pd.Series(np.ones(n_assets) / n_assets, index=self.assets),
                'risk_contributions': pd.Series(np.ones(n_assets) / n_assets, index=self.assets),
                'target_contributions': pd.Series(target_contrib, index=self.assets),
                'portfolio_volatility': 0.0,
                'success': False,
                'message': f'Optimization failed: {result.message}',
                'iterations': result.nit
            }
    
    def optimize_naive_risk_parity(self) -> Dict:
        """
        Calculate naive risk parity portfolio (inverse volatility weighting).
        
        Returns:
            Dictionary with optimization results
        """
        # Calculate individual asset volatilities
        asset_vols = np.sqrt(np.diag(self.covariance_matrix))
        
        # Inverse volatility weights
        inv_vol_weights = 1.0 / asset_vols
        weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        # Calculate risk contributions
        risk_contrib = self.calculate_risk_contributions(weights)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
        
        return {
            'weights': pd.Series(weights, index=self.assets),
            'risk_contributions': pd.Series(risk_contrib, index=self.assets),
            'portfolio_volatility': portfolio_vol,
            'individual_volatilities': pd.Series(asset_vols, index=self.assets),
            'success': True,
            'message': 'Naive risk parity calculated'
        }
    
    def optimize(self, risk_budget: Optional[RiskBudget] = None, **kwargs) -> Dict:
        """
        Optimize portfolio using the specified method.
        
        Args:
            risk_budget: Risk budget specification (for risk budgeting method)
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimization results
        """
        if self.method == 'equal_risk_contribution':
            return self.optimize_equal_risk_contribution(**kwargs)
        elif self.method == 'risk_budgeting':
            if risk_budget is None:
                risk_budget = RiskBudget()  # Default to equal risk contribution
            return self.optimize_risk_budgeting(risk_budget, **kwargs)
        elif self.method == 'naive':
            return self.optimize_naive_risk_parity()
        else:
            raise ValueError(f"Unknown method: {self.method}")

class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) portfolio optimizer.
    
    HRP uses hierarchical clustering to build a portfolio allocation
    that accounts for the hierarchical structure of asset correlations.
    """
    
    def __init__(self, linkage_method: str = 'single'):
        """
        Initialize HRP optimizer.
        
        Args:
            linkage_method: Linkage method for clustering ('single', 'complete', 'average', 'ward')
        """
        self.linkage_method = linkage_method
        self.returns_data = None
        self.correlation_matrix = None
        self.assets = None
        self.clusters = None
        
    def fit(self, returns_data: pd.DataFrame) -> None:
        """
        Fit the model with historical returns data.
        
        Args:
            returns_data: DataFrame of asset returns
        """
        self.returns_data = returns_data.copy()
        self.assets = returns_data.columns.tolist()
        self.correlation_matrix = returns_data.corr().values
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """
        Calculate distance matrix from correlation matrix.
        
        Returns:
            Distance matrix
        """
        # Convert correlation to distance: d = sqrt(0.5 * (1 - corr))
        distance_matrix = np.sqrt(0.5 * (1 - self.correlation_matrix))
        np.fill_diagonal(distance_matrix, 0)
        return distance_matrix
    
    def perform_clustering(self) -> np.ndarray:
        """
        Perform hierarchical clustering.
        
        Returns:
            Linkage matrix
        """
        distance_matrix = self.calculate_distance_matrix()
        
        # Convert to condensed distance matrix
        condensed_distances = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ClusterWarning)
            linkage_matrix = linkage(condensed_distances, method=self.linkage_method)
        
        self.clusters = linkage_matrix
        return linkage_matrix
    
    def get_cluster_variance(self, cluster_items: List[int]) -> float:
        """
        Calculate variance of a cluster.
        
        Args:
            cluster_items: List of asset indices in the cluster
            
        Returns:
            Cluster variance
        """
        if len(cluster_items) == 1:
            return self.returns_data.iloc[:, cluster_items[0]].var()
        
        # Equal-weight variance of the cluster
        cluster_returns = self.returns_data.iloc[:, cluster_items]
        equal_weight_returns = cluster_returns.mean(axis=1)
        return equal_weight_returns.var()
    
    def get_quasi_diagonalization(self, linkage_matrix: np.ndarray) -> List[int]:
        """
        Get quasi-diagonalization order from linkage matrix.
        
        Args:
            linkage_matrix: Hierarchical clustering linkage matrix
            
        Returns:
            List of asset indices in quasi-diagonal order
        """
        # Recursive function to get the order
        def get_order(cluster_id, n_assets):
            if cluster_id < n_assets:
                return [cluster_id]
            else:
                left_child = int(linkage_matrix[cluster_id - n_assets, 0])
                right_child = int(linkage_matrix[cluster_id - n_assets, 1])
                return get_order(left_child, n_assets) + get_order(right_child, n_assets)
        
        n_assets = len(self.assets)
        return get_order(len(linkage_matrix) + n_assets - 1, n_assets)
    
    def recursive_bisection(self, cluster_items: List[int]) -> np.ndarray:
        """
        Recursive bisection to allocate weights.
        
        Args:
            cluster_items: List of asset indices in the cluster
            
        Returns:
            Array of weights for the cluster items
        """
        n_items = len(cluster_items)
        
        # Base case: single asset
        if n_items == 1:
            return np.array([1.0])
        
        # Base case: two assets
        if n_items == 2:
            var1 = self.get_cluster_variance([cluster_items[0]])
            var2 = self.get_cluster_variance([cluster_items[1]])
            
            # Inverse variance weighting
            w1 = 1.0 / var1
            w2 = 1.0 / var2
            total_weight = w1 + w2
            
            return np.array([w1 / total_weight, w2 / total_weight])
        
        # Recursive case: split the cluster
        mid = n_items // 2
        left_cluster = cluster_items[:mid]
        right_cluster = cluster_items[mid:]
        
        # Calculate cluster variances
        left_var = self.get_cluster_variance(left_cluster)
        right_var = self.get_cluster_variance(right_cluster)
        
        # Allocate weight between left and right clusters
        left_weight = 1.0 / left_var
        right_weight = 1.0 / right_var
        total_weight = left_weight + right_weight
        
        left_allocation = left_weight / total_weight
        right_allocation = right_weight / total_weight
        
        # Recursive allocation within each cluster
        left_weights = self.recursive_bisection(left_cluster) * left_allocation
        right_weights = self.recursive_bisection(right_cluster) * right_allocation
        
        return np.concatenate([left_weights, right_weights])
    
    def optimize(self) -> Dict:
        """
        Optimize portfolio using HRP.
        
        Returns:
            Dictionary with optimization results
        """
        if self.correlation_matrix is None:
            raise ValueError("Must fit model first")
        
        # Perform clustering
        linkage_matrix = self.perform_clustering()
        
        # Get quasi-diagonalization order
        quasi_order = self.get_quasi_diagonalization(linkage_matrix)
        
        # Allocate weights using recursive bisection
        weights = self.recursive_bisection(quasi_order)
        
        # Reorder weights to match original asset order
        ordered_weights = np.zeros(len(self.assets))
        for i, asset_idx in enumerate(quasi_order):
            ordered_weights[asset_idx] = weights[i]
        
        # Calculate portfolio metrics
        portfolio_vol = np.sqrt(np.dot(ordered_weights, 
                                     np.dot(self.returns_data.cov().values, ordered_weights)))
        
        return {
            'weights': pd.Series(ordered_weights, index=self.assets),
            'portfolio_volatility': portfolio_vol,
            'quasi_order': [self.assets[i] for i in quasi_order],
            'linkage_matrix': linkage_matrix,
            'success': True,
            'message': 'HRP optimization successful'
        }
    
    def plot_dendrogram(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot hierarchical clustering dendrogram.
        
        Args:
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.clusters is None:
                self.perform_clustering()
            
            plt.figure(figsize=figsize)
            dendrogram(self.clusters, labels=self.assets, orientation='top')
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Assets')
            plt.ylabel('Distance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def get_cluster_allocation(self, n_clusters: int = 3) -> Dict:
        """
        Get cluster-based allocation.
        
        Args:
            n_clusters: Number of clusters to form
            
        Returns:
            Dictionary with cluster allocations
        """
        from scipy.cluster.hierarchy import fcluster
        
        if self.clusters is None:
            self.perform_clustering()
        
        # Form clusters
        cluster_labels = fcluster(self.clusters, n_clusters, criterion='maxclust')
        
        # Group assets by cluster
        clusters = {}
        for i, asset in enumerate(self.assets):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(asset)
        
        return clusters 