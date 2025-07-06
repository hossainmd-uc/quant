"""
Monte Carlo Risk Simulation Module

Advanced Monte Carlo simulation for risk assessment, portfolio simulation,
and scenario analysis based on concepts from "Python for Finance" by Yves Hilpisch.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from scipy import stats
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SimulationParameters:
    """Parameters for Monte Carlo simulation"""
    n_simulations: int = 10000
    n_periods: int = 252
    initial_value: float = 100000.0
    random_seed: Optional[int] = None
    confidence_levels: List[float] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.01, 0.05, 0.10]


@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation"""
    final_values: np.ndarray
    paths: np.ndarray
    statistics: Dict[str, float]
    var_estimates: Dict[str, float]
    cvar_estimates: Dict[str, float]
    percentiles: Dict[str, float]


class MonteCarloRiskSimulator:
    """
    Monte Carlo simulation for risk assessment and portfolio analysis.
    
    This class implements various Monte Carlo methods for:
    - Portfolio value simulation
    - Risk assessment (VaR, CVaR)
    - Scenario analysis
    - Stress testing
    - Option pricing
    """
    
    def __init__(self, parameters: Optional[SimulationParameters] = None):
        """
        Initialize Monte Carlo risk simulator.
        
        Args:
            parameters: Simulation parameters
        """
        self.parameters = parameters or SimulationParameters()
        if self.parameters.random_seed is not None:
            np.random.seed(self.parameters.random_seed)
        
        logger.info(f"MonteCarloRiskSimulator initialized with {self.parameters.n_simulations} simulations")
    
    def geometric_brownian_motion(
        self,
        initial_value: float,
        drift: float,
        volatility: float,
        time_horizon: float,
        n_periods: int,
        n_simulations: int
    ) -> np.ndarray:
        """
        Simulate geometric Brownian motion paths.
        
        The geometric Brownian motion follows:
        dS/S = μ*dt + σ*dW
        
        Args:
            initial_value: Initial asset value
            drift: Drift parameter (μ)
            volatility: Volatility parameter (σ)
            time_horizon: Time horizon in years
            n_periods: Number of time periods
            n_simulations: Number of simulation paths
            
        Returns:
            Array of simulated paths (n_simulations x n_periods)
        """
        dt = time_horizon / n_periods
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (n_simulations, n_periods))
        
        # Calculate price changes
        price_changes = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shocks
        
        # Calculate cumulative price changes
        cumulative_changes = np.cumsum(price_changes, axis=1)
        
        # Calculate asset paths
        paths = initial_value * np.exp(cumulative_changes)
        
        # Add initial value at t=0
        initial_column = np.full((n_simulations, 1), initial_value)
        paths = np.hstack([initial_column, paths])
        
        return paths
    
    def jump_diffusion_simulation(
        self,
        initial_value: float,
        drift: float,
        volatility: float,
        jump_intensity: float,
        jump_size_mean: float,
        jump_size_std: float,
        time_horizon: float,
        n_periods: int,
        n_simulations: int
    ) -> np.ndarray:
        """
        Simulate jump diffusion process (Merton model).
        
        Combines geometric Brownian motion with Poisson jumps:
        dS/S = μ*dt + σ*dW + J*dN
        
        Args:
            initial_value: Initial asset value
            drift: Drift parameter (μ)
            volatility: Volatility parameter (σ)
            jump_intensity: Jump arrival rate (λ)
            jump_size_mean: Mean jump size
            jump_size_std: Jump size standard deviation
            time_horizon: Time horizon in years
            n_periods: Number of time periods
            n_simulations: Number of simulation paths
            
        Returns:
            Array of simulated paths with jumps
        """
        dt = time_horizon / n_periods
        
        # Generate geometric Brownian motion
        gbm_paths = self.geometric_brownian_motion(
            initial_value, drift, volatility, time_horizon, n_periods, n_simulations
        )
        
        # Generate jumps
        jump_times = np.random.poisson(jump_intensity * dt, (n_simulations, n_periods))
        jump_sizes = np.random.normal(jump_size_mean, jump_size_std, (n_simulations, n_periods))
        
        # Apply jumps
        jump_multipliers = np.exp(jump_sizes * jump_times)
        
        # Apply jumps to paths (starting from second column)
        paths = gbm_paths.copy()
        for i in range(1, n_periods + 1):
            paths[:, i] = paths[:, i-1] * jump_multipliers[:, i-1]
        
        return paths
    
    def portfolio_simulation(
        self,
        assets: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
        correlation_matrix: Optional[np.ndarray] = None,
        time_horizon: float = 1.0
    ) -> SimulationResults:
        """
        Simulate portfolio performance using Monte Carlo.
        
        Args:
            assets: Dictionary of asset parameters {asset_name: {drift, volatility}}
            weights: Portfolio weights {asset_name: weight}
            correlation_matrix: Optional correlation matrix between assets
            time_horizon: Time horizon in years
            
        Returns:
            SimulationResults object
        """
        n_assets = len(assets)
        asset_names = list(assets.keys())
        
        # Extract parameters
        drifts = np.array([assets[asset]['drift'] for asset in asset_names])
        volatilities = np.array([assets[asset]['volatility'] for asset in asset_names])
        weight_vector = np.array([weights.get(asset, 0) for asset in asset_names])
        
        # Normalize weights
        weight_vector = weight_vector / np.sum(weight_vector)
        
        # Generate correlated random returns
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_assets)
        
        # Simulate returns
        dt = time_horizon / self.parameters.n_periods
        
        # Generate correlated random shocks
        random_shocks = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=correlation_matrix,
            size=(self.parameters.n_simulations, self.parameters.n_periods)
        )
        
        # Calculate asset returns
        asset_returns = np.zeros((self.parameters.n_simulations, self.parameters.n_periods, n_assets))
        
        for i, asset in enumerate(asset_names):
            drift = drifts[i]
            vol = volatilities[i]
            
            asset_returns[:, :, i] = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * random_shocks[:, :, i]
        
        # Calculate portfolio returns
        portfolio_returns = np.sum(asset_returns * weight_vector, axis=2)
        
        # Calculate portfolio paths
        portfolio_paths = self.parameters.initial_value * np.exp(np.cumsum(portfolio_returns, axis=1))
        
        # Add initial value
        initial_column = np.full((self.parameters.n_simulations, 1), self.parameters.initial_value)
        portfolio_paths = np.hstack([initial_column, portfolio_paths])
        
        # Calculate final values
        final_values = portfolio_paths[:, -1]
        
        # Calculate statistics
        statistics = self._calculate_simulation_statistics(final_values)
        
        # Calculate VaR and CVaR estimates
        var_estimates = {}
        cvar_estimates = {}
        
        for confidence_level in self.parameters.confidence_levels:
            var_estimates[f'VaR_{int(confidence_level*100)}%'] = np.percentile(final_values, confidence_level * 100)
            
            # Calculate CVaR
            var_threshold = var_estimates[f'VaR_{int(confidence_level*100)}%']
            tail_losses = final_values[final_values <= var_threshold]
            cvar_estimates[f'CVaR_{int(confidence_level*100)}%'] = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
        
        # Calculate percentiles
        percentiles = {
            f'P{p}': np.percentile(final_values, p) 
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
        
        return SimulationResults(
            final_values=final_values,
            paths=portfolio_paths,
            statistics=statistics,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            percentiles=percentiles
        )
    
    def scenario_analysis(
        self,
        base_case: Dict[str, float],
        scenarios: Dict[str, Dict[str, float]],
        weights: Dict[str, float]
    ) -> Dict[str, SimulationResults]:
        """
        Perform scenario analysis with different market conditions.
        
        Args:
            base_case: Base case parameters {asset: {drift, volatility}}
            scenarios: Different scenarios {scenario_name: {asset: {drift, volatility}}}
            weights: Portfolio weights
            
        Returns:
            Dictionary of simulation results for each scenario
        """
        results = {}
        
        # Run base case
        results['base_case'] = self.portfolio_simulation(base_case, weights)
        
        # Run scenarios
        for scenario_name, scenario_params in scenarios.items():
            results[scenario_name] = self.portfolio_simulation(scenario_params, weights)
        
        return results
    
    def stress_testing(
        self,
        assets: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
        stress_factors: Dict[str, float]
    ) -> SimulationResults:
        """
        Perform stress testing with extreme market conditions.
        
        Args:
            assets: Base asset parameters
            weights: Portfolio weights
            stress_factors: Stress factors {parameter: multiplier}
            
        Returns:
            SimulationResults under stressed conditions
        """
        # Apply stress factors
        stressed_assets = {}
        for asset, params in assets.items():
            stressed_assets[asset] = {
                'drift': params['drift'] * stress_factors.get('drift_multiplier', 1.0),
                'volatility': params['volatility'] * stress_factors.get('volatility_multiplier', 1.0)
            }
        
        # Run simulation with stressed parameters
        return self.portfolio_simulation(stressed_assets, weights)
    
    def option_pricing_simulation(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Price European options using Monte Carlo simulation.
        
        Args:
            spot_price: Current spot price
            strike_price: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            volatility: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with option price and Greeks
        """
        # Simulate final stock prices
        final_prices = self.geometric_brownian_motion(
            initial_value=spot_price,
            drift=risk_free_rate,
            volatility=volatility,
            time_horizon=time_to_expiry,
            n_periods=1,
            n_simulations=self.parameters.n_simulations
        )[:, -1]
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - strike_price, 0)
        else:  # put
            payoffs = np.maximum(strike_price - final_prices, 0)
        
        # Discount to present value
        option_price = np.exp(-risk_free_rate * time_to_expiry) * np.mean(payoffs)
        
        # Calculate Greeks (simplified using finite differences)
        delta = self._calculate_delta(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type)
        gamma = self._calculate_gamma(spot_price, strike_price, time_to_expiry, risk_free_rate, volatility, option_type)
        
        return {
            'option_price': option_price,
            'delta': delta,
            'gamma': gamma,
            'payoff_std': np.std(payoffs),
            'payoff_mean': np.mean(payoffs)
        }
    
    def _calculate_delta(self, S, K, T, r, sigma, option_type):
        """Calculate Delta using finite differences"""
        dS = S * 0.01  # 1% shift
        
        price_up = self.option_pricing_simulation(S + dS, K, T, r, sigma, option_type)['option_price']
        price_down = self.option_pricing_simulation(S - dS, K, T, r, sigma, option_type)['option_price']
        
        return (price_up - price_down) / (2 * dS)
    
    def _calculate_gamma(self, S, K, T, r, sigma, option_type):
        """Calculate Gamma using finite differences"""
        dS = S * 0.01  # 1% shift
        
        price_up = self.option_pricing_simulation(S + dS, K, T, r, sigma, option_type)['option_price']
        price_center = self.option_pricing_simulation(S, K, T, r, sigma, option_type)['option_price']
        price_down = self.option_pricing_simulation(S - dS, K, T, r, sigma, option_type)['option_price']
        
        return (price_up - 2 * price_center + price_down) / (dS**2)
    
    def _calculate_simulation_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistics from simulation results"""
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
            'sharpe_ratio': np.mean(values) / np.std(values) if np.std(values) != 0 else 0
        }
    
    def generate_simulation_report(self, results: SimulationResults, title: str = "Monte Carlo Simulation") -> str:
        """
        Generate a comprehensive simulation report.
        
        Args:
            results: SimulationResults object
            title: Title for the report
            
        Returns:
            Formatted report string
        """
        report = f"""
{'='*60}
{title.upper()}
{'='*60}

SIMULATION PARAMETERS:
• Number of Simulations:   {self.parameters.n_simulations:,}
• Time Periods:           {self.parameters.n_periods}
• Initial Value:          ${self.parameters.initial_value:,.2f}

FINAL VALUE STATISTICS:
• Mean:                   ${results.statistics['mean']:,.2f}
• Median:                 ${results.statistics['median']:,.2f}
• Standard Deviation:     ${results.statistics['std']:,.2f}
• Minimum:                ${results.statistics['min']:,.2f}
• Maximum:                ${results.statistics['max']:,.2f}
• Skewness:               {results.statistics['skewness']:.4f}
• Kurtosis:               {results.statistics['kurtosis']:.4f}

RISK METRICS:
"""
        
        # Add VaR estimates
        for var_name, var_value in results.var_estimates.items():
            report += f"• {var_name:20} ${var_value:,.2f}\n"
        
        # Add CVaR estimates
        for cvar_name, cvar_value in results.cvar_estimates.items():
            report += f"• {cvar_name:20} ${cvar_value:,.2f}\n"
        
        report += f"""
PERCENTILES:
"""
        # Add percentiles
        for percentile_name, percentile_value in results.percentiles.items():
            report += f"• {percentile_name:20} ${percentile_value:,.2f}\n"
        
        report += f"""
{'='*60}
"""
        
        return report
    
    def export_results(self, results: SimulationResults, filename: str):
        """
        Export simulation results to CSV file.
        
        Args:
            results: SimulationResults object
            filename: Output filename
        """
        # Create DataFrame with results
        df = pd.DataFrame({
            'simulation_id': range(len(results.final_values)),
            'final_value': results.final_values
        })
        
        # Add percentile rankings
        df['percentile_rank'] = df['final_value'].rank(pct=True)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        logger.info(f"Simulation results exported to {filename}")
    
    def visualize_results(self, results: SimulationResults):
        """
        Create visualizations of simulation results.
        
        Args:
            results: SimulationResults object
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Final value distribution
            axes[0, 0].hist(results.final_values, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Final Value Distribution')
            axes[0, 0].set_xlabel('Final Value')
            axes[0, 0].set_ylabel('Frequency')
            
            # Plot 2: Sample paths
            sample_paths = results.paths[:100]  # Show first 100 paths
            for path in sample_paths:
                axes[0, 1].plot(path, alpha=0.1, color='blue')
            axes[0, 1].set_title('Sample Simulation Paths')
            axes[0, 1].set_xlabel('Time Period')
            axes[0, 1].set_ylabel('Portfolio Value')
            
            # Plot 3: Q-Q plot
            stats.probplot(results.final_values, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot vs Normal Distribution')
            
            # Plot 4: Risk metrics
            var_values = list(results.var_estimates.values())
            var_labels = list(results.var_estimates.keys())
            axes[1, 1].bar(var_labels, var_values)
            axes[1, 1].set_title('Value at Risk Estimates')
            axes[1, 1].set_ylabel('VaR Value')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}") 