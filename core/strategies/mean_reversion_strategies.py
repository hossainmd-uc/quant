"""
Mean Reversion Trading Strategies
================================

This module implements mean reversion strategies that capitalize on the tendency
of asset prices to revert to their historical averages.

Strategies included:
- Basic Mean Reversion Strategy
- Pairs Trading Strategy
- Statistical Arbitrage Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .base_strategy import BaseStrategy, StrategySignal
from scipy import stats

class MeanReversionStrategy(BaseStrategy):
    """
    Basic mean reversion strategy that trades when prices deviate significantly
    from their historical average.
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 **kwargs):
        """
        Initialize the mean reversion strategy.
        
        Args:
            lookback_period: Number of periods for calculating mean and std
            entry_threshold: Number of standard deviations for entry
            exit_threshold: Number of standard deviations for exit
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="Mean Reversion Strategy", **kwargs)
        
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # Update parameters
        self.parameters.update({
            'lookback_period': lookback_period,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold
        })
    
    def calculate_z_score(self, symbol: str, timestamp: datetime) -> float:
        """
        Calculate Z-score for mean reversion analysis.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            Z-score value
        """
        if symbol not in self.data:
            return 0.0
        
        data = self.data[symbol]
        mask = data.index <= timestamp
        prices = data.loc[mask, 'Close']
        
        if len(prices) < self.lookback_period:
            return 0.0
        
        # Calculate rolling statistics
        recent_prices = prices.iloc[-self.lookback_period:]
        mean_price = recent_prices.mean()
        std_price = recent_prices.std()
        
        if std_price == 0:
            return 0.0
        
        current_price = prices.iloc[-1]
        z_score = (current_price - mean_price) / std_price
        
        return z_score
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate mean reversion signals."""
        signals = []
        
        for symbol in self.data.keys():
            try:
                # Calculate Z-score
                z_score = self.calculate_z_score(symbol, timestamp)
                
                if abs(z_score) < self.entry_threshold:
                    continue
                
                # Get current price
                data = self.data[symbol]
                mask = data.index <= timestamp
                current_price = data.loc[mask, 'Close'].iloc[-1]
                
                # Generate signal based on Z-score
                signal_type = 'HOLD'
                strength = 0.0
                
                if z_score > self.entry_threshold:
                    # Price is above mean - sell (expect reversion down)
                    signal_type = 'SELL'
                    strength = min((z_score - self.entry_threshold) / self.entry_threshold, 1.0)
                elif z_score < -self.entry_threshold:
                    # Price is below mean - buy (expect reversion up)
                    signal_type = 'BUY'
                    strength = min((abs(z_score) - self.entry_threshold) / self.entry_threshold, 1.0)
                
                if signal_type != 'HOLD' and strength > 0:
                    # Calculate quantity
                    base_quantity = 100
                    quantity = int(base_quantity * strength)
                    
                    signal = StrategySignal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=strength,
                        price=current_price,
                        quantity=quantity,
                        metadata={
                            'z_score': z_score,
                            'lookback_period': self.lookback_period,
                            'entry_threshold': self.entry_threshold,
                            'strategy': 'mean_reversion'
                        }
                    )
                    signals.append(signal)
                    
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the mean reversion strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        # Validate parameters
        if self.lookback_period < 2:
            raise ValueError("Lookback period must be at least 2")
        
        if self.entry_threshold <= 0:
            raise ValueError("Entry threshold must be positive")
        
        if self.exit_threshold <= 0:
            raise ValueError("Exit threshold must be positive")
        
        self.is_initialized = True

class PairsTrading(BaseStrategy):
    """
    Pairs trading strategy that trades the spread between two correlated assets.
    """
    
    def __init__(self, 
                 pairs: List[Tuple[str, str]],
                 lookback_period: int = 60,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 min_correlation: float = 0.7,
                 **kwargs):
        """
        Initialize the pairs trading strategy.
        
        Args:
            pairs: List of (asset1, asset2) tuples to trade
            lookback_period: Period for calculating spread statistics
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            min_correlation: Minimum correlation for valid pairs
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="Pairs Trading Strategy", **kwargs)
        
        self.pairs = pairs
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_correlation = min_correlation
        
        # Update parameters
        self.parameters.update({
            'pairs': pairs,
            'lookback_period': lookback_period,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'min_correlation': min_correlation
        })
    
    def calculate_spread(self, symbol1: str, symbol2: str, timestamp: datetime) -> Tuple[float, float]:
        """
        Calculate the spread between two assets and its Z-score.
        
        Args:
            symbol1: First asset symbol
            symbol2: Second asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            Tuple of (spread, z_score)
        """
        if symbol1 not in self.data or symbol2 not in self.data:
            return 0.0, 0.0
        
        # Get price data
        data1 = self.data[symbol1]
        data2 = self.data[symbol2]
        
        mask1 = data1.index <= timestamp
        mask2 = data2.index <= timestamp
        
        prices1 = data1.loc[mask1, 'Close']
        prices2 = data2.loc[mask2, 'Close']
        
        if len(prices1) < self.lookback_period or len(prices2) < self.lookback_period:
            return 0.0, 0.0
        
        # Align prices by taking the last N periods
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1.iloc[-min_len:]
        prices2 = prices2.iloc[-min_len:]
        
        # Calculate spread (log price ratio)
        spread = np.log(prices1 / prices2)
        
        if len(spread) < self.lookback_period:
            return 0.0, 0.0
        
        # Calculate rolling statistics
        recent_spread = spread.iloc[-self.lookback_period:]
        mean_spread = recent_spread.mean()
        std_spread = recent_spread.std()
        
        if std_spread == 0:
            return 0.0, 0.0
        
        current_spread = spread.iloc[-1]
        z_score = (current_spread - mean_spread) / std_spread
        
        return current_spread, z_score
    
    def check_correlation(self, symbol1: str, symbol2: str, timestamp: datetime) -> float:
        """
        Check correlation between two assets.
        
        Args:
            symbol1: First asset symbol
            symbol2: Second asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            Correlation coefficient
        """
        if symbol1 not in self.data or symbol2 not in self.data:
            return 0.0
        
        # Get returns data
        returns1 = self.calculate_returns(symbol1)
        returns2 = self.calculate_returns(symbol2)
        
        # Filter by timestamp
        mask1 = returns1.index <= timestamp
        mask2 = returns2.index <= timestamp
        
        returns1 = returns1.loc[mask1]
        returns2 = returns2.loc[mask2]
        
        if len(returns1) < self.lookback_period or len(returns2) < self.lookback_period:
            return 0.0
        
        # Align returns
        common_dates = returns1.index.intersection(returns2.index)
        if len(common_dates) < self.lookback_period:
            return 0.0
        
        returns1_aligned = returns1.loc[common_dates]
        returns2_aligned = returns2.loc[common_dates]
        
        # Calculate correlation
        correlation = returns1_aligned.corr(returns2_aligned)
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate pairs trading signals."""
        signals = []
        
        for symbol1, symbol2 in self.pairs:
            try:
                # Check if both symbols have data
                if symbol1 not in self.data or symbol2 not in self.data:
                    continue
                
                # Check correlation
                correlation = self.check_correlation(symbol1, symbol2, timestamp)
                if abs(correlation) < self.min_correlation:
                    continue
                
                # Calculate spread and Z-score
                spread, z_score = self.calculate_spread(symbol1, symbol2, timestamp)
                
                if abs(z_score) < self.entry_threshold:
                    continue
                
                # Get current prices
                data1 = self.data[symbol1]
                data2 = self.data[symbol2]
                
                mask1 = data1.index <= timestamp
                mask2 = data2.index <= timestamp
                
                price1 = data1.loc[mask1, 'Close'].iloc[-1]
                price2 = data2.loc[mask2, 'Close'].iloc[-1]
                
                # Generate signals
                strength = min(abs(z_score) / self.entry_threshold, 1.0)
                base_quantity = 100
                quantity = int(base_quantity * strength)
                
                if z_score > self.entry_threshold:
                    # Spread is high - sell symbol1, buy symbol2
                    signal1 = StrategySignal(
                        timestamp=timestamp,
                        symbol=symbol1,
                        signal_type='SELL',
                        strength=strength,
                        price=price1,
                        quantity=quantity,
                        metadata={
                            'pair': (symbol1, symbol2),
                            'spread': spread,
                            'z_score': z_score,
                            'correlation': correlation,
                            'strategy': 'pairs_trading'
                        }
                    )
                    
                    signal2 = StrategySignal(
                        timestamp=timestamp,
                        symbol=symbol2,
                        signal_type='BUY',
                        strength=strength,
                        price=price2,
                        quantity=quantity,
                        metadata={
                            'pair': (symbol1, symbol2),
                            'spread': spread,
                            'z_score': z_score,
                            'correlation': correlation,
                            'strategy': 'pairs_trading'
                        }
                    )
                    
                    signals.extend([signal1, signal2])
                
                elif z_score < -self.entry_threshold:
                    # Spread is low - buy symbol1, sell symbol2
                    signal1 = StrategySignal(
                        timestamp=timestamp,
                        symbol=symbol1,
                        signal_type='BUY',
                        strength=strength,
                        price=price1,
                        quantity=quantity,
                        metadata={
                            'pair': (symbol1, symbol2),
                            'spread': spread,
                            'z_score': z_score,
                            'correlation': correlation,
                            'strategy': 'pairs_trading'
                        }
                    )
                    
                    signal2 = StrategySignal(
                        timestamp=timestamp,
                        symbol=symbol2,
                        signal_type='SELL',
                        strength=strength,
                        price=price2,
                        quantity=quantity,
                        metadata={
                            'pair': (symbol1, symbol2),
                            'spread': spread,
                            'z_score': z_score,
                            'correlation': correlation,
                            'strategy': 'pairs_trading'
                        }
                    )
                    
                    signals.extend([signal1, signal2])
                    
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the pairs trading strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        if not self.pairs:
            raise ValueError("No pairs specified for trading")
        
        # Validate all pairs have data
        for symbol1, symbol2 in self.pairs:
            if symbol1 not in self.data or symbol2 not in self.data:
                raise ValueError(f"Missing data for pair ({symbol1}, {symbol2})")
        
        # Validate parameters
        if self.lookback_period < 2:
            raise ValueError("Lookback period must be at least 2")
        
        if self.entry_threshold <= 0:
            raise ValueError("Entry threshold must be positive")
        
        if not (0 <= self.min_correlation <= 1):
            raise ValueError("Minimum correlation must be between 0 and 1")
        
        self.is_initialized = True

class StatisticalArbitrage(BaseStrategy):
    """
    Statistical arbitrage strategy that identifies and trades
    temporary price discrepancies between related assets.
    """
    
    def __init__(self, 
                 basket_symbols: List[str],
                 lookback_period: int = 60,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 **kwargs):
        """
        Initialize the statistical arbitrage strategy.
        
        Args:
            basket_symbols: List of symbols to trade as a basket
            lookback_period: Period for calculating statistics
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="Statistical Arbitrage Strategy", **kwargs)
        
        self.basket_symbols = basket_symbols
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # Update parameters
        self.parameters.update({
            'basket_symbols': basket_symbols,
            'lookback_period': lookback_period,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold
        })
    
    def calculate_basket_score(self, symbol: str, timestamp: datetime) -> float:
        """
        Calculate statistical arbitrage score for a symbol relative to the basket.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            Z-score relative to basket
        """
        if symbol not in self.data:
            return 0.0
        
        # Get returns for all symbols in basket
        basket_returns = []
        for basket_symbol in self.basket_symbols:
            if basket_symbol in self.data:
                returns = self.calculate_returns(basket_symbol)
                mask = returns.index <= timestamp
                if mask.sum() >= self.lookback_period:
                    basket_returns.append(returns.loc[mask].iloc[-self.lookback_period:])
        
        if len(basket_returns) < 2:
            return 0.0
        
        # Calculate basket average return
        basket_df = pd.concat(basket_returns, axis=1)
        basket_avg_return = basket_df.mean(axis=1)
        
        # Get target symbol returns
        target_returns = self.calculate_returns(symbol)
        mask = target_returns.index <= timestamp
        if mask.sum() < self.lookback_period:
            return 0.0
        
        target_recent = target_returns.loc[mask].iloc[-self.lookback_period:]
        
        # Align returns
        common_dates = basket_avg_return.index.intersection(target_recent.index)
        if len(common_dates) < self.lookback_period:
            return 0.0
        
        basket_aligned = basket_avg_return.loc[common_dates]
        target_aligned = target_recent.loc[common_dates]
        
        # Calculate relative performance
        relative_performance = target_aligned - basket_aligned
        
        # Calculate Z-score
        mean_rel_perf = relative_performance.mean()
        std_rel_perf = relative_performance.std()
        
        if std_rel_perf == 0:
            return 0.0
        
        current_rel_perf = relative_performance.iloc[-1]
        z_score = (current_rel_perf - mean_rel_perf) / std_rel_perf
        
        return z_score
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate statistical arbitrage signals."""
        signals = []
        
        for symbol in self.basket_symbols:
            if symbol not in self.data:
                continue
            
            try:
                # Calculate statistical arbitrage score
                z_score = self.calculate_basket_score(symbol, timestamp)
                
                if abs(z_score) < self.entry_threshold:
                    continue
                
                # Get current price
                data = self.data[symbol]
                mask = data.index <= timestamp
                current_price = data.loc[mask, 'Close'].iloc[-1]
                
                # Generate signal
                signal_type = 'HOLD'
                strength = 0.0
                
                if z_score > self.entry_threshold:
                    # Symbol outperforming basket - sell (expect reversion)
                    signal_type = 'SELL'
                    strength = min(abs(z_score) / self.entry_threshold, 1.0)
                elif z_score < -self.entry_threshold:
                    # Symbol underperforming basket - buy (expect reversion)
                    signal_type = 'BUY'
                    strength = min(abs(z_score) / self.entry_threshold, 1.0)
                
                if signal_type != 'HOLD' and strength > 0:
                    # Calculate quantity
                    base_quantity = 100
                    quantity = int(base_quantity * strength)
                    
                    signal = StrategySignal(
                        timestamp=timestamp,
                        symbol=symbol,
                        signal_type=signal_type,
                        strength=strength,
                        price=current_price,
                        quantity=quantity,
                        metadata={
                            'basket_z_score': z_score,
                            'basket_symbols': self.basket_symbols,
                            'lookback_period': self.lookback_period,
                            'strategy': 'statistical_arbitrage'
                        }
                    )
                    signals.append(signal)
                    
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the statistical arbitrage strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        if len(self.basket_symbols) < 2:
            raise ValueError("Need at least 2 symbols in basket")
        
        # Validate parameters
        if self.lookback_period < 2:
            raise ValueError("Lookback period must be at least 2")
        
        if self.entry_threshold <= 0:
            raise ValueError("Entry threshold must be positive")
        
        self.is_initialized = True 