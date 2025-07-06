"""
Momentum Trading Strategies
==========================

This module implements various momentum-based trading strategies that capitalize on
the tendency of asset prices to continue moving in the same direction.

Strategies included:
- Basic Momentum Strategy
- RSI Momentum Strategy 
- Bollinger Bands Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from .base_strategy import BaseStrategy, StrategySignal

class MomentumStrategy(BaseStrategy):
    """
    Basic momentum strategy that buys assets with strong recent performance
    and sells assets with poor recent performance.
    
    This strategy implements the classic momentum effect where past winners
    continue to outperform and past losers continue to underperform.
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 min_momentum_threshold: float = 0.02,
                 max_momentum_threshold: float = 0.10,
                 **kwargs):
        """
        Initialize the momentum strategy.
        
        Args:
            lookback_period: Number of periods to calculate momentum
            min_momentum_threshold: Minimum momentum for buy signal
            max_momentum_threshold: Maximum momentum (avoid overheated assets)
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="Momentum Strategy", **kwargs)
        
        self.lookback_period = lookback_period
        self.min_momentum_threshold = min_momentum_threshold
        self.max_momentum_threshold = max_momentum_threshold
        
        # Update parameters
        self.parameters.update({
            'lookback_period': lookback_period,
            'min_momentum_threshold': min_momentum_threshold,
            'max_momentum_threshold': max_momentum_threshold
        })
    
    def calculate_momentum(self, symbol: str, timestamp: datetime) -> float:
        """
        Calculate momentum for a symbol at a given timestamp.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            Momentum value (price change over lookback period)
        """
        if symbol not in self.data:
            return 0.0
        
        data = self.data[symbol]
        
        # Get data up to timestamp
        mask = data.index <= timestamp
        prices = data.loc[mask, 'Close']
        
        if len(prices) < self.lookback_period + 1:
            return 0.0
        
        # Calculate momentum as percentage change
        current_price = prices.iloc[-1]
        past_price = prices.iloc[-self.lookback_period-1]
        
        momentum = (current_price - past_price) / past_price
        return momentum
    
    def calculate_momentum_strength(self, momentum: float) -> float:
        """
        Convert momentum to signal strength (0-1).
        
        Args:
            momentum: Raw momentum value
            
        Returns:
            Signal strength between 0 and 1
        """
        # Normalize momentum to signal strength
        abs_momentum = abs(momentum)
        
        if abs_momentum < self.min_momentum_threshold:
            return 0.0
        elif abs_momentum > self.max_momentum_threshold:
            return 1.0
        else:
            # Linear scaling between min and max thresholds
            return (abs_momentum - self.min_momentum_threshold) / (self.max_momentum_threshold - self.min_momentum_threshold)
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate momentum-based trading signals."""
        signals = []
        
        for symbol in self.data.keys():
            try:
                # Calculate momentum
                momentum = self.calculate_momentum(symbol, timestamp)
                
                # Calculate signal strength
                strength = self.calculate_momentum_strength(momentum)
                
                if strength > 0:
                    # Get current price
                    data = self.data[symbol]
                    mask = data.index <= timestamp
                    current_price = data.loc[mask, 'Close'].iloc[-1]
                    
                    # Determine signal type
                    if momentum > self.min_momentum_threshold:
                        signal_type = 'BUY'
                    elif momentum < -self.min_momentum_threshold:
                        signal_type = 'SELL'
                    else:
                        signal_type = 'HOLD'
                    
                    if signal_type != 'HOLD':
                        # Calculate suggested quantity (simplified)
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
                                'momentum': momentum,
                                'lookback_period': self.lookback_period,
                                'strategy': 'momentum'
                            }
                        )
                        signals.append(signal)
                        
            except Exception as e:
                # Skip this symbol if there's an error
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the momentum strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        # Validate parameters
        if self.lookback_period < 1:
            raise ValueError("Lookback period must be at least 1")
        
        if self.min_momentum_threshold >= self.max_momentum_threshold:
            raise ValueError("Min momentum threshold must be less than max threshold")
        
        self.is_initialized = True

class RSIMomentumStrategy(BaseStrategy):
    """
    RSI-based momentum strategy that uses Relative Strength Index
    to identify overbought and oversold conditions.
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 oversold_threshold: float = 30,
                 overbought_threshold: float = 70,
                 **kwargs):
        """
        Initialize the RSI momentum strategy.
        
        Args:
            rsi_period: Period for RSI calculation
            oversold_threshold: RSI level considered oversold (buy signal)
            overbought_threshold: RSI level considered overbought (sell signal)
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="RSI Momentum Strategy", **kwargs)
        
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        
        # Update parameters
        self.parameters.update({
            'rsi_period': rsi_period,
            'oversold_threshold': oversold_threshold,
            'overbought_threshold': overbought_threshold
        })
    
    def calculate_rsi_strength(self, rsi_value: float) -> tuple:
        """
        Calculate signal type and strength from RSI value.
        
        Args:
            rsi_value: Current RSI value
            
        Returns:
            Tuple of (signal_type, strength)
        """
        if rsi_value <= self.oversold_threshold:
            # Oversold - buy signal
            strength = (self.oversold_threshold - rsi_value) / self.oversold_threshold
            return 'BUY', min(strength, 1.0)
        elif rsi_value >= self.overbought_threshold:
            # Overbought - sell signal
            strength = (rsi_value - self.overbought_threshold) / (100 - self.overbought_threshold)
            return 'SELL', min(strength, 1.0)
        else:
            # Neutral zone
            return 'HOLD', 0.0
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate RSI-based momentum signals."""
        signals = []
        
        for symbol in self.data.keys():
            try:
                # Calculate RSI
                rsi_series = self.calculate_rsi(symbol, self.rsi_period)
                
                # Get RSI value at timestamp
                mask = rsi_series.index <= timestamp
                if mask.sum() == 0:
                    continue
                
                current_rsi = rsi_series.loc[mask].iloc[-1]
                
                if np.isnan(current_rsi):
                    continue
                
                # Calculate signal
                signal_type, strength = self.calculate_rsi_strength(current_rsi)
                
                if signal_type != 'HOLD' and strength > 0:
                    # Get current price
                    data = self.data[symbol]
                    price_mask = data.index <= timestamp
                    current_price = data.loc[price_mask, 'Close'].iloc[-1]
                    
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
                            'rsi': current_rsi,
                            'rsi_period': self.rsi_period,
                            'strategy': 'rsi_momentum'
                        }
                    )
                    signals.append(signal)
                    
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the RSI momentum strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        # Validate parameters
        if self.rsi_period < 1:
            raise ValueError("RSI period must be at least 1")
        
        if not (0 <= self.oversold_threshold <= 100):
            raise ValueError("Oversold threshold must be between 0 and 100")
        
        if not (0 <= self.overbought_threshold <= 100):
            raise ValueError("Overbought threshold must be between 0 and 100")
        
        if self.oversold_threshold >= self.overbought_threshold:
            raise ValueError("Oversold threshold must be less than overbought threshold")
        
        self.is_initialized = True

class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands strategy that uses price bands to identify
    potential reversal points and breakout opportunities.
    """
    
    def __init__(self, 
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 breakout_mode: bool = False,
                 **kwargs):
        """
        Initialize the Bollinger Bands strategy.
        
        Args:
            bb_period: Period for Bollinger Bands calculation
            bb_std: Number of standard deviations for bands
            breakout_mode: If True, trade breakouts; if False, trade reversals
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="Bollinger Bands Strategy", **kwargs)
        
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.breakout_mode = breakout_mode
        
        # Update parameters
        self.parameters.update({
            'bb_period': bb_period,
            'bb_std': bb_std,
            'breakout_mode': breakout_mode
        })
    
    def calculate_bb_position(self, price: float, bb_data: Dict) -> float:
        """
        Calculate price position within Bollinger Bands.
        
        Args:
            price: Current price
            bb_data: Bollinger Bands data (upper, lower, middle)
            
        Returns:
            Position value (-1 to 1, where -1 is at lower band, 1 is at upper band)
        """
        upper = bb_data['upper']
        lower = bb_data['lower']
        middle = bb_data['middle']
        
        if upper == lower:  # Avoid division by zero
            return 0.0
        
        # Calculate position within bands
        position = (price - middle) / (upper - middle)
        return np.clip(position, -1.0, 1.0)
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate Bollinger Bands-based signals."""
        signals = []
        
        for symbol in self.data.keys():
            try:
                # Calculate Bollinger Bands
                bb_data = self.calculate_bollinger_bands(symbol, self.bb_period, self.bb_std)
                
                # Get current data
                data = self.data[symbol]
                mask = data.index <= timestamp
                
                if mask.sum() == 0:
                    continue
                
                current_data = data.loc[mask].iloc[-1]
                current_price = current_data['Close']
                
                # Get BB values at timestamp
                bb_mask = bb_data['upper'].index <= timestamp
                if bb_mask.sum() == 0:
                    continue
                
                bb_upper = bb_data['upper'].loc[bb_mask].iloc[-1]
                bb_lower = bb_data['lower'].loc[bb_mask].iloc[-1]
                bb_middle = bb_data['middle'].loc[bb_mask].iloc[-1]
                
                if np.isnan(bb_upper) or np.isnan(bb_lower) or np.isnan(bb_middle):
                    continue
                
                bb_current = {
                    'upper': bb_upper,
                    'lower': bb_lower,
                    'middle': bb_middle
                }
                
                # Calculate position within bands
                position = self.calculate_bb_position(current_price, bb_current)
                
                # Generate signals based on mode
                signal_type = 'HOLD'
                strength = 0.0
                
                if self.breakout_mode:
                    # Breakout strategy: buy above upper band, sell below lower band
                    if current_price > bb_upper:
                        signal_type = 'BUY'
                        strength = min(abs(position), 1.0)
                    elif current_price < bb_lower:
                        signal_type = 'SELL'
                        strength = min(abs(position), 1.0)
                else:
                    # Reversal strategy: buy at lower band, sell at upper band
                    if current_price <= bb_lower:
                        signal_type = 'BUY'
                        strength = min(abs(position), 1.0)
                    elif current_price >= bb_upper:
                        signal_type = 'SELL'
                        strength = min(abs(position), 1.0)
                
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
                            'bb_position': position,
                            'bb_upper': bb_upper,
                            'bb_lower': bb_lower,
                            'bb_middle': bb_middle,
                            'bb_period': self.bb_period,
                            'bb_std': self.bb_std,
                            'breakout_mode': self.breakout_mode,
                            'strategy': 'bollinger_bands'
                        }
                    )
                    signals.append(signal)
                    
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the Bollinger Bands strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        # Validate parameters
        if self.bb_period < 2:
            raise ValueError("Bollinger Bands period must be at least 2")
        
        if self.bb_std <= 0:
            raise ValueError("Bollinger Bands standard deviation must be positive")
        
        self.is_initialized = True 