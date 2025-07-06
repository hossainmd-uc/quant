"""
Trend Following Trading Strategies
=================================

This module implements trend following strategies that identify and trade
with the direction of price trends.

Strategies included:
- Basic Trend Following Strategy
- Moving Average Crossover Strategy
- Breakout Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from .base_strategy import BaseStrategy, StrategySignal

class TrendFollowingStrategy(BaseStrategy):
    """
    Basic trend following strategy that identifies trend direction
    and trades in the direction of the trend.
    """
    
    def __init__(self, 
                 short_window: int = 20,
                 long_window: int = 50,
                 trend_strength_threshold: float = 0.02,
                 **kwargs):
        """
        Initialize the trend following strategy.
        
        Args:
            short_window: Short-term moving average window
            long_window: Long-term moving average window
            trend_strength_threshold: Minimum trend strength for signal
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="Trend Following Strategy", **kwargs)
        
        self.short_window = short_window
        self.long_window = long_window
        self.trend_strength_threshold = trend_strength_threshold
        
        # Update parameters
        self.parameters.update({
            'short_window': short_window,
            'long_window': long_window,
            'trend_strength_threshold': trend_strength_threshold
        })
    
    def calculate_trend_strength(self, symbol: str, timestamp: datetime) -> Tuple[float, str]:
        """
        Calculate trend strength and direction.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            Tuple of (trend_strength, trend_direction)
        """
        if symbol not in self.data:
            return 0.0, 'NEUTRAL'
        
        # Calculate moving averages
        short_ma = self.calculate_moving_average(symbol, self.short_window)
        long_ma = self.calculate_moving_average(symbol, self.long_window)
        
        # Get values at timestamp
        mask_short = short_ma.index <= timestamp
        mask_long = long_ma.index <= timestamp
        
        if mask_short.sum() == 0 or mask_long.sum() == 0:
            return 0.0, 'NEUTRAL'
        
        current_short_ma = short_ma.loc[mask_short].iloc[-1]
        current_long_ma = long_ma.loc[mask_long].iloc[-1]
        
        if np.isnan(current_short_ma) or np.isnan(current_long_ma):
            return 0.0, 'NEUTRAL'
        
        # Calculate trend strength
        trend_strength = abs(current_short_ma - current_long_ma) / current_long_ma
        
        # Determine trend direction
        if current_short_ma > current_long_ma:
            trend_direction = 'UP'
        elif current_short_ma < current_long_ma:
            trend_direction = 'DOWN'
        else:
            trend_direction = 'NEUTRAL'
        
        return trend_strength, trend_direction
    
    def calculate_momentum_confirmation(self, symbol: str, timestamp: datetime) -> float:
        """
        Calculate momentum confirmation for trend signals.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            Momentum score (-1 to 1)
        """
        if symbol not in self.data:
            return 0.0
        
        data = self.data[symbol]
        mask = data.index <= timestamp
        prices = data.loc[mask, 'Close']
        
        if len(prices) < 10:
            return 0.0
        
        # Calculate short-term momentum
        short_momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
        
        # Calculate medium-term momentum
        medium_momentum = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10]
        
        # Combine momentums
        momentum_score = (short_momentum + medium_momentum) / 2
        
        # Normalize to -1 to 1
        return np.clip(momentum_score * 10, -1, 1)
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate trend following signals."""
        signals = []
        
        for symbol in self.data.keys():
            try:
                # Calculate trend strength and direction
                trend_strength, trend_direction = self.calculate_trend_strength(symbol, timestamp)
                
                if trend_strength < self.trend_strength_threshold or trend_direction == 'NEUTRAL':
                    continue
                
                # Calculate momentum confirmation
                momentum_score = self.calculate_momentum_confirmation(symbol, timestamp)
                
                # Get current price
                data = self.data[symbol]
                mask = data.index <= timestamp
                current_price = data.loc[mask, 'Close'].iloc[-1]
                
                # Generate signal
                signal_type = 'HOLD'
                strength = 0.0
                
                if trend_direction == 'UP' and momentum_score > 0:
                    signal_type = 'BUY'
                    strength = min(trend_strength * abs(momentum_score), 1.0)
                elif trend_direction == 'DOWN' and momentum_score < 0:
                    signal_type = 'SELL'
                    strength = min(trend_strength * abs(momentum_score), 1.0)
                
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
                            'trend_strength': trend_strength,
                            'trend_direction': trend_direction,
                            'momentum_score': momentum_score,
                            'short_window': self.short_window,
                            'long_window': self.long_window,
                            'strategy': 'trend_following'
                        }
                    )
                    signals.append(signal)
                    
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the trend following strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        # Validate parameters
        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window")
        
        if self.trend_strength_threshold <= 0:
            raise ValueError("Trend strength threshold must be positive")
        
        self.is_initialized = True

class MovingAverageCrossover(BaseStrategy):
    """
    Moving average crossover strategy that generates signals when
    short-term MA crosses above or below long-term MA.
    """
    
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 **kwargs):
        """
        Initialize the moving average crossover strategy.
        
        Args:
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            signal_period: Signal smoothing period
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="Moving Average Crossover Strategy", **kwargs)
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Update parameters
        self.parameters.update({
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        })
    
    def detect_crossover(self, symbol: str, timestamp: datetime) -> Tuple[str, float]:
        """
        Detect moving average crossover.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            Tuple of (crossover_type, signal_strength)
        """
        if symbol not in self.data:
            return 'NONE', 0.0
        
        # Calculate moving averages
        fast_ma = self.calculate_moving_average(symbol, self.fast_period)
        slow_ma = self.calculate_moving_average(symbol, self.slow_period)
        
        # Get recent values
        mask_fast = fast_ma.index <= timestamp
        mask_slow = slow_ma.index <= timestamp
        
        if mask_fast.sum() < 2 or mask_slow.sum() < 2:
            return 'NONE', 0.0
        
        fast_current = fast_ma.loc[mask_fast].iloc[-1]
        fast_previous = fast_ma.loc[mask_fast].iloc[-2]
        slow_current = slow_ma.loc[mask_slow].iloc[-1]
        slow_previous = slow_ma.loc[mask_slow].iloc[-2]
        
        if any(np.isnan(x) for x in [fast_current, fast_previous, slow_current, slow_previous]):
            return 'NONE', 0.0
        
        # Check for crossover
        crossover_type = 'NONE'
        signal_strength = 0.0
        
        # Bullish crossover: fast MA crosses above slow MA
        if fast_previous <= slow_previous and fast_current > slow_current:
            crossover_type = 'BULLISH'
            signal_strength = abs(fast_current - slow_current) / slow_current
        
        # Bearish crossover: fast MA crosses below slow MA
        elif fast_previous >= slow_previous and fast_current < slow_current:
            crossover_type = 'BEARISH'
            signal_strength = abs(fast_current - slow_current) / slow_current
        
        return crossover_type, min(signal_strength, 1.0)
    
    def calculate_macd_confirmation(self, symbol: str, timestamp: datetime) -> float:
        """
        Calculate MACD confirmation for crossover signals.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            MACD signal strength
        """
        try:
            macd_data = self.calculate_macd(symbol, self.fast_period, self.slow_period, self.signal_period)
            
            mask = macd_data['histogram'].index <= timestamp
            if mask.sum() == 0:
                return 0.0
            
            current_histogram = macd_data['histogram'].loc[mask].iloc[-1]
            
            if np.isnan(current_histogram):
                return 0.0
            
            # Normalize histogram value
            return np.clip(current_histogram / 0.01, -1, 1)  # Assuming 0.01 is typical scale
            
        except Exception:
            return 0.0
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate moving average crossover signals."""
        signals = []
        
        for symbol in self.data.keys():
            try:
                # Detect crossover
                crossover_type, crossover_strength = self.detect_crossover(symbol, timestamp)
                
                if crossover_type == 'NONE':
                    continue
                
                # Get MACD confirmation
                macd_confirmation = self.calculate_macd_confirmation(symbol, timestamp)
                
                # Get current price
                data = self.data[symbol]
                mask = data.index <= timestamp
                current_price = data.loc[mask, 'Close'].iloc[-1]
                
                # Generate signal
                signal_type = 'HOLD'
                strength = 0.0
                
                if crossover_type == 'BULLISH' and macd_confirmation > 0:
                    signal_type = 'BUY'
                    strength = min(crossover_strength * abs(macd_confirmation), 1.0)
                elif crossover_type == 'BEARISH' and macd_confirmation < 0:
                    signal_type = 'SELL'
                    strength = min(crossover_strength * abs(macd_confirmation), 1.0)
                
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
                            'crossover_type': crossover_type,
                            'crossover_strength': crossover_strength,
                            'macd_confirmation': macd_confirmation,
                            'fast_period': self.fast_period,
                            'slow_period': self.slow_period,
                            'strategy': 'ma_crossover'
                        }
                    )
                    signals.append(signal)
                    
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the moving average crossover strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        # Validate parameters
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        if self.signal_period < 1:
            raise ValueError("Signal period must be at least 1")
        
        self.is_initialized = True

class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy that identifies price breakouts from consolidation patterns.
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 breakout_threshold: float = 0.02,
                 volume_confirmation: bool = True,
                 **kwargs):
        """
        Initialize the breakout strategy.
        
        Args:
            lookback_period: Period for calculating support/resistance levels
            breakout_threshold: Minimum breakout percentage
            volume_confirmation: Whether to require volume confirmation
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="Breakout Strategy", **kwargs)
        
        self.lookback_period = lookback_period
        self.breakout_threshold = breakout_threshold
        self.volume_confirmation = volume_confirmation
        
        # Update parameters
        self.parameters.update({
            'lookback_period': lookback_period,
            'breakout_threshold': breakout_threshold,
            'volume_confirmation': volume_confirmation
        })
    
    def calculate_support_resistance(self, symbol: str, timestamp: datetime) -> Tuple[float, float]:
        """
        Calculate support and resistance levels.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        if symbol not in self.data:
            return 0.0, 0.0
        
        data = self.data[symbol]
        mask = data.index <= timestamp
        recent_data = data.loc[mask].iloc[-self.lookback_period:]
        
        if len(recent_data) < self.lookback_period:
            return 0.0, 0.0
        
        # Calculate support and resistance
        support_level = recent_data['Low'].min()
        resistance_level = recent_data['High'].max()
        
        return support_level, resistance_level
    
    def detect_breakout(self, symbol: str, timestamp: datetime) -> Tuple[str, float]:
        """
        Detect price breakouts.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            Tuple of (breakout_type, breakout_strength)
        """
        if symbol not in self.data:
            return 'NONE', 0.0
        
        # Get support and resistance levels
        support, resistance = self.calculate_support_resistance(symbol, timestamp)
        
        if support == 0.0 or resistance == 0.0:
            return 'NONE', 0.0
        
        # Get current price
        data = self.data[symbol]
        mask = data.index <= timestamp
        current_price = data.loc[mask, 'Close'].iloc[-1]
        
        # Check for breakout
        breakout_type = 'NONE'
        breakout_strength = 0.0
        
        # Upward breakout
        if current_price > resistance:
            breakout_strength = (current_price - resistance) / resistance
            if breakout_strength >= self.breakout_threshold:
                breakout_type = 'UPWARD'
        
        # Downward breakout
        elif current_price < support:
            breakout_strength = (support - current_price) / support
            if breakout_strength >= self.breakout_threshold:
                breakout_type = 'DOWNWARD'
        
        return breakout_type, min(breakout_strength, 1.0)
    
    def check_volume_confirmation(self, symbol: str, timestamp: datetime) -> bool:
        """
        Check if breakout is confirmed by volume.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for calculation
            
        Returns:
            True if volume confirms breakout
        """
        if not self.volume_confirmation:
            return True
        
        if symbol not in self.data:
            return False
        
        data = self.data[symbol]
        mask = data.index <= timestamp
        recent_data = data.loc[mask].iloc[-self.lookback_period:]
        
        if len(recent_data) < 2:
            return False
        
        # Calculate average volume
        avg_volume = recent_data['Volume'].mean()
        current_volume = recent_data['Volume'].iloc[-1]
        
        # Volume should be above average for confirmation
        return current_volume > avg_volume * 1.2
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate breakout signals."""
        signals = []
        
        for symbol in self.data.keys():
            try:
                # Detect breakout
                breakout_type, breakout_strength = self.detect_breakout(symbol, timestamp)
                
                if breakout_type == 'NONE':
                    continue
                
                # Check volume confirmation
                if not self.check_volume_confirmation(symbol, timestamp):
                    continue
                
                # Get current price
                data = self.data[symbol]
                mask = data.index <= timestamp
                current_price = data.loc[mask, 'Close'].iloc[-1]
                
                # Get support and resistance for metadata
                support, resistance = self.calculate_support_resistance(symbol, timestamp)
                
                # Generate signal
                signal_type = 'HOLD'
                strength = breakout_strength
                
                if breakout_type == 'UPWARD':
                    signal_type = 'BUY'
                elif breakout_type == 'DOWNWARD':
                    signal_type = 'SELL'
                
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
                            'breakout_type': breakout_type,
                            'breakout_strength': breakout_strength,
                            'support_level': support,
                            'resistance_level': resistance,
                            'volume_confirmed': self.volume_confirmation,
                            'lookback_period': self.lookback_period,
                            'strategy': 'breakout'
                        }
                    )
                    signals.append(signal)
                    
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the breakout strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        # Validate parameters
        if self.lookback_period < 2:
            raise ValueError("Lookback period must be at least 2")
        
        if self.breakout_threshold <= 0:
            raise ValueError("Breakout threshold must be positive")
        
        self.is_initialized = True 