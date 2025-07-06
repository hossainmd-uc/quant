"""
Technical Indicator Trading Strategies
=====================================

This module implements trading strategies based on traditional technical indicators
like MACD, Stochastic, and other momentum oscillators.

Strategies included:
- Base Technical Strategy
- MACD Strategy
- Stochastic Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .base_strategy import BaseStrategy, StrategySignal

class TechnicalStrategy(BaseStrategy):
    """
    Base class for technical indicator trading strategies.
    """
    
    def __init__(self, **kwargs):
        """Initialize the technical strategy."""
        super().__init__(name="Technical Strategy", **kwargs)
    
    def calculate_stochastic(self, symbol: str, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic oscillator.
        
        Args:
            symbol: Asset symbol
            k_period: Period for %K calculation
            d_period: Period for %D smoothing
            
        Returns:
            Dictionary with %K and %D series
        """
        if symbol not in self.data:
            return {}
        
        data = self.data[symbol]
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (smoothed %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    def calculate_williams_r(self, symbol: str, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R oscillator.
        
        Args:
            symbol: Asset symbol
            period: Lookback period
            
        Returns:
            Williams %R series
        """
        if symbol not in self.data:
            return pd.Series()
        
        data = self.data[symbol]
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate Williams %R
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r
    
    def calculate_commodity_channel_index(self, symbol: str, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            symbol: Asset symbol
            period: Period for calculation
            
        Returns:
            CCI series
        """
        if symbol not in self.data:
            return pd.Series()
        
        data = self.data[symbol]
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate moving average of typical price
        ma_tp = typical_price.rolling(window=period).mean()
        
        # Calculate mean deviation
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        
        # Calculate CCI
        cci = (typical_price - ma_tp) / (0.015 * mad)
        
        return cci

class MACDStrategy(TechnicalStrategy):
    """
    MACD (Moving Average Convergence Divergence) trading strategy.
    """
    
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 **kwargs):
        """
        Initialize the MACD strategy.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="MACD Strategy", **kwargs)
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        # Update parameters
        self.parameters.update({
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        })
    
    def analyze_macd_signals(self, symbol: str, timestamp: datetime) -> Tuple[str, float]:
        """
        Analyze MACD signals for the given symbol and timestamp.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for analysis
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        if symbol not in self.data:
            return 'HOLD', 0.0
        
        # Calculate MACD
        macd_data = self.calculate_macd(symbol, self.fast_period, self.slow_period, self.signal_period)
        
        # Get recent values
        mask = macd_data['macd'].index <= timestamp
        if mask.sum() < 2:
            return 'HOLD', 0.0
        
        macd_line = macd_data['macd'].loc[mask]
        signal_line = macd_data['signal'].loc[mask]
        histogram = macd_data['histogram'].loc[mask]
        
        if len(macd_line) < 2 or len(signal_line) < 2 or len(histogram) < 2:
            return 'HOLD', 0.0
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        previous_macd = macd_line.iloc[-2]
        previous_signal = signal_line.iloc[-2]
        previous_histogram = histogram.iloc[-2]
        
        if any(np.isnan(x) for x in [current_macd, current_signal, current_histogram, 
                                    previous_macd, previous_signal, previous_histogram]):
            return 'HOLD', 0.0
        
        signal_type = 'HOLD'
        signal_strength = 0.0
        
        # MACD crossover signals
        if previous_macd <= previous_signal and current_macd > current_signal:
            # Bullish crossover
            signal_type = 'BUY'
            signal_strength = min(abs(current_macd - current_signal) / abs(current_signal), 1.0)
        elif previous_macd >= previous_signal and current_macd < current_signal:
            # Bearish crossover
            signal_type = 'SELL'
            signal_strength = min(abs(current_macd - current_signal) / abs(current_signal), 1.0)
        
        # Histogram divergence confirmation
        if signal_type != 'HOLD':
            if signal_type == 'BUY' and current_histogram > previous_histogram:
                signal_strength *= 1.2  # Strengthen signal
            elif signal_type == 'SELL' and current_histogram < previous_histogram:
                signal_strength *= 1.2  # Strengthen signal
            else:
                signal_strength *= 0.8  # Weaken signal
        
        return signal_type, min(signal_strength, 1.0)
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate MACD-based trading signals."""
        signals = []
        
        for symbol in self.data.keys():
            try:
                # Analyze MACD signals
                signal_type, signal_strength = self.analyze_macd_signals(symbol, timestamp)
                
                if signal_type == 'HOLD' or signal_strength <= 0:
                    continue
                
                # Get current price
                data = self.data[symbol]
                mask = data.index <= timestamp
                current_price = data.loc[mask, 'Close'].iloc[-1]
                
                # Get MACD values for metadata
                macd_data = self.calculate_macd(symbol, self.fast_period, self.slow_period, self.signal_period)
                macd_mask = macd_data['macd'].index <= timestamp
                
                if macd_mask.sum() > 0:
                    current_macd = macd_data['macd'].loc[macd_mask].iloc[-1]
                    current_signal_line = macd_data['signal'].loc[macd_mask].iloc[-1]
                    current_histogram = macd_data['histogram'].loc[macd_mask].iloc[-1]
                else:
                    current_macd = current_signal_line = current_histogram = 0.0
                
                # Calculate quantity
                base_quantity = 100
                quantity = int(base_quantity * signal_strength)
                
                signal = StrategySignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=signal_strength,
                    price=current_price,
                    quantity=quantity,
                    metadata={
                        'macd_line': current_macd,
                        'signal_line': current_signal_line,
                        'histogram': current_histogram,
                        'fast_period': self.fast_period,
                        'slow_period': self.slow_period,
                        'signal_period': self.signal_period,
                        'strategy': 'macd'
                    }
                )
                signals.append(signal)
                
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the MACD strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        # Validate parameters
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        if self.signal_period < 1:
            raise ValueError("Signal period must be at least 1")
        
        self.is_initialized = True

class StochasticStrategy(TechnicalStrategy):
    """
    Stochastic oscillator trading strategy.
    """
    
    def __init__(self, 
                 k_period: int = 14,
                 d_period: int = 3,
                 oversold_threshold: float = 20,
                 overbought_threshold: float = 80,
                 **kwargs):
        """
        Initialize the Stochastic strategy.
        
        Args:
            k_period: Period for %K calculation
            d_period: Period for %D smoothing
            oversold_threshold: Oversold threshold (buy signal)
            overbought_threshold: Overbought threshold (sell signal)
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="Stochastic Strategy", **kwargs)
        
        self.k_period = k_period
        self.d_period = d_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        
        # Update parameters
        self.parameters.update({
            'k_period': k_period,
            'd_period': d_period,
            'oversold_threshold': oversold_threshold,
            'overbought_threshold': overbought_threshold
        })
    
    def analyze_stochastic_signals(self, symbol: str, timestamp: datetime) -> Tuple[str, float]:
        """
        Analyze Stochastic signals for the given symbol and timestamp.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for analysis
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        if symbol not in self.data:
            return 'HOLD', 0.0
        
        # Calculate Stochastic oscillator
        stoch_data = self.calculate_stochastic(symbol, self.k_period, self.d_period)
        
        if not stoch_data:
            return 'HOLD', 0.0
        
        # Get recent values
        k_mask = stoch_data['k_percent'].index <= timestamp
        d_mask = stoch_data['d_percent'].index <= timestamp
        
        if k_mask.sum() < 2 or d_mask.sum() < 2:
            return 'HOLD', 0.0
        
        k_current = stoch_data['k_percent'].loc[k_mask].iloc[-1]
        d_current = stoch_data['d_percent'].loc[d_mask].iloc[-1]
        
        k_previous = stoch_data['k_percent'].loc[k_mask].iloc[-2]
        d_previous = stoch_data['d_percent'].loc[d_mask].iloc[-2]
        
        if any(np.isnan(x) for x in [k_current, d_current, k_previous, d_previous]):
            return 'HOLD', 0.0
        
        signal_type = 'HOLD'
        signal_strength = 0.0
        
        # Oversold condition with bullish crossover
        if (k_current <= self.oversold_threshold and d_current <= self.oversold_threshold and
            k_previous <= d_previous and k_current > d_current):
            signal_type = 'BUY'
            signal_strength = (self.oversold_threshold - min(k_current, d_current)) / self.oversold_threshold
        
        # Overbought condition with bearish crossover
        elif (k_current >= self.overbought_threshold and d_current >= self.overbought_threshold and
              k_previous >= d_previous and k_current < d_current):
            signal_type = 'SELL'
            signal_strength = (min(k_current, d_current) - self.overbought_threshold) / (100 - self.overbought_threshold)
        
        return signal_type, min(signal_strength, 1.0)
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate Stochastic-based trading signals."""
        signals = []
        
        for symbol in self.data.keys():
            try:
                # Analyze Stochastic signals
                signal_type, signal_strength = self.analyze_stochastic_signals(symbol, timestamp)
                
                if signal_type == 'HOLD' or signal_strength <= 0:
                    continue
                
                # Get current price
                data = self.data[symbol]
                mask = data.index <= timestamp
                current_price = data.loc[mask, 'Close'].iloc[-1]
                
                # Get Stochastic values for metadata
                stoch_data = self.calculate_stochastic(symbol, self.k_period, self.d_period)
                
                if stoch_data:
                    k_mask = stoch_data['k_percent'].index <= timestamp
                    d_mask = stoch_data['d_percent'].index <= timestamp
                    
                    if k_mask.sum() > 0 and d_mask.sum() > 0:
                        current_k = stoch_data['k_percent'].loc[k_mask].iloc[-1]
                        current_d = stoch_data['d_percent'].loc[d_mask].iloc[-1]
                    else:
                        current_k = current_d = 50.0
                else:
                    current_k = current_d = 50.0
                
                # Calculate quantity
                base_quantity = 100
                quantity = int(base_quantity * signal_strength)
                
                signal = StrategySignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=signal_strength,
                    price=current_price,
                    quantity=quantity,
                    metadata={
                        'k_percent': current_k,
                        'd_percent': current_d,
                        'k_period': self.k_period,
                        'd_period': self.d_period,
                        'oversold_threshold': self.oversold_threshold,
                        'overbought_threshold': self.overbought_threshold,
                        'strategy': 'stochastic'
                    }
                )
                signals.append(signal)
                
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the Stochastic strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        # Validate parameters
        if self.k_period < 1:
            raise ValueError("K period must be at least 1")
        
        if self.d_period < 1:
            raise ValueError("D period must be at least 1")
        
        if not (0 <= self.oversold_threshold <= 100):
            raise ValueError("Oversold threshold must be between 0 and 100")
        
        if not (0 <= self.overbought_threshold <= 100):
            raise ValueError("Overbought threshold must be between 0 and 100")
        
        if self.oversold_threshold >= self.overbought_threshold:
            raise ValueError("Oversold threshold must be less than overbought threshold")
        
        self.is_initialized = True 