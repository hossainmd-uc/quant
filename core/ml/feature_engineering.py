"""
Financial Feature Engineering
============================

Advanced feature engineering for financial data implementing various
techniques from machine learning for finance.

Key Features:
- Technical indicators and overlays
- Market microstructure features
- Macroeconomic indicators
- Statistical features
- Time-based features
- Volatility measures
- Risk-adjusted returns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
from dataclasses import dataclass

class FinancialFeatureEngineer:
    """
    Comprehensive feature engineering for financial data.
    """
    
    def __init__(self):
        self.features = {}
        self.feature_names = []
    
    def fit(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None):
        """
        Fit the feature engineer to the data.
        
        Args:
            price_data: DataFrame with OHLCV data
            volume_data: Optional volume data
        """
        self.price_data = price_data
        self.volume_data = volume_data
        
        # Generate all features
        self._generate_all_features()
        
    def _generate_all_features(self):
        """Generate all feature categories."""
        
        # Technical indicators
        tech_engineer = TechnicalIndicatorFeatures()
        tech_features = tech_engineer.generate_features(self.price_data, self.volume_data)
        self.features.update(tech_features)
        
        # Market microstructure features
        micro_engineer = MarketMicrostructureFeatures()
        micro_features = micro_engineer.generate_features(self.price_data, self.volume_data)
        self.features.update(micro_features)
        
        # Statistical features
        stat_features = self._generate_statistical_features()
        self.features.update(stat_features)
        
        # Time-based features
        time_features = self._generate_time_features()
        self.features.update(time_features)
        
        self.feature_names = list(self.features.keys())
    
    def _generate_statistical_features(self) -> Dict[str, pd.Series]:
        """Generate statistical features."""
        features = {}
        
        if 'Close' in self.price_data.columns:
            prices = self.price_data['Close']
            returns = prices.pct_change()
            
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                features[f'return_mean_{window}'] = returns.rolling(window).mean()
                features[f'return_std_{window}'] = returns.rolling(window).std()
                features[f'return_skew_{window}'] = returns.rolling(window).skew()
                features[f'return_kurt_{window}'] = returns.rolling(window).kurt()
                
                # Price-based features
                features[f'price_mean_{window}'] = prices.rolling(window).mean()
                features[f'price_std_{window}'] = prices.rolling(window).std()
                features[f'price_z_score_{window}'] = (prices - prices.rolling(window).mean()) / prices.rolling(window).std()
        
        return features
    
    def _generate_time_features(self) -> Dict[str, pd.Series]:
        """Generate time-based features."""
        features = {}
        
        if isinstance(self.price_data.index, pd.DatetimeIndex):
            # Calendar features
            features['day_of_week'] = self.price_data.index.dayofweek
            features['month'] = self.price_data.index.month
            features['quarter'] = self.price_data.index.quarter
            features['is_month_end'] = self.price_data.index.is_month_end.astype(int)
            features['is_quarter_end'] = self.price_data.index.is_quarter_end.astype(int)
            
            # Convert to Series
            for name, values in features.items():
                features[name] = pd.Series(values, index=self.price_data.index)
        
        return features
    
    def get_features_dataframe(self) -> pd.DataFrame:
        """Get all features as a DataFrame."""
        if not self.features:
            raise ValueError("No features generated. Call fit() first.")
        
        return pd.DataFrame(self.features)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names

class TechnicalIndicatorFeatures:
    """
    Technical indicators feature generator.
    """
    
    def generate_features(self, price_data: pd.DataFrame, 
                         volume_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """Generate technical indicator features."""
        features = {}
        
        if 'Close' in price_data.columns:
            close = price_data['Close']
            
            # Moving averages
            for window in [5, 10, 20, 50, 100, 200]:
                features[f'sma_{window}'] = close.rolling(window).mean()
                features[f'ema_{window}'] = close.ewm(span=window).mean()
                features[f'price_sma_ratio_{window}'] = close / close.rolling(window).mean()
        
        if 'High' in price_data.columns and 'Low' in price_data.columns and 'Close' in price_data.columns:
            high = price_data['High']
            low = price_data['Low']
            close = price_data['Close']
            
            # RSI
            features['rsi_14'] = self._calculate_rsi(close, 14)
            features['rsi_30'] = self._calculate_rsi(close, 30)
            
            # Bollinger Bands
            bb_features = self._calculate_bollinger_bands(close, 20, 2)
            features.update(bb_features)
            
            # MACD
            macd_features = self._calculate_macd(close, 12, 26, 9)
            features.update(macd_features)
            
            # Stochastic
            stoch_features = self._calculate_stochastic(high, low, close, 14)
            features.update(stoch_features)
            
            # Average True Range
            features['atr_14'] = self._calculate_atr(high, low, close, 14)
            
            # Williams %R
            features['williams_r_14'] = self._calculate_williams_r(high, low, close, 14)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int, 
                                  num_std: float) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_width': (upper_band - lower_band) / sma,
            'bb_position': (prices - lower_band) / (upper_band - lower_band)
        }
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, 
                       signal: int) -> Dict[str, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, 
                             close: pd.Series, window: int) -> Dict[str, pd.Series]:
        """Calculate Stochastic oscillator."""
        highest_high = high.rolling(window).max()
        lowest_low = low.rolling(window).min()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(3).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, window: int) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window).mean()
        
        return atr
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, 
                             close: pd.Series, window: int) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window).max()
        lowest_low = low.rolling(window).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r

class MarketMicrostructureFeatures:
    """
    Market microstructure features generator.
    """
    
    def generate_features(self, price_data: pd.DataFrame, 
                         volume_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """Generate market microstructure features."""
        features = {}
        
        if 'Volume' in price_data.columns:
            volume = price_data['Volume']
            
            # Volume features
            for window in [5, 10, 20]:
                features[f'volume_sma_{window}'] = volume.rolling(window).mean()
                features[f'volume_ratio_{window}'] = volume / volume.rolling(window).mean()
                features[f'volume_std_{window}'] = volume.rolling(window).std()
        
        if 'Close' in price_data.columns and 'Volume' in price_data.columns:
            close = price_data['Close']
            volume = price_data['Volume']
            
            # Price-volume features
            features['vwap'] = self._calculate_vwap(price_data)
            features['price_volume_trend'] = self._calculate_pvt(close, volume)
            
            # On-Balance Volume
            features['obv'] = self._calculate_obv(close, volume)
            
            # Volume-weighted features
            returns = close.pct_change()
            features['volume_weighted_returns'] = returns * volume
        
        if 'High' in price_data.columns and 'Low' in price_data.columns:
            high = price_data['High']
            low = price_data['Low']
            
            # Spread measures
            features['hl_spread'] = (high - low) / low
            features['hl_spread_ma'] = features['hl_spread'].rolling(20).mean()
        
        return features
    
    def _calculate_vwap(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        if 'High' in price_data.columns and 'Low' in price_data.columns and 'Close' in price_data.columns:
            typical_price = (price_data['High'] + price_data['Low'] + price_data['Close']) / 3
            volume = price_data['Volume']
            
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            return vwap
        else:
            return pd.Series(index=price_data.index)
    
    def _calculate_pvt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Price Volume Trend."""
        returns = close.pct_change()
        pvt = (returns * volume).cumsum()
        return pvt
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = close.diff()
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0))
        obv = pd.Series(obv, index=close.index).cumsum()
        return obv

class MacroeconomicFeatures:
    """
    Macroeconomic features generator.
    """
    
    def generate_features(self, price_data: pd.DataFrame, 
                         macro_data: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """Generate macroeconomic features."""
        features = {}
        
        if macro_data is not None:
            # Reindex macro data to match price data
            macro_reindexed = macro_data.reindex(price_data.index, method='ffill')
            
            # Add macro features
            for col in macro_reindexed.columns:
                features[f'macro_{col}'] = macro_reindexed[col]
                
                # Macro momentum
                features[f'macro_{col}_momentum'] = macro_reindexed[col].pct_change()
                
                # Macro moving average
                features[f'macro_{col}_ma'] = macro_reindexed[col].rolling(20).mean()
        
        # Generate synthetic macro proxies if no macro data provided
        if 'Close' in price_data.columns and macro_data is None:
            close = price_data['Close']
            
            # Market volatility as macro proxy
            features['market_volatility'] = close.pct_change().rolling(20).std()
            
            # Trend strength
            features['trend_strength'] = abs(close.rolling(20).mean().pct_change())
        
        return features 