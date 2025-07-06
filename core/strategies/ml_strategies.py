"""
Machine Learning Trading Strategies
==================================

This module implements ML-based trading strategies that use machine learning
models to predict price movements and generate trading signals.

Strategies included:
- Base ML Strategy
- Linear Regression Strategy
- Random Forest Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from .base_strategy import BaseStrategy, StrategySignal
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class MLStrategy(BaseStrategy):
    """
    Base class for machine learning trading strategies.
    """
    
    def __init__(self, 
                 prediction_horizon: int = 5,
                 feature_window: int = 20,
                 retrain_frequency: int = 100,
                 **kwargs):
        """
        Initialize the ML strategy.
        
        Args:
            prediction_horizon: Number of periods to predict ahead
            feature_window: Number of periods for feature calculation
            retrain_frequency: Frequency of model retraining
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="ML Strategy", **kwargs)
        
        self.prediction_horizon = prediction_horizon
        self.feature_window = feature_window
        self.retrain_frequency = retrain_frequency
        
        # ML components
        self.model = None
        self.scaler = StandardScaler()
        self.last_training_index = 0
        
        # Update parameters
        self.parameters.update({
            'prediction_horizon': prediction_horizon,
            'feature_window': feature_window,
            'retrain_frequency': retrain_frequency
        })
    
    def create_features(self, symbol: str, timestamp: datetime) -> np.ndarray:
        """
        Create feature matrix for ML model.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for feature calculation
            
        Returns:
            Feature matrix
        """
        if symbol not in self.data:
            return np.array([])
        
        data = self.data[symbol]
        mask = data.index <= timestamp
        prices = data.loc[mask, 'Close']
        
        if len(prices) < self.feature_window:
            return np.array([])
        
        recent_prices = prices.iloc[-self.feature_window:]
        
        features = []
        
        # Price-based features
        returns = recent_prices.pct_change().dropna()
        features.extend([
            returns.mean(),  # Average return
            returns.std(),   # Volatility
            returns.skew(),  # Skewness
            returns.kurtosis()  # Kurtosis
        ])
        
        # Technical indicators
        # Moving averages
        ma_short = recent_prices.rolling(5).mean().iloc[-1]
        ma_long = recent_prices.rolling(15).mean().iloc[-1]
        features.extend([
            ma_short / recent_prices.iloc[-1] - 1,  # Short MA relative to price
            ma_long / recent_prices.iloc[-1] - 1,   # Long MA relative to price
            (ma_short - ma_long) / ma_long          # MA convergence
        ])
        
        # RSI
        try:
            rsi = self.calculate_rsi(symbol, 14)
            rsi_mask = rsi.index <= timestamp
            if rsi_mask.sum() > 0:
                features.append(rsi.loc[rsi_mask].iloc[-1] / 100)  # Normalize RSI
            else:
                features.append(0.5)  # Neutral RSI
        except:
            features.append(0.5)
        
        # Momentum indicators
        momentum_5 = (recent_prices.iloc[-1] - recent_prices.iloc[-6]) / recent_prices.iloc[-6]
        momentum_10 = (recent_prices.iloc[-1] - recent_prices.iloc[-11]) / recent_prices.iloc[-11]
        features.extend([momentum_5, momentum_10])
        
        # Volume indicators (if available)
        if 'Volume' in data.columns:
            volume_data = data.loc[mask, 'Volume'].iloc[-self.feature_window:]
            volume_ma = volume_data.rolling(5).mean().iloc[-1]
            current_volume = volume_data.iloc[-1]
            features.append(current_volume / volume_ma - 1)  # Volume relative to average
        else:
            features.append(0.0)
        
        # Handle any NaN values
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0)
        
        return features
    
    def create_target(self, symbol: str, timestamp: datetime) -> float:
        """
        Create target variable for ML model.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for target calculation
            
        Returns:
            Target value (future return)
        """
        if symbol not in self.data:
            return 0.0
        
        data = self.data[symbol]
        mask = data.index <= timestamp
        prices = data.loc[mask, 'Close']
        
        if len(prices) < self.prediction_horizon + 1:
            return 0.0
        
        # Calculate future return
        current_price = prices.iloc[-self.prediction_horizon-1]
        future_price = prices.iloc[-1]
        
        future_return = (future_price - current_price) / current_price
        return future_return
    
    def prepare_training_data(self, symbol: str, timestamp: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for ML model.
        
        Args:
            symbol: Asset symbol
            timestamp: Current timestamp
            
        Returns:
            Tuple of (features, targets)
        """
        if symbol not in self.data:
            return np.array([]), np.array([])
        
        data = self.data[symbol]
        mask = data.index <= timestamp
        
        # We need enough data for features and prediction horizon
        min_data_points = self.feature_window + self.prediction_horizon + 50
        
        if mask.sum() < min_data_points:
            return np.array([]), np.array([])
        
        # Get available timestamps
        available_timestamps = data.loc[mask].index[self.feature_window:-self.prediction_horizon]
        
        if len(available_timestamps) < 20:  # Minimum training samples
            return np.array([])
        
        features_list = []
        targets_list = []
        
        for ts in available_timestamps:
            try:
                # Create features
                features = self.create_features(symbol, ts)
                if len(features) == 0:
                    continue
                
                # Create target
                target = self.create_target(symbol, ts)
                
                features_list.append(features)
                targets_list.append(target)
                
            except Exception:
                continue
        
        if len(features_list) == 0:
            return np.array([]), np.array([])
        
        X = np.array(features_list)
        y = np.array(targets_list)
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the ML model.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        if len(X) == 0 or len(y) == 0:
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model (to be implemented in subclasses)
        if self.model is not None:
            self.model.fit(X_scaled, y)
    
    def predict(self, symbol: str, timestamp: datetime) -> float:
        """
        Make prediction using the trained model.
        
        Args:
            symbol: Asset symbol
            timestamp: Timestamp for prediction
            
        Returns:
            Predicted return
        """
        if self.model is None:
            return 0.0
        
        # Create features
        features = self.create_features(symbol, timestamp)
        if len(features) == 0:
            return 0.0
        
        # Scale features
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.model.predict(features_scaled)[0]
            return prediction
        except Exception:
            return 0.0
    
    def should_retrain(self, current_index: int) -> bool:
        """
        Check if model should be retrained.
        
        Args:
            current_index: Current data index
            
        Returns:
            True if model should be retrained
        """
        return (current_index - self.last_training_index) >= self.retrain_frequency
    
    def generate_signals(self, timestamp: datetime) -> List[StrategySignal]:
        """Generate ML-based trading signals."""
        signals = []
        
        for symbol in self.data.keys():
            try:
                # Get current index
                data = self.data[symbol]
                mask = data.index <= timestamp
                current_index = mask.sum()
                
                # Check if we should retrain
                if self.model is None or self.should_retrain(current_index):
                    X, y = self.prepare_training_data(symbol, timestamp)
                    if len(X) > 0:
                        self.train_model(X, y)
                        self.last_training_index = current_index
                
                # Make prediction
                if self.model is not None:
                    prediction = self.predict(symbol, timestamp)
                    
                    # Convert prediction to signal
                    signal_type = 'HOLD'
                    strength = 0.0
                    
                    # Define thresholds for signal generation
                    buy_threshold = 0.01   # 1% predicted return
                    sell_threshold = -0.01  # -1% predicted return
                    
                    if prediction > buy_threshold:
                        signal_type = 'BUY'
                        strength = min(abs(prediction) * 10, 1.0)  # Scale prediction to strength
                    elif prediction < sell_threshold:
                        signal_type = 'SELL'
                        strength = min(abs(prediction) * 10, 1.0)
                    
                    if signal_type != 'HOLD' and strength > 0:
                        # Get current price
                        current_price = data.loc[mask, 'Close'].iloc[-1]
                        
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
                                'prediction': prediction,
                                'prediction_horizon': self.prediction_horizon,
                                'feature_window': self.feature_window,
                                'model_type': self.__class__.__name__,
                                'strategy': 'ml_strategy'
                            }
                        )
                        signals.append(signal)
                        
            except Exception as e:
                continue
        
        return signals
    
    def initialize(self) -> None:
        """Initialize the ML strategy."""
        if not self.data:
            raise ValueError("No data available for strategy initialization")
        
        # Validate parameters
        if self.prediction_horizon < 1:
            raise ValueError("Prediction horizon must be at least 1")
        
        if self.feature_window < 5:
            raise ValueError("Feature window must be at least 5")
        
        self.is_initialized = True

class LinearRegressionStrategy(MLStrategy):
    """
    Linear regression based trading strategy.
    """
    
    def __init__(self, **kwargs):
        """Initialize the linear regression strategy."""
        super().__init__(name="Linear Regression Strategy", **kwargs)
        
        # Initialize linear regression model
        self.model = LinearRegression()
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the linear regression model."""
        if len(X) == 0 or len(y) == 0:
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train linear regression
        self.model.fit(X_scaled, y)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from linear regression coefficients.
        
        Returns:
            Dictionary of feature names and their importance
        """
        if self.model is None or not hasattr(self.model, 'coef_'):
            return {}
        
        feature_names = [
            'avg_return', 'volatility', 'skewness', 'kurtosis',
            'ma_short_rel', 'ma_long_rel', 'ma_convergence',
            'rsi', 'momentum_5', 'momentum_10', 'volume_rel'
        ]
        
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(self.model.coef_):
                importance[name] = abs(self.model.coef_[i])
        
        return importance

class RandomForestStrategy(MLStrategy):
    """
    Random Forest based trading strategy.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 10,
                 **kwargs):
        """
        Initialize the random forest strategy.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            **kwargs: Additional parameters for base strategy
        """
        super().__init__(name="Random Forest Strategy", **kwargs)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        # Initialize random forest model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
        # Update parameters
        self.parameters.update({
            'n_estimators': n_estimators,
            'max_depth': max_depth
        })
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the random forest model."""
        if len(X) == 0 or len(y) == 0:
            return
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train random forest
        self.model.fit(X_scaled, y)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from random forest.
        
        Returns:
            Dictionary of feature names and their importance
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        feature_names = [
            'avg_return', 'volatility', 'skewness', 'kurtosis',
            'ma_short_rel', 'ma_long_rel', 'ma_convergence',
            'rsi', 'momentum_5', 'momentum_10', 'volume_rel'
        ]
        
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(self.model.feature_importances_):
                importance[name] = self.model.feature_importances_[i]
        
        return importance 