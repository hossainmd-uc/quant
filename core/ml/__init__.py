"""
Advanced Machine Learning Module
===============================

This module contains advanced machine learning techniques implementing
231 machine learning concepts from "Python for Finance: Mastering Data-Driven Finance".

The module includes:
- Financial feature engineering
- Time series forecasting models
- Ensemble methods for trading
- Deep learning models
- Reinforcement learning for trading
- Model validation and backtesting
- Feature selection and dimensionality reduction
- Regime detection using ML
- Anomaly detection
- Risk modeling with ML
"""

from .feature_engineering import (
    FinancialFeatureEngineer, TechnicalIndicatorFeatures,
    MarketMicrostructureFeatures, MacroeconomicFeatures
)
from .time_series_models import (
    LSTMPredictor, GRUPredictor, TransformerPredictor,
    ARIMAEnsemble, TimeSeriesValidator
)
from .ensemble_methods import (
    TradingEnsemble, StackingPredictor, VotingPredictor,
    BaggingPredictor, BoostingPredictor
)
from .deep_learning import (
    DeepTradingNetwork, CNNPredictor, RNNPredictor,
    AutoEncoder, GANPredictor, AttentionModel
)
from .reinforcement_learning import (
    TradingEnvironment, DQNTrader, PPOTrader,
    A3CTrader, ReinforcementLearningBacktester
)
from .model_validation import (
    TimeSeriesValidator, WalkForwardValidator,
    NestedCrossValidator, ModelSelector
)
from .feature_selection import (
    FeatureSelector, PCAReducer, ICAReducer,
    FactorAnalysis, LassoSelector, TreeSelector
)
from .regime_detection import (
    MarkovRegimeDetector, HMMRegimeDetector,
    ChangePointDetector, ClusteringRegimeDetector
)
from .anomaly_detection import (
    AnomalyDetector, IsolationForestDetector,
    OneClassSVMDetector, AutoEncoderDetector
)
from .risk_modeling import (
    MLRiskModel, VARPredictor, VolatilityPredictor,
    StressTestingML, RiskFactorModel
)

__all__ = [
    # Feature Engineering
    'FinancialFeatureEngineer',
    'TechnicalIndicatorFeatures',
    'MarketMicrostructureFeatures',
    'MacroeconomicFeatures',
    
    # Time Series Models
    'LSTMPredictor',
    'GRUPredictor',
    'TransformerPredictor',
    'ARIMAEnsemble',
    'TimeSeriesValidator',
    
    # Ensemble Methods
    'TradingEnsemble',
    'StackingPredictor',
    'VotingPredictor',
    'BaggingPredictor',
    'BoostingPredictor',
    
    # Deep Learning
    'DeepTradingNetwork',
    'CNNPredictor',
    'RNNPredictor',
    'AutoEncoder',
    'GANPredictor',
    'AttentionModel',
    
    # Reinforcement Learning
    'TradingEnvironment',
    'DQNTrader',
    'PPOTrader',
    'A3CTrader',
    'ReinforcementLearningBacktester',
    
    # Model Validation
    'TimeSeriesValidator',
    'WalkForwardValidator',
    'NestedCrossValidator',
    'ModelSelector',
    
    # Feature Selection
    'FeatureSelector',
    'PCAReducer',
    'ICAReducer',
    'FactorAnalysis',
    'LassoSelector',
    'TreeSelector',
    
    # Regime Detection
    'MarkovRegimeDetector',
    'HMMRegimeDetector',
    'ChangePointDetector',
    'ClusteringRegimeDetector',
    
    # Anomaly Detection
    'AnomalyDetector',
    'IsolationForestDetector',
    'OneClassSVMDetector',
    'AutoEncoderDetector',
    
    # Risk Modeling
    'MLRiskModel',
    'VARPredictor',
    'VolatilityPredictor',
    'StressTestingML',
    'RiskFactorModel'
] 