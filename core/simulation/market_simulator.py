"""
Market Simulator - Realistic Market Conditions

Models realistic market conditions including:
- Volatility clustering and regime changes
- Liquidity constraints and market impact
- Bid-ask spreads and slippage
- Market hours and trading halts
- Macro-economic events and news impact
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    CRISIS = "crisis"

@dataclass
class MarketConditions:
    """Current market conditions"""
    timestamp: datetime
    regime: MarketRegime
    volatility: float
    liquidity_factor: float
    bid_ask_spread: float
    market_impact: float
    is_trading_hours: bool
    news_sentiment: float
    macro_stress: float

class MarketSimulator:
    """
    Simulates realistic market conditions and their impact on trading
    """
    
    def __init__(self, 
                 base_volatility: float = 0.15,
                 volatility_clustering: float = 0.1,
                 liquidity_impact: float = 0.001,
                 min_spread: float = 0.0001,
                 max_spread: float = 0.01,
                 news_impact: float = 0.05):
        """
        Initialize market simulator
        
        Args:
            base_volatility: Base annualized volatility
            volatility_clustering: Volatility clustering parameter
            liquidity_impact: Market impact per dollar traded
            min_spread: Minimum bid-ask spread
            max_spread: Maximum bid-ask spread
            news_impact: News sentiment impact on prices
        """
        self.base_volatility = base_volatility
        self.volatility_clustering = volatility_clustering
        self.liquidity_impact = liquidity_impact
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.news_impact = news_impact
        
        # Market state
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_duration = 0
        self.volatility_state = base_volatility
        self.liquidity_state = 1.0
        self.news_sentiment = 0.0
        
        # Regime transition probabilities
        self.regime_transitions = {
            MarketRegime.BULL: {
                MarketRegime.BULL: 0.85,
                MarketRegime.SIDEWAYS: 0.10,
                MarketRegime.BEAR: 0.03,
                MarketRegime.HIGH_VOLATILITY: 0.02
            },
            MarketRegime.BEAR: {
                MarketRegime.BEAR: 0.80,
                MarketRegime.SIDEWAYS: 0.12,
                MarketRegime.BULL: 0.05,
                MarketRegime.CRISIS: 0.03
            },
            MarketRegime.SIDEWAYS: {
                MarketRegime.SIDEWAYS: 0.70,
                MarketRegime.BULL: 0.15,
                MarketRegime.BEAR: 0.10,
                MarketRegime.HIGH_VOLATILITY: 0.05
            },
            MarketRegime.HIGH_VOLATILITY: {
                MarketRegime.HIGH_VOLATILITY: 0.60,
                MarketRegime.SIDEWAYS: 0.25,
                MarketRegime.BULL: 0.10,
                MarketRegime.BEAR: 0.05
            },
            MarketRegime.LOW_VOLATILITY: {
                MarketRegime.LOW_VOLATILITY: 0.75,
                MarketRegime.SIDEWAYS: 0.20,
                MarketRegime.BULL: 0.05
            },
            MarketRegime.CRISIS: {
                MarketRegime.CRISIS: 0.70,
                MarketRegime.BEAR: 0.25,
                MarketRegime.HIGH_VOLATILITY: 0.05
            }
        }
    
    def update_regime(self, timestamp: datetime) -> MarketRegime:
        """Update market regime based on transition probabilities"""
        self.regime_duration += 1
        
        # Get transition probabilities for current regime
        transitions = self.regime_transitions.get(self.current_regime, {})
        
        # Sample next regime
        regimes = list(transitions.keys())
        probabilities = list(transitions.values())
        
        if regimes and sum(probabilities) > 0:
            next_regime = np.random.choice(regimes, p=probabilities)
            
            # Reset duration if regime changes
            if next_regime != self.current_regime:
                self.regime_duration = 0
                self.current_regime = next_regime
        
        return self.current_regime
    
    def update_volatility(self, returns: np.ndarray, lookback: int = 20) -> float:
        """Update volatility with clustering effects"""
        if len(returns) < lookback:
            return self.base_volatility
        
        # Calculate realized volatility
        recent_returns = returns[-lookback:]
        realized_vol = np.std(recent_returns) * np.sqrt(252)
        
        # Apply volatility clustering
        vol_shock = np.random.normal(0, self.volatility_clustering)
        self.volatility_state = (0.9 * self.volatility_state + 
                                0.1 * realized_vol + vol_shock)
        
        # Ensure volatility stays in reasonable bounds
        self.volatility_state = np.clip(self.volatility_state, 0.05, 1.0)
        
        return self.volatility_state
    
    def update_liquidity(self, volume: float, avg_volume: float) -> float:
        """Update liquidity conditions based on volume"""
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # Higher volume generally means better liquidity
        liquidity_factor = 1.0 + np.log(volume_ratio) * 0.1
        
        # Add random liquidity shocks
        liquidity_shock = np.random.normal(0, 0.05)
        self.liquidity_state = np.clip(liquidity_factor + liquidity_shock, 0.1, 3.0)
        
        return self.liquidity_state
    
    def calculate_bid_ask_spread(self, price: float, volatility: float, 
                               liquidity: float) -> float:
        """Calculate realistic bid-ask spread"""
        # Base spread increases with volatility and decreases with liquidity
        base_spread = self.min_spread * (1 + volatility * 10) / liquidity
        
        # Add regime-specific adjustments
        regime_multiplier = {
            MarketRegime.BULL: 1.0,
            MarketRegime.BEAR: 1.5,
            MarketRegime.SIDEWAYS: 1.2,
            MarketRegime.HIGH_VOLATILITY: 2.0,
            MarketRegime.LOW_VOLATILITY: 0.8,
            MarketRegime.CRISIS: 3.0
        }.get(self.current_regime, 1.0)
        
        spread = base_spread * regime_multiplier
        return np.clip(spread, self.min_spread, self.max_spread)
    
    def calculate_market_impact(self, trade_size: float, avg_volume: float,
                              liquidity: float) -> float:
        """Calculate market impact of a trade"""
        # Market impact is square root of trade size relative to volume
        relative_size = trade_size / avg_volume if avg_volume > 0 else 0
        base_impact = np.sqrt(relative_size) * self.liquidity_impact
        
        # Adjust for liquidity conditions
        impact = base_impact / liquidity
        
        # Cap maximum impact
        return min(impact, 0.05)  # Max 5% impact
    
    def generate_news_sentiment(self, timestamp: datetime) -> float:
        """Generate news sentiment impact"""
        # Random news events with persistence
        if np.random.random() < 0.1:  # 10% chance of news event
            self.news_sentiment = np.random.normal(0, 0.02)
        
        # Sentiment decays over time
        self.news_sentiment *= 0.95
        
        return self.news_sentiment
    
    def is_trading_hours(self, timestamp: datetime) -> bool:
        """Check if market is open (simplified to weekdays 9:30-16:00 ET)"""
        if timestamp.weekday() >= 5:  # Weekend
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = timestamp.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= timestamp <= market_close
    
    def simulate_market_conditions(self, 
                                 timestamp: datetime,
                                 price: float,
                                 volume: float,
                                 avg_volume: float,
                                 returns: np.ndarray) -> MarketConditions:
        """
        Simulate comprehensive market conditions
        
        Args:
            timestamp: Current timestamp
            price: Current price
            volume: Current volume
            avg_volume: Average volume
            returns: Recent returns for volatility calculation
            
        Returns:
            MarketConditions object with all market state
        """
        # Update market regime
        regime = self.update_regime(timestamp)
        
        # Update volatility with clustering
        volatility = self.update_volatility(returns)
        
        # Update liquidity conditions
        liquidity = self.update_liquidity(volume, avg_volume)
        
        # Calculate bid-ask spread
        bid_ask_spread = self.calculate_bid_ask_spread(price, volatility, liquidity)
        
        # Calculate market impact
        market_impact = self.calculate_market_impact(volume, avg_volume, liquidity)
        
        # Generate news sentiment
        news_sentiment = self.generate_news_sentiment(timestamp)
        
        # Check trading hours
        is_trading = self.is_trading_hours(timestamp)
        
        # Calculate macro stress indicator
        macro_stress = self.calculate_macro_stress(volatility, regime)
        
        return MarketConditions(
            timestamp=timestamp,
            regime=regime,
            volatility=volatility,
            liquidity_factor=liquidity,
            bid_ask_spread=bid_ask_spread,
            market_impact=market_impact,
            is_trading_hours=is_trading,
            news_sentiment=news_sentiment,
            macro_stress=macro_stress
        )
    
    def calculate_macro_stress(self, volatility: float, regime: MarketRegime) -> float:
        """Calculate macro-economic stress indicator"""
        base_stress = volatility / self.base_volatility
        
        # Regime-specific stress multipliers
        regime_stress = {
            MarketRegime.BULL: 0.5,
            MarketRegime.BEAR: 1.5,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.HIGH_VOLATILITY: 2.0,
            MarketRegime.LOW_VOLATILITY: 0.3,
            MarketRegime.CRISIS: 3.0
        }.get(regime, 1.0)
        
        return base_stress * regime_stress
    
    def apply_market_impact_to_price(self, price: float, trade_size: float,
                                   side: str, market_conditions: MarketConditions) -> Tuple[float, float]:
        """
        Apply market impact to execution price
        
        Args:
            price: Target price
            trade_size: Size of trade
            side: 'buy' or 'sell'
            market_conditions: Current market conditions
            
        Returns:
            Tuple of (execution_price, total_slippage)
        """
        # Base slippage from bid-ask spread
        spread_slippage = market_conditions.bid_ask_spread / 2
        
        # Market impact slippage
        impact_slippage = market_conditions.market_impact
        
        # News sentiment impact
        news_slippage = abs(market_conditions.news_sentiment) * 0.1
        
        # Total slippage
        total_slippage = spread_slippage + impact_slippage + news_slippage
        
        # Apply slippage based on trade direction
        if side == 'buy':
            execution_price = price * (1 + total_slippage)
        else:
            execution_price = price * (1 - total_slippage)
        
        return execution_price, total_slippage
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime transitions"""
        return {
            'current_regime': self.current_regime.value,
            'regime_duration': self.regime_duration,
            'volatility_state': self.volatility_state,
            'liquidity_state': self.liquidity_state,
            'news_sentiment': self.news_sentiment,
            'regime_transitions': {k.value: v for k, v in self.regime_transitions.items()}
        } 