"""
Position Sizing System

Advanced position sizing that integrates:
- Risk management constraints
- Market conditions and volatility
- Portfolio optimization principles
- Kelly criterion and risk parity
- Dynamic sizing based on confidence and regime
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""
    symbol: str
    target_size: float
    dollar_amount: float
    risk_contribution: float
    confidence_adjusted: float
    volatility_adjusted: float
    portfolio_weight: float
    max_size_reason: str
    kelly_fraction: float
    risk_parity_weight: float

class PositionSizer:
    """
    Comprehensive position sizing system
    """
    
    def __init__(self,
                 base_risk_per_trade: float = 0.02,
                 max_position_size: float = 0.2,
                 max_sector_concentration: float = 0.3,
                 max_portfolio_leverage: float = 1.0,
                 volatility_lookback: int = 20,
                 confidence_scaling: bool = True,
                 use_kelly_criterion: bool = True,
                 use_risk_parity: bool = True):
        """
        Initialize position sizer
        
        Args:
            base_risk_per_trade: Base risk per trade (2% default)
            max_position_size: Maximum position size (20% default)
            max_sector_concentration: Maximum sector concentration (30% default)
            max_portfolio_leverage: Maximum portfolio leverage (1.0 = no leverage)
            volatility_lookback: Lookback period for volatility calculation
            confidence_scaling: Whether to scale by signal confidence
            use_kelly_criterion: Whether to use Kelly criterion
            use_risk_parity: Whether to use risk parity principles
        """
        self.base_risk_per_trade = base_risk_per_trade
        self.max_position_size = max_position_size
        self.max_sector_concentration = max_sector_concentration
        self.max_portfolio_leverage = max_portfolio_leverage
        self.volatility_lookback = volatility_lookback
        self.confidence_scaling = confidence_scaling
        self.use_kelly_criterion = use_kelly_criterion
        self.use_risk_parity = use_risk_parity
        
        # Position sizing models
        self.kelly_multiplier = 0.25  # Conservative Kelly fraction
        self.risk_parity_target = 0.15  # Target volatility for risk parity
        
    def calculate_volatility_adjusted_size(self,
                                         symbol: str,
                                         price: float,
                                         returns: np.ndarray,
                                         portfolio_value: float) -> float:
        """Calculate position size adjusted for volatility"""
        if len(returns) < self.volatility_lookback:
            return 0.0
        
        # Calculate realized volatility
        recent_returns = returns[-self.volatility_lookback:]
        volatility = np.std(recent_returns) * np.sqrt(252)
        
        # Target volatility contribution
        target_vol_contribution = self.base_risk_per_trade
        
        # Calculate position size to achieve target volatility
        if volatility > 0:
            position_value = (target_vol_contribution * portfolio_value) / volatility
            position_size = position_value / price
        else:
            position_size = 0.0
        
        return position_size
    
    def calculate_kelly_size(self,
                           symbol: str,
                           price: float,
                           win_rate: float,
                           avg_win: float,
                           avg_loss: float,
                           portfolio_value: float) -> Tuple[float, float]:
        """Calculate Kelly optimal position size"""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0, 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply conservative multiplier
        kelly_fraction = kelly_fraction * self.kelly_multiplier
        
        # Ensure non-negative
        kelly_fraction = max(0, kelly_fraction)
        
        # Calculate position size
        position_value = kelly_fraction * portfolio_value
        position_size = position_value / price
        
        return position_size, kelly_fraction
    
    def calculate_risk_parity_size(self,
                                 symbol: str,
                                 price: float,
                                 volatility: float,
                                 portfolio_value: float,
                                 correlation_matrix: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Calculate risk parity position size"""
        if volatility <= 0:
            return 0.0, 0.0
        
        # Target risk contribution
        target_risk = self.risk_parity_target
        
        # Calculate inverse volatility weight
        risk_parity_weight = (1 / volatility) if volatility > 0 else 0
        
        # Adjust for correlations if provided
        if correlation_matrix is not None:
            # This is a simplified approach - in practice, you'd solve
            # the risk parity optimization problem
            risk_parity_weight = risk_parity_weight * 0.8  # Correlation adjustment
        
        # Calculate position size
        position_value = risk_parity_weight * portfolio_value
        position_size = position_value / price
        
        return position_size, risk_parity_weight
    
    def calculate_confidence_adjusted_size(self,
                                         base_size: float,
                                         signal_strength: float,
                                         signal_confidence: float) -> float:
        """Adjust position size based on signal confidence"""
        if not self.confidence_scaling:
            return base_size
        
        # Combine signal strength and confidence
        combined_confidence = (signal_strength * signal_confidence) ** 0.5
        
        # Scale position size
        adjusted_size = base_size * combined_confidence
        
        return adjusted_size
    
    def apply_risk_limits(self,
                         symbol: str,
                         proposed_size: float,
                         price: float,
                         portfolio_value: float,
                         current_positions: Dict[str, float],
                         sector_exposures: Dict[str, float]) -> Tuple[float, str]:
        """Apply risk management limits to position size"""
        max_size = proposed_size
        limit_reason = "no_limit"
        
        # Maximum position size limit
        max_position_value = self.max_position_size * portfolio_value
        max_position_shares = max_position_value / price
        
        if proposed_size > max_position_shares:
            max_size = max_position_shares
            limit_reason = "max_position_size"
        
        # Portfolio leverage limit
        current_exposure = sum(abs(pos) * price for pos in current_positions.values())
        proposed_exposure = current_exposure + abs(max_size) * price
        max_leverage_value = self.max_portfolio_leverage * portfolio_value
        
        if proposed_exposure > max_leverage_value:
            available_capacity = max_leverage_value - current_exposure
            max_size = available_capacity / price
            limit_reason = "leverage_limit"
        
        # Sector concentration limit (if sector info available)
        if sector_exposures:
            # This would require sector mapping - simplified for now
            pass
        
        return max(0, max_size), limit_reason
    
    def calculate_position_size(self,
                              symbol: str,
                              price: float,
                              signal_strength: float,
                              signal_confidence: float,
                              returns: np.ndarray,
                              portfolio_value: float,
                              current_positions: Dict[str, float],
                              trade_history: Optional[List[Dict]] = None,
                              sector_exposures: Optional[Dict[str, float]] = None,
                              correlation_matrix: Optional[np.ndarray] = None) -> PositionSizeResult:
        """
        Calculate comprehensive position size using multiple methods
        
        Args:
            symbol: Asset symbol
            price: Current price
            signal_strength: Signal strength (-1 to 1)
            signal_confidence: Signal confidence (0 to 1)
            returns: Historical returns
            portfolio_value: Current portfolio value
            current_positions: Current positions
            trade_history: Historical trade data
            sector_exposures: Current sector exposures
            correlation_matrix: Asset correlation matrix
            
        Returns:
            PositionSizeResult with detailed sizing information
        """
        # Base volatility-adjusted size
        vol_adjusted_size = self.calculate_volatility_adjusted_size(
            symbol, price, returns, portfolio_value)
        
        # Calculate volatility for other methods
        volatility = np.std(returns[-self.volatility_lookback:]) * np.sqrt(252) if len(returns) >= self.volatility_lookback else 0.15
        
        # Kelly criterion size
        kelly_size, kelly_fraction = 0.0, 0.0
        if self.use_kelly_criterion and trade_history:
            # Extract trade statistics from history
            symbol_trades = [t for t in trade_history if t.get('symbol') == symbol]
            if len(symbol_trades) > 10:  # Need sufficient history
                profits = [t['profit'] for t in symbol_trades if 'profit' in t]
                if profits:
                    wins = [p for p in profits if p > 0]
                    losses = [p for p in profits if p < 0]
                    
                    if wins and losses:
                        win_rate = len(wins) / len(profits)
                        avg_win = np.mean(wins)
                        avg_loss = np.mean(losses)
                        
                        kelly_size, kelly_fraction = self.calculate_kelly_size(
                            symbol, price, win_rate, avg_win, avg_loss, portfolio_value)
        
        # Risk parity size
        risk_parity_size, risk_parity_weight = 0.0, 0.0
        if self.use_risk_parity:
            risk_parity_size, risk_parity_weight = self.calculate_risk_parity_size(
                symbol, price, volatility, portfolio_value, correlation_matrix)
        
        # Combine sizing methods
        size_methods = [vol_adjusted_size]
        if kelly_size > 0:
            size_methods.append(kelly_size)
        if risk_parity_size > 0:
            size_methods.append(risk_parity_size)
        
        # Use conservative approach - take minimum of methods
        if size_methods:
            combined_size = min(size_methods)
        else:
            combined_size = 0.0
        
        # Apply signal strength
        combined_size = combined_size * abs(signal_strength)
        
        # Apply confidence adjustment
        confidence_adjusted_size = self.calculate_confidence_adjusted_size(
            combined_size, abs(signal_strength), signal_confidence)
        
        # Apply risk limits
        final_size, limit_reason = self.apply_risk_limits(
            symbol, confidence_adjusted_size, price, portfolio_value,
            current_positions, sector_exposures or {})
        
        # Calculate risk contribution
        risk_contribution = (final_size * price * volatility) / portfolio_value
        
        # Calculate portfolio weight
        portfolio_weight = (final_size * price) / portfolio_value
        
        # Calculate dollar amount
        dollar_amount = final_size * price
        
        return PositionSizeResult(
            symbol=symbol,
            target_size=final_size,
            dollar_amount=dollar_amount,
            risk_contribution=risk_contribution,
            confidence_adjusted=confidence_adjusted_size,
            volatility_adjusted=vol_adjusted_size,
            portfolio_weight=portfolio_weight,
            max_size_reason=limit_reason,
            kelly_fraction=kelly_fraction,
            risk_parity_weight=risk_parity_weight
        )
    
    def calculate_portfolio_sizes(self,
                                signals: Dict[str, Tuple[float, float]],
                                prices: Dict[str, float],
                                returns_data: Dict[str, np.ndarray],
                                portfolio_value: float,
                                current_positions: Dict[str, float],
                                trade_history: Optional[List[Dict]] = None,
                                correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, PositionSizeResult]:
        """
        Calculate position sizes for entire portfolio
        
        Args:
            signals: Dict of {symbol: (signal_strength, confidence)}
            prices: Dict of {symbol: price}
            returns_data: Dict of {symbol: returns_array}
            portfolio_value: Current portfolio value
            current_positions: Current positions
            trade_history: Historical trade data
            correlation_matrix: Asset correlation matrix
            
        Returns:
            Dict of {symbol: PositionSizeResult}
        """
        results = {}
        
        for symbol, (signal_strength, confidence) in signals.items():
            if symbol not in prices or symbol not in returns_data:
                continue
            
            result = self.calculate_position_size(
                symbol=symbol,
                price=prices[symbol],
                signal_strength=signal_strength,
                signal_confidence=confidence,
                returns=returns_data[symbol],
                portfolio_value=portfolio_value,
                current_positions=current_positions,
                trade_history=trade_history,
                correlation_matrix=correlation_matrix
            )
            
            results[symbol] = result
        
        return results
    
    def optimize_portfolio_allocation(self,
                                    position_results: Dict[str, PositionSizeResult],
                                    portfolio_value: float,
                                    max_total_risk: float = 0.15) -> Dict[str, float]:
        """
        Optimize portfolio allocation to meet risk constraints
        
        Args:
            position_results: Position sizing results
            portfolio_value: Current portfolio value
            max_total_risk: Maximum total portfolio risk
            
        Returns:
            Dict of {symbol: optimized_size}
        """
        # Calculate total risk contribution
        total_risk = sum(result.risk_contribution for result in position_results.values())
        
        # If total risk exceeds limit, scale down proportionally
        if total_risk > max_total_risk:
            scaling_factor = max_total_risk / total_risk
            
            optimized_sizes = {}
            for symbol, result in position_results.items():
                optimized_sizes[symbol] = result.target_size * scaling_factor
            
            return optimized_sizes
        
        # Return original sizes if within risk limits
        return {symbol: result.target_size for symbol, result in position_results.items()}
    
    def get_sizing_summary(self, position_results: Dict[str, PositionSizeResult]) -> Dict[str, Any]:
        """Get summary statistics of position sizing"""
        if not position_results:
            return {}
        
        total_dollar_amount = sum(result.dollar_amount for result in position_results.values())
        total_risk_contribution = sum(result.risk_contribution for result in position_results.values())
        
        return {
            'total_positions': len(position_results),
            'total_dollar_amount': total_dollar_amount,
            'total_risk_contribution': total_risk_contribution,
            'average_position_size': total_dollar_amount / len(position_results),
            'max_position_size': max(result.dollar_amount for result in position_results.values()),
            'min_position_size': min(result.dollar_amount for result in position_results.values()),
            'risk_concentration': max(result.risk_contribution for result in position_results.values()),
            'leverage_ratio': sum(abs(result.portfolio_weight) for result in position_results.values()),
            'long_exposure': sum(result.portfolio_weight for result in position_results.values() if result.portfolio_weight > 0),
            'short_exposure': sum(result.portfolio_weight for result in position_results.values() if result.portfolio_weight < 0)
        } 