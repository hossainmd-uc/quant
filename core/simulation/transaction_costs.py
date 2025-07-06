"""
Transaction Cost Model

Comprehensive transaction cost modeling including:
- Commission costs
- Bid-ask spreads
- Market impact
- Slippage
- Financing costs
- Tax implications
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class AssetClass(Enum):
    """Asset class types for cost modeling"""
    EQUITY = "equity"
    BOND = "bond"
    FOREX = "forex"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    OPTION = "option"
    FUTURE = "future"

@dataclass
class TransactionCost:
    """Comprehensive transaction cost breakdown"""
    symbol: str
    trade_value: float
    commission: float
    spread_cost: float
    market_impact_cost: float
    slippage_cost: float
    financing_cost: float
    tax_cost: float
    total_cost: float
    cost_per_share: float
    cost_basis_points: float

class TransactionCostModel:
    """
    Realistic transaction cost modeling for various asset classes
    """
    
    def __init__(self):
        """Initialize transaction cost model with realistic parameters"""
        
        # Commission structures by asset class
        self.commission_rates = {
            AssetClass.EQUITY: {
                'fixed': 0.0,  # Most modern brokers have zero commissions
                'per_share': 0.0,
                'percentage': 0.0
            },
            AssetClass.BOND: {
                'fixed': 5.0,
                'per_share': 0.0,
                'percentage': 0.0005
            },
            AssetClass.FOREX: {
                'fixed': 0.0,
                'per_share': 0.0,
                'percentage': 0.0001  # Spread built into price
            },
            AssetClass.COMMODITY: {
                'fixed': 2.0,
                'per_share': 0.0,
                'percentage': 0.0002
            },
            AssetClass.CRYPTO: {
                'fixed': 0.0,
                'per_share': 0.0,
                'percentage': 0.001  # Maker fee
            },
            AssetClass.OPTION: {
                'fixed': 0.0,
                'per_share': 0.65,  # Per contract
                'percentage': 0.0
            },
            AssetClass.FUTURE: {
                'fixed': 2.5,
                'per_share': 0.0,
                'percentage': 0.0
            }
        }
        
        # Typical bid-ask spreads by asset class (in basis points)
        self.typical_spreads = {
            AssetClass.EQUITY: 5,    # Large cap: 2-5 bps
            AssetClass.BOND: 25,     # Corporate bonds: 20-50 bps
            AssetClass.FOREX: 1,     # Major pairs: 0.5-2 bps
            AssetClass.COMMODITY: 10, # Commodities: 5-20 bps
            AssetClass.CRYPTO: 50,   # Crypto: 20-100 bps
            AssetClass.OPTION: 100,  # Options: 50-200 bps
            AssetClass.FUTURE: 5     # Futures: 2-10 bps
        }
        
        # Market impact parameters
        self.market_impact_params = {
            AssetClass.EQUITY: {
                'temporary': 0.5,  # Temporary impact coefficient
                'permanent': 0.3,  # Permanent impact coefficient
                'power': 0.5      # Square root law
            },
            AssetClass.BOND: {
                'temporary': 0.8,
                'permanent': 0.6,
                'power': 0.6
            },
            AssetClass.FOREX: {
                'temporary': 0.1,
                'permanent': 0.05,
                'power': 0.3
            },
            AssetClass.COMMODITY: {
                'temporary': 0.6,
                'permanent': 0.4,
                'power': 0.5
            },
            AssetClass.CRYPTO: {
                'temporary': 1.5,
                'permanent': 1.0,
                'power': 0.7
            },
            AssetClass.OPTION: {
                'temporary': 2.0,
                'permanent': 1.5,
                'power': 0.8
            },
            AssetClass.FUTURE: {
                'temporary': 0.3,
                'permanent': 0.2,
                'power': 0.4
            }
        }
        
        # Financing rates (for overnight positions)
        self.financing_rates = {
            AssetClass.EQUITY: 0.05,     # 5% annual
            AssetClass.BOND: 0.03,       # 3% annual
            AssetClass.FOREX: 0.04,      # 4% annual
            AssetClass.COMMODITY: 0.06,  # 6% annual
            AssetClass.CRYPTO: 0.0,      # No financing for spot
            AssetClass.OPTION: 0.05,     # 5% annual
            AssetClass.FUTURE: 0.0       # No financing for futures
        }
        
        # Tax rates (simplified)
        self.tax_rates = {
            'short_term_capital_gains': 0.25,  # < 1 year
            'long_term_capital_gains': 0.15,   # > 1 year
            'dividend_tax': 0.15
        }
    
    def calculate_commission(self, 
                           asset_class: AssetClass,
                           trade_value: float,
                           quantity: float) -> float:
        """Calculate commission costs"""
        rates = self.commission_rates[asset_class]
        
        commission = (rates['fixed'] + 
                     rates['per_share'] * abs(quantity) + 
                     rates['percentage'] * trade_value)
        
        return commission
    
    def calculate_spread_cost(self,
                            asset_class: AssetClass,
                            trade_value: float,
                            liquidity_factor: float = 1.0,
                            volatility_factor: float = 1.0) -> float:
        """Calculate bid-ask spread cost"""
        base_spread_bps = self.typical_spreads[asset_class]
        
        # Adjust for market conditions
        adjusted_spread_bps = (base_spread_bps * 
                              (1 / liquidity_factor) * 
                              (1 + volatility_factor))
        
        # Convert to dollar cost (half spread for one-way transaction)
        spread_cost = trade_value * (adjusted_spread_bps / 10000) * 0.5
        
        return spread_cost
    
    def calculate_market_impact(self,
                              asset_class: AssetClass,
                              trade_value: float,
                              avg_daily_volume: float,
                              side: str = 'buy') -> Tuple[float, float]:
        """
        Calculate market impact cost (temporary and permanent)
        
        Args:
            asset_class: Asset class
            trade_value: Value of the trade
            avg_daily_volume: Average daily trading volume
            side: 'buy' or 'sell'
            
        Returns:
            Tuple of (temporary_impact, permanent_impact)
        """
        if avg_daily_volume <= 0:
            return 0.0, 0.0
        
        params = self.market_impact_params[asset_class]
        
        # Participation rate (trade size as fraction of daily volume)
        participation_rate = trade_value / avg_daily_volume
        
        # Market impact using square root law
        temporary_impact = (params['temporary'] * 
                           (participation_rate ** params['power']) * 
                           trade_value)
        
        permanent_impact = (params['permanent'] * 
                           (participation_rate ** params['power']) * 
                           trade_value)
        
        return temporary_impact, permanent_impact
    
    def calculate_slippage(self,
                         execution_price: float,
                         target_price: float,
                         quantity: float) -> float:
        """Calculate slippage cost"""
        price_difference = abs(execution_price - target_price)
        slippage_cost = price_difference * abs(quantity)
        
        return slippage_cost
    
    def calculate_financing_cost(self,
                               asset_class: AssetClass,
                               position_value: float,
                               holding_period_days: int,
                               is_long: bool = True) -> float:
        """Calculate financing cost for overnight positions"""
        if holding_period_days <= 0:
            return 0.0
        
        annual_rate = self.financing_rates[asset_class]
        
        # Short positions typically receive financing, long positions pay
        if not is_long:
            annual_rate = -annual_rate
        
        # Calculate daily financing cost
        daily_rate = annual_rate / 365
        financing_cost = position_value * daily_rate * holding_period_days
        
        return financing_cost
    
    def calculate_tax_cost(self,
                         profit: float,
                         holding_period_days: int,
                         dividend_income: float = 0.0) -> float:
        """Calculate tax implications"""
        if profit <= 0:
            return 0.0  # No tax on losses
        
        # Determine tax rate based on holding period
        if holding_period_days <= 365:
            tax_rate = self.tax_rates['short_term_capital_gains']
        else:
            tax_rate = self.tax_rates['long_term_capital_gains']
        
        # Capital gains tax
        capital_gains_tax = profit * tax_rate
        
        # Dividend tax
        dividend_tax = dividend_income * self.tax_rates['dividend_tax']
        
        total_tax = capital_gains_tax + dividend_tax
        
        return total_tax
    
    def calculate_total_transaction_cost(self,
                                       symbol: str,
                                       asset_class: AssetClass,
                                       quantity: float,
                                       price: float,
                                       execution_price: float,
                                       avg_daily_volume: float,
                                       liquidity_factor: float = 1.0,
                                       volatility_factor: float = 1.0,
                                       holding_period_days: int = 0,
                                       profit: float = 0.0,
                                       dividend_income: float = 0.0,
                                       side: str = 'buy') -> TransactionCost:
        """
        Calculate comprehensive transaction costs
        
        Args:
            symbol: Asset symbol
            asset_class: Asset class
            quantity: Number of shares/units
            price: Target price
            execution_price: Actual execution price
            avg_daily_volume: Average daily trading volume
            liquidity_factor: Liquidity adjustment factor
            volatility_factor: Volatility adjustment factor
            holding_period_days: Days held (for financing and tax)
            profit: Realized profit (for tax calculation)
            dividend_income: Dividend income (for tax calculation)
            side: 'buy' or 'sell'
            
        Returns:
            TransactionCost object with detailed breakdown
        """
        trade_value = abs(quantity) * price
        
        # Calculate individual cost components
        commission = self.calculate_commission(asset_class, trade_value, quantity)
        
        spread_cost = self.calculate_spread_cost(
            asset_class, trade_value, liquidity_factor, volatility_factor)
        
        temp_impact, perm_impact = self.calculate_market_impact(
            asset_class, trade_value, avg_daily_volume, side)
        market_impact_cost = temp_impact + perm_impact
        
        slippage_cost = self.calculate_slippage(execution_price, price, quantity)
        
        financing_cost = self.calculate_financing_cost(
            asset_class, trade_value, holding_period_days, side == 'buy')
        
        tax_cost = self.calculate_tax_cost(profit, holding_period_days, dividend_income)
        
        # Total cost
        total_cost = (commission + spread_cost + market_impact_cost + 
                     slippage_cost + financing_cost + tax_cost)
        
        # Cost per share
        cost_per_share = total_cost / abs(quantity) if quantity != 0 else 0
        
        # Cost in basis points
        cost_basis_points = (total_cost / trade_value * 10000) if trade_value > 0 else 0
        
        return TransactionCost(
            symbol=symbol,
            trade_value=trade_value,
            commission=commission,
            spread_cost=spread_cost,
            market_impact_cost=market_impact_cost,
            slippage_cost=slippage_cost,
            financing_cost=financing_cost,
            tax_cost=tax_cost,
            total_cost=total_cost,
            cost_per_share=cost_per_share,
            cost_basis_points=cost_basis_points
        )
    
    def estimate_round_trip_cost(self,
                               symbol: str,
                               asset_class: AssetClass,
                               quantity: float,
                               price: float,
                               avg_daily_volume: float,
                               liquidity_factor: float = 1.0,
                               volatility_factor: float = 1.0) -> float:
        """
        Estimate round-trip transaction cost (buy + sell)
        
        Args:
            symbol: Asset symbol
            asset_class: Asset class
            quantity: Number of shares/units
            price: Current price
            avg_daily_volume: Average daily trading volume
            liquidity_factor: Liquidity adjustment factor
            volatility_factor: Volatility adjustment factor
            
        Returns:
            Estimated round-trip cost in dollars
        """
        # Calculate buy cost
        buy_cost = self.calculate_total_transaction_cost(
            symbol, asset_class, quantity, price, price, avg_daily_volume,
            liquidity_factor, volatility_factor, side='buy')
        
        # Calculate sell cost
        sell_cost = self.calculate_total_transaction_cost(
            symbol, asset_class, -quantity, price, price, avg_daily_volume,
            liquidity_factor, volatility_factor, side='sell')
        
        # Total round-trip cost
        total_cost = buy_cost.total_cost + sell_cost.total_cost
        
        return total_cost
    
    def get_cost_breakdown_summary(self, transaction_costs: List[TransactionCost]) -> Dict[str, Any]:
        """Get summary of transaction costs"""
        if not transaction_costs:
            return {}
        
        total_trade_value = sum(tc.trade_value for tc in transaction_costs)
        total_cost = sum(tc.total_cost for tc in transaction_costs)
        
        return {
            'total_transactions': len(transaction_costs),
            'total_trade_value': total_trade_value,
            'total_cost': total_cost,
            'avg_cost_per_trade': total_cost / len(transaction_costs),
            'avg_cost_basis_points': (total_cost / total_trade_value * 10000) if total_trade_value > 0 else 0,
            'commission_cost': sum(tc.commission for tc in transaction_costs),
            'spread_cost': sum(tc.spread_cost for tc in transaction_costs),
            'market_impact_cost': sum(tc.market_impact_cost for tc in transaction_costs),
            'slippage_cost': sum(tc.slippage_cost for tc in transaction_costs),
            'financing_cost': sum(tc.financing_cost for tc in transaction_costs),
            'tax_cost': sum(tc.tax_cost for tc in transaction_costs),
            'cost_percentage': (total_cost / total_trade_value * 100) if total_trade_value > 0 else 0
        } 