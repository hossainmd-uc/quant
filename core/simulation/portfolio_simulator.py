"""
Portfolio Simulation Engine

Core simulation engine that manages portfolio evolution over time with realistic
market conditions, position sizing, and risk management integration.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Position:
    """Represents a position in the portfolio"""
    symbol: str
    quantity: float
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_price(self, price: float) -> None:
        """Update current price and unrealized PnL"""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity
    
    def close_position(self, exit_price: float) -> float:
        """Close position and return realized PnL"""
        self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        return self.realized_pnl

@dataclass
class Trade:
    """Represents a trade execution"""
    symbol: str
    quantity: float
    price: float
    timestamp: datetime
    side: str  # 'buy' or 'sell'
    transaction_cost: float = 0.0
    slippage: float = 0.0
    
@dataclass
class SimulationResult:
    """Results from portfolio simulation"""
    portfolio_values: pd.Series
    positions: Dict[str, Position]
    trades: List[Trade]
    cash: float
    total_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    daily_returns: pd.Series
    monthly_returns: pd.Series
    performance_metrics: Dict[str, float]

class PortfolioSimulator:
    """
    Comprehensive portfolio simulation engine with realistic market conditions
    """
    
    def __init__(self, 
                 initial_cash: float = 100000,
                 max_position_size: float = 0.1,
                 transaction_cost_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 margin_requirement: float = 0.5,
                 max_leverage: float = 2.0):
        """
        Initialize portfolio simulator
        
        Args:
            initial_cash: Starting cash amount
            max_position_size: Maximum position size as fraction of portfolio
            transaction_cost_rate: Transaction cost rate (0.001 = 0.1%)
            slippage_rate: Slippage rate (0.0005 = 0.05%)
            margin_requirement: Margin requirement for leveraged positions
            max_leverage: Maximum leverage allowed
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_position_size = max_position_size
        self.transaction_cost_rate = transaction_cost_rate
        self.slippage_rate = slippage_rate
        self.margin_requirement = margin_requirement
        self.max_leverage = max_leverage
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        self.daily_returns: List[float] = []
        
        # Performance tracking
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])
                total_value += position.quantity * prices[symbol]
        
        return total_value
    
    def get_position_size(self, symbol: str, price: float, signal_strength: float = 1.0) -> float:
        """
        Calculate position size based on risk management rules
        
        Args:
            symbol: Asset symbol
            price: Current price
            signal_strength: Signal strength (0-1)
            
        Returns:
            Position size in shares
        """
        portfolio_value = self.get_portfolio_value({symbol: price})
        max_position_value = portfolio_value * self.max_position_size * signal_strength
        
        # Account for margin requirements
        available_cash = self.cash / self.margin_requirement
        max_affordable = min(max_position_value, available_cash)
        
        position_size = max_affordable / price
        return position_size
    
    def execute_trade(self, symbol: str, quantity: float, price: float, 
                     timestamp: datetime, side: str) -> bool:
        """
        Execute a trade with realistic market conditions
        
        Args:
            symbol: Asset symbol
            quantity: Number of shares
            price: Target price
            timestamp: Trade timestamp
            side: 'buy' or 'sell'
            
        Returns:
            True if trade executed successfully
        """
        # Apply slippage
        if side == 'buy':
            execution_price = price * (1 + self.slippage_rate)
        else:
            execution_price = price * (1 - self.slippage_rate)
        
        # Calculate transaction cost
        trade_value = abs(quantity) * execution_price
        transaction_cost = trade_value * self.transaction_cost_rate
        
        # Check if we have enough cash for buy orders
        if side == 'buy':
            required_cash = trade_value + transaction_cost
            if required_cash > self.cash:
                return False  # Insufficient funds
        
        # Execute trade
        if side == 'buy':
            self.cash -= trade_value + transaction_cost
            
            if symbol in self.positions:
                # Add to existing position
                existing_pos = self.positions[symbol]
                new_quantity = existing_pos.quantity + quantity
                new_avg_price = ((existing_pos.quantity * existing_pos.entry_price + 
                                quantity * execution_price) / new_quantity)
                existing_pos.quantity = new_quantity
                existing_pos.entry_price = new_avg_price
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=execution_price,
                    entry_date=timestamp,
                    current_price=execution_price
                )
                
        else:  # sell
            if symbol not in self.positions:
                return False  # No position to sell
            
            position = self.positions[symbol]
            if quantity > position.quantity:
                quantity = position.quantity  # Sell all available
            
            # Calculate realized PnL
            realized_pnl = (execution_price - position.entry_price) * quantity
            
            self.cash += trade_value - transaction_cost
            position.quantity -= quantity
            position.realized_pnl += realized_pnl
            
            # Remove position if fully closed
            if position.quantity <= 0:
                del self.positions[symbol]
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            quantity=quantity,
            price=execution_price,
            timestamp=timestamp,
            side=side,
            transaction_cost=transaction_cost,
            slippage=abs(execution_price - price)
        )
        self.trades.append(trade)
        
        return True
    
    def simulate_strategy(self, 
                         signals: pd.DataFrame,
                         price_data: pd.DataFrame,
                         strategy_name: str = "Custom Strategy") -> SimulationResult:
        """
        Simulate a trading strategy over historical data
        
        Args:
            signals: DataFrame with trading signals (columns: symbol, signal, timestamp)
            price_data: DataFrame with price data (columns: timestamp, symbol, price)
            strategy_name: Name of the strategy
            
        Returns:
            SimulationResult with comprehensive performance metrics
        """
        if signals.empty or price_data.empty:
            raise ValueError("Signals and price data cannot be empty")
        
        # Reset portfolio state
        self.cash = self.initial_cash
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        # Merge signals with price data
        merged_data = pd.merge(signals, price_data, on=['timestamp', 'symbol'], how='inner')
        merged_data = merged_data.sort_values('timestamp')
        
        # Track portfolio value over time
        portfolio_values = []
        timestamps = []
        
        for _, row in merged_data.iterrows():
            timestamp = row['timestamp']
            symbol = row['symbol']
            price = row['price']
            signal = row['signal']
            
            # Get current prices for all positions
            current_prices = price_data[price_data['timestamp'] == timestamp].set_index('symbol')['price'].to_dict()
            
            # Execute trades based on signals
            if signal > 0:  # Buy signal
                position_size = self.get_position_size(symbol, price, abs(signal))
                if position_size > 0:
                    self.execute_trade(symbol, position_size, price, timestamp, 'buy')
            elif signal < 0:  # Sell signal
                if symbol in self.positions:
                    position_size = self.positions[symbol].quantity * abs(signal)
                    self.execute_trade(symbol, position_size, price, timestamp, 'sell')
            
            # Record portfolio value
            portfolio_value = self.get_portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            timestamps.append(timestamp)
            
            # Store portfolio state
            self.portfolio_history.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions': len(self.positions),
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
            })
        
        # Calculate performance metrics
        portfolio_series = pd.Series(portfolio_values, index=timestamps)
        daily_returns = portfolio_series.pct_change().dropna()
        
        # Calculate metrics
        total_return = (portfolio_series.iloc[-1] / self.initial_cash - 1) * 100
        
        # Sharpe ratio (assuming 252 trading days)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Trade statistics
        winning_trades = [t for t in self.trades if 
                         (t.side == 'sell' and symbol in self.positions and 
                          self.positions[symbol].realized_pnl > 0)]
        total_trades = len([t for t in self.trades if t.side == 'sell'])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(pos.realized_pnl for pos in self.positions.values() if pos.realized_pnl > 0)
        gross_loss = abs(sum(pos.realized_pnl for pos in self.positions.values() if pos.realized_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Monthly returns
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Additional performance metrics
        performance_metrics = {
            'total_return': total_return,
            'annualized_return': (portfolio_series.iloc[-1] / self.initial_cash) ** (252 / len(daily_returns)) - 1,
            'volatility': daily_returns.std() * np.sqrt(252),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'avg_trade_return': daily_returns.mean() if len(daily_returns) > 0 else 0,
            'skewness': daily_returns.skew() if len(daily_returns) > 2 else 0,
            'kurtosis': daily_returns.kurtosis() if len(daily_returns) > 2 else 0,
            'calmar_ratio': (total_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else 0,
            'sortino_ratio': daily_returns.mean() / daily_returns[daily_returns < 0].std() if len(daily_returns[daily_returns < 0]) > 0 else 0
        }
        
        return SimulationResult(
            portfolio_values=portfolio_series,
            positions=self.positions.copy(),
            trades=self.trades.copy(),
            cash=self.cash,
            total_value=portfolio_series.iloc[-1] if len(portfolio_series) > 0 else self.initial_cash,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            performance_metrics=performance_metrics
        )
    
    def run_monte_carlo_simulation(self, 
                                  base_signals: pd.DataFrame,
                                  price_data: pd.DataFrame,
                                  num_simulations: int = 1000,
                                  noise_level: float = 0.1) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with signal noise
        
        Args:
            base_signals: Base trading signals
            price_data: Price data
            num_simulations: Number of Monte Carlo runs
            noise_level: Signal noise level (0-1)
            
        Returns:
            Dictionary with simulation statistics
        """
        results = []
        
        for i in range(num_simulations):
            # Add noise to signals
            noisy_signals = base_signals.copy()
            noise = np.random.normal(0, noise_level, len(noisy_signals))
            noisy_signals['signal'] = noisy_signals['signal'] + noise
            
            # Clip signals to [-1, 1] range
            noisy_signals['signal'] = np.clip(noisy_signals['signal'], -1, 1)
            
            # Run simulation
            try:
                result = self.simulate_strategy(noisy_signals, price_data, f"MC_Sim_{i}")
                results.append(result.performance_metrics)
            except Exception as e:
                continue  # Skip failed simulations
        
        if not results:
            return {"error": "All simulations failed"}
        
        # Calculate statistics across simulations
        results_df = pd.DataFrame(results)
        
        return {
            'mean_return': results_df['total_return'].mean(),
            'std_return': results_df['total_return'].std(),
            'mean_sharpe': results_df['sharpe_ratio'].mean(),
            'std_sharpe': results_df['sharpe_ratio'].std(),
            'mean_max_drawdown': results_df['max_drawdown'].mean(),
            'worst_drawdown': results_df['max_drawdown'].max(),
            'best_return': results_df['total_return'].max(),
            'worst_return': results_df['total_return'].min(),
            'win_rate': (results_df['total_return'] > 0).mean(),
            'percentile_5': results_df['total_return'].quantile(0.05),
            'percentile_95': results_df['total_return'].quantile(0.95),
            'num_successful_sims': len(results),
            'all_results': results_df
        } 