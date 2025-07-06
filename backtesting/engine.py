"""
Advanced Backtesting Engine

This module provides a comprehensive backtesting framework with realistic
market simulation, transaction costs, slippage, and advanced performance metrics.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class Order:
    """Order representation"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0


@dataclass
class Trade:
    """Trade execution record"""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    order_id: str
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """Position representation"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_market_value(self, current_price: float):
        """Update market value and unrealized PnL"""
        self.market_value = self.quantity * current_price
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.avg_price) * self.quantity


@dataclass
class Portfolio:
    """Portfolio representation"""
    cash: float = 100000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 100000.0
    daily_returns: List[float] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    
    def get_position(self, symbol: str) -> Position:
        """Get position for symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]
    
    def update_portfolio_value(self, market_data: Dict[str, float]):
        """Update total portfolio value"""
        total_positions_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.update_market_value(market_data[symbol])
                total_positions_value += position.market_value
        
        self.total_value = self.cash + total_positions_value


class MarketSimulator:
    """Realistic market simulation with slippage and transaction costs"""
    
    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1% commission
        slippage_rate: float = 0.0005,   # 0.05% slippage
        bid_ask_spread: float = 0.001,   # 0.1% bid-ask spread
        market_impact_factor: float = 0.01
    ):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.bid_ask_spread = bid_ask_spread
        self.market_impact_factor = market_impact_factor
    
    def calculate_commission(self, price: float, quantity: float) -> float:
        """Calculate commission"""
        return price * quantity * self.commission_rate
    
    def calculate_slippage(self, price: float, quantity: float, side: OrderSide) -> float:
        """Calculate slippage based on order size and market conditions"""
        base_slippage = self.slippage_rate * price
        market_impact = self.market_impact_factor * np.log(1 + abs(quantity) / 1000)
        
        total_slippage = base_slippage + market_impact
        
        if side == OrderSide.BUY:
            return total_slippage
        else:
            return -total_slippage
    
    def simulate_order_execution(
        self,
        order: Order,
        market_price: float,
        volume: float = 1000000
    ) -> Optional[Trade]:
        """Simulate realistic order execution"""
        if order.order_type == OrderType.MARKET:
            # Market orders execute immediately with slippage
            slippage = self.calculate_slippage(market_price, order.quantity, order.side)
            execution_price = market_price + slippage
            
            commission = self.calculate_commission(execution_price, order.quantity)
            
            return Trade(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=order.timestamp,
                order_id=order.order_id,
                commission=commission,
                slippage=slippage
            )
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders only execute if price is favorable
            if order.side == OrderSide.BUY and market_price <= order.price:
                execution_price = order.price
            elif order.side == OrderSide.SELL and market_price >= order.price:
                execution_price = order.price
            else:
                return None  # Order not filled
            
            commission = self.calculate_commission(execution_price, order.quantity)
            
            return Trade(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=execution_price,
                timestamp=order.timestamp,
                order_id=order.order_id,
                commission=commission,
                slippage=0.0
            )
        
        return None


class PerformanceAnalyzer:
    """Advanced performance analysis"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = np.sqrt(252) * downside_returns.std()
        return excess_returns.mean() * np.sqrt(252) / downside_deviation
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        if len(portfolio_values) == 0:
            return 0.0, 0
        
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start + 1
            else:
                if drawdown_start is not None:
                    max_duration = max(max_duration, current_duration)
                    drawdown_start = None
                    current_duration = 0
        
        # Handle case where drawdown extends to end
        if drawdown_start is not None:
            max_duration = max(max_duration, current_duration)
        
        return max_drawdown, max_duration
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, portfolio_values: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if len(returns) == 0 or len(portfolio_values) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_drawdown, _ = PerformanceAnalyzer.calculate_max_drawdown(portfolio_values)
        
        if max_drawdown == 0:
            return np.inf
        
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        var = PerformanceAnalyzer.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        risk_free_rate: float = 0.02
    ):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Initialize components
        self.market_simulator = MarketSimulator(
            commission_rate=commission_rate,
            slippage_rate=slippage_rate
        )
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Portfolio and tracking
        self.portfolio = Portfolio(cash=initial_capital)
        self.orders: List[Order] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        
        # Performance metrics
        self.metrics: Dict[str, Any] = {}
        
        logger.info(f"BacktestEngine initialized with ${initial_capital:,.2f}")
    
    def submit_order(self, order: Order) -> str:
        """Submit an order"""
        order.order_id = f"order_{len(self.orders)}"
        order.timestamp = datetime.now()
        self.orders.append(order)
        
        logger.debug(f"Order submitted: {order.symbol} {order.side.value} {order.quantity} @ {order.price}")
        return order.order_id
    
    def process_orders(self, market_data: Dict[str, float], timestamp: datetime) -> List[Trade]:
        """Process pending orders"""
        executed_trades = []
        
        for order in self.orders:
            if order.status == OrderStatus.PENDING and order.symbol in market_data:
                trade = self.market_simulator.simulate_order_execution(
                    order, market_data[order.symbol]
                )
                
                if trade:
                    trade.timestamp = timestamp
                    executed_trades.append(trade)
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = trade.quantity
                    order.filled_price = trade.price
                    
                    # Update portfolio
                    self._update_portfolio_from_trade(trade)
        
        return executed_trades
    
    def _update_portfolio_from_trade(self, trade: Trade):
        """Update portfolio from executed trade"""
        position = self.portfolio.get_position(trade.symbol)
        
        if trade.side == OrderSide.BUY:
            # Calculate new average price
            total_shares = position.quantity + trade.quantity
            if total_shares > 0:
                position.avg_price = (
                    (position.avg_price * position.quantity + trade.price * trade.quantity) / total_shares
                )
            position.quantity = total_shares
            self.portfolio.cash -= trade.price * trade.quantity + trade.commission
        else:  # SELL
            position.quantity -= trade.quantity
            self.portfolio.cash += trade.price * trade.quantity - trade.commission
            
            # Calculate realized PnL
            if position.quantity >= 0:
                realized_pnl = (trade.price - position.avg_price) * trade.quantity
                position.realized_pnl += realized_pnl
        
        self.portfolio.trades.append(trade)
    
    def update_portfolio(self, market_data: Dict[str, float], timestamp: datetime):
        """Update portfolio values"""
        # Process any pending orders
        self.process_orders(market_data, timestamp)
        
        # Update portfolio value
        self.portfolio.update_portfolio_value(market_data)
        
        # Track equity curve
        self.equity_curve.append(self.portfolio.total_value)
        self.timestamps.append(timestamp)
        
        # Calculate daily returns
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.portfolio.daily_returns.append(daily_return)
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Run backtest with given strategy"""
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        logger.info(f"Starting backtest from {data.index[0]} to {data.index[-1]}")
        
        # Initialize
        self.portfolio = Portfolio(cash=self.initial_capital)
        self.orders = []
        self.equity_curve = []
        self.timestamps = []
        
        # Run backtest
        for timestamp, row in data.iterrows():
            market_data = row.to_dict()
            
            # Execute strategy
            strategy_orders = strategy_func(market_data, self.portfolio, timestamp)
            
            # Submit orders
            if strategy_orders:
                for order in strategy_orders:
                    self.submit_order(order)
            
            # Update portfolio
            self.update_portfolio(market_data, timestamp)
        
        # Calculate final metrics
        self._calculate_metrics()
        
        logger.info(f"Backtest completed. Final portfolio value: ${self.portfolio.total_value:,.2f}")
        
        return self.get_results()
    
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio.daily_returns) == 0:
            logger.warning("No returns data available for metrics calculation")
            return
        
        returns = pd.Series(self.portfolio.daily_returns)
        portfolio_values = pd.Series(self.equity_curve)
        
        # Basic metrics
        total_return = (self.portfolio.total_value - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Risk metrics
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = self.performance_analyzer.calculate_sharpe_ratio(returns, self.risk_free_rate)
        sortino_ratio = self.performance_analyzer.calculate_sortino_ratio(returns, self.risk_free_rate)
        max_drawdown, drawdown_duration = self.performance_analyzer.calculate_max_drawdown(portfolio_values)
        calmar_ratio = self.performance_analyzer.calculate_calmar_ratio(returns, portfolio_values)
        
        # Risk measures
        var_95 = self.performance_analyzer.calculate_var(returns, 0.05)
        cvar_95 = self.performance_analyzer.calculate_cvar(returns, 0.05)
        
        # Trading metrics
        total_trades = len(self.portfolio.trades)
        winning_trades = sum(1 for trade in self.portfolio.trades 
                           if trade.side == OrderSide.SELL and trade.price > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Total commissions
        total_commissions = sum(trade.commission for trade in self.portfolio.trades)
        
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_commissions': total_commissions,
            'final_portfolio_value': self.portfolio.total_value
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get backtest results"""
        return {
            'portfolio': self.portfolio,
            'equity_curve': self.equity_curve,
            'timestamps': self.timestamps,
            'metrics': self.metrics,
            'trades': self.portfolio.trades
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Equity curve
        axes[0, 0].plot(self.timestamps, self.equity_curve, linewidth=2)
        axes[0, 0].set_title('Portfolio Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Daily returns
        if len(self.portfolio.daily_returns) > 0:
            axes[0, 1].hist(self.portfolio.daily_returns, bins=50, alpha=0.7, color='skyblue')
            axes[0, 1].set_title('Daily Returns Distribution')
            axes[0, 1].set_xlabel('Daily Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
        
        # Drawdown
        if len(self.equity_curve) > 0:
            portfolio_values = pd.Series(self.equity_curve)
            peak = portfolio_values.expanding().max()
            drawdown = (portfolio_values - peak) / peak
            
            axes[1, 0].fill_between(self.timestamps, drawdown, 0, alpha=0.3, color='red')
            axes[1, 0].set_title('Drawdown')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].grid(True)
        
        # Performance metrics table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        metrics_data = [
            ['Total Return', f"{self.metrics.get('total_return', 0):.2%}"],
            ['Annual Return', f"{self.metrics.get('annual_return', 0):.2%}"],
            ['Annual Volatility', f"{self.metrics.get('annual_volatility', 0):.2%}"],
            ['Sharpe Ratio', f"{self.metrics.get('sharpe_ratio', 0):.2f}"],
            ['Max Drawdown', f"{self.metrics.get('max_drawdown', 0):.2%}"],
            ['Total Trades', f"{self.metrics.get('total_trades', 0)}"],
            ['Win Rate', f"{self.metrics.get('win_rate', 0):.2%}"],
        ]
        
        table = axes[1, 1].table(cellText=metrics_data, colLabels=['Metric', 'Value'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_results(self, filepath: str):
        """Export results to CSV"""
        results_df = pd.DataFrame({
            'timestamp': self.timestamps,
            'portfolio_value': self.equity_curve,
            'daily_returns': [np.nan] + self.portfolio.daily_returns
        })
        
        results_df.to_csv(filepath, index=False)
        logger.info(f"Results exported to {filepath}")


def simple_momentum_strategy(market_data: Dict[str, float], portfolio: Portfolio, timestamp: datetime) -> List[Order]:
    """Simple momentum strategy example"""
    orders = []
    
    # Example: Buy if price is above 20-day moving average
    # This is a placeholder - real strategies would use technical indicators
    
    symbol = 'AAPL'  # Example symbol
    if symbol in market_data:
        current_price = market_data[symbol]
        position = portfolio.get_position(symbol)
        
        # Simple buy signal (placeholder logic)
        if position.quantity == 0 and current_price > 100:  # Simple threshold
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
        
        # Simple sell signal
        elif position.quantity > 0 and current_price < 95:  # Simple threshold
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                order_type=OrderType.MARKET,
                timestamp=timestamp
            )
            orders.append(order)
    
    return orders 