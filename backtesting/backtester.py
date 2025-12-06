import numpy as np
import pandas as pd
from typing import Dict


class Backtester:
    """Comprehensive backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000,
                 commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, signals: pd.DataFrame, 
                     prices: pd.DataFrame) -> Dict:
        """Run backtest with given signals"""
        capital = self.initial_capital
        position = 0
        
        for i in range(len(signals)):
            # Get signal and price
            signal = signals.iloc[i]
            price = prices.iloc[i]
            
            # Execute trades based on signal
            if signal > 0 and position <= 0:  # Buy
                shares = (capital * 0.95) // price  # 95% of capital
                cost = shares * price * (1 + self.commission)
                if cost <= capital:
                    position += shares
                    capital -= cost
                    self.trades.append({
                        'date': signals.index[i],
                        'action': 'BUY',
                        'shares': shares,
                        'price': price
                    })
            elif signal < 0 and position > 0:  # Sell
                proceeds = position * price * (1 - self.commission)
                capital += proceeds
                self.trades.append({
                    'date': signals.index[i],
                    'action': 'SELL',
                    'shares': position,
                    'price': price
                })
                position = 0
            
            # Record equity
            total_equity = capital + position * price
            self.equity_curve.append(total_equity)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Calculate metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        max_dd = self._calculate_max_drawdown(equity)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'num_trades': len(self.trades),
            'final_equity': equity[-1]
        }
    
    @staticmethod
    def _calculate_max_drawdown(equity: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return np.min(drawdown)