import numpy as np
import pandas as pd
from typing import Dict
from financeml.backtesting.backtester import Backtester
class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                              risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series,
                               risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, 
                              equity_curve: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_dd = Backtester._calculate_max_drawdown(equity_curve)
        return annual_return / abs(max_dd)
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = PerformanceMetrics.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
