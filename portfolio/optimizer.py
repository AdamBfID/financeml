import cvxpy as cp
from scipy.optimize import minimize
import numpy as np
from typing import Optional, Dict
import pandas as pd
class PortfolioOptimizer:
    """Modern Portfolio Theory optimizer"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.weights = None
        self.expected_returns = None
        self.cov_matrix = None
        
    def fit(self, returns: pd.DataFrame):
        """Calculate expected returns and covariance matrix"""
        self.expected_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252  # Annualized
        
    def optimize_sharpe(self, constraints: Optional[Dict] = None) -> np.ndarray:
        """Optimize portfolio for maximum Sharpe ratio"""
        n_assets = len(self.expected_returns)
        
        # Objective function: negative Sharpe ratio
        def neg_sharpe(weights):
            port_return = np.dot(weights, self.expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return -(port_return - self.risk_free_rate) / port_vol
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Add custom constraints
        if constraints:
            if 'max_weight' in constraints:
                bounds = tuple((0, constraints['max_weight']) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(neg_sharpe, x0, method='SLSQP', 
                         bounds=bounds, constraints=cons)
        
        self.weights = result.x
        return self.weights
    
    def optimize_min_variance(self) -> np.ndarray:
        """Optimize for minimum variance"""
        n_assets = len(self.expected_returns)
        
        # CVXPY optimization
        w = cp.Variable(n_assets)
        risk = cp.quad_form(w, self.cov_matrix)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        prob = cp.Problem(cp.Minimize(risk), constraints)
        prob.solve()
        
        self.weights = w.value
        return self.weights
    
    def optimize_risk_parity(self) -> np.ndarray:
        """Risk parity optimization"""
        n_assets = len(self.expected_returns)
        
        def risk_parity_objective(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / port_vol
            risk_contrib = weights * marginal_contrib
            target_risk = port_vol / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(risk_parity_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        self.weights = result.x
        return self.weights
    
    def get_portfolio_metrics(self, weights: Optional[np.ndarray] = None) -> Dict:
        """Calculate portfolio metrics"""
        if weights is None:
            weights = self.weights
        
        port_return = np.dot(weights, self.expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol
        
        return {
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'weights': dict(zip(self.expected_returns.index, weights))
        }
