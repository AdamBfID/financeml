from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from pandas import DataFrame
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from financeml.predictors.base import BasePredictor

class EnsemblePredictor(BasePredictor):
    """Ensemble predictor combining multiple models"""
    
    def __init__(self, lookback: int = 60, forecast_horizon: int = 1,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__(lookback, forecast_horizon)
        self.models = {
            'lgbm': LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1),
            'xgb': XGBRegressor(n_estimators=200, learning_rate=0.05, verbosity=0),
            'rf': RandomForestRegressor(n_estimators=200, max_depth=10),
            'gbm': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05)
        }
        self.weights = weights or {'lgbm': 0.3, 'xgb': 0.3, 'rf': 0.2, 'gbm': 0.2}
        
    def build_model(self):
        """Models are already initialized"""
        pass
        
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple] = None):
        """Train all ensemble models"""
        # Flatten sequences for tree-based models
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.ravel() if len(y.shape) > 1 else y
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_flat, y_flat)
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted ensemble predictions"""
        X_flat = X.reshape(X.shape[0], -1)
        
        predictions = np.zeros(X_flat.shape[0])
        for name, model in self.models.items():
            pred = model.predict(X_flat)
            predictions += self.weights[name] * pred
        
        return predictions.reshape(-1, 1)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        importance_dict = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
        
        return pd.DataFrame(importance_dict)

