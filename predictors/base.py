import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler

class BasePredictor(ABC):
    """Abstract base class for all predictors"""
    
    def __init__(self, lookback: int = 60, forecast_horizon: int = 1):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    @abstractmethod
    def build_model(self):
        """Build the prediction model"""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple] = None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series sequences for training"""
        X, y = [], []
        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.forecast_horizon])
        return np.array(X), np.array(y)
    
    def save_model(self, path: str):
        """Save model to disk"""
        raise NotImplementedError
    
    def load_model(self, path: str):
        """Load model from disk"""
        raise NotImplementedError