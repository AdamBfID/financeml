from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd
import numpy as np
from typing import Dict
class CreditRiskModel:
    """Credit risk assessment and default prediction"""
    
    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5
        )
        self.feature_importance = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train credit risk model"""
        self.model.fit(X, y)
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict default probability"""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model performance"""
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return {
            'auc_roc': roc_auc_score(y, y_pred_proba),
            'classification_report': classification_report(y, y_pred)
        }