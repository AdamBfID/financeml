import shap
import matplotlib.pyplot as plt
import numpy as np
from typing import List
class ModelExplainer:
    """Model interpretability and explainability"""
    
    def __init__(self, model, X_train: np.ndarray):
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
        
    def calculate_shap_values(self, X_test: np.ndarray):
        """Calculate SHAP values for model predictions"""
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict, self.X_train[:100]
            )
        
        self.shap_values = self.explainer.shap_values(X_test)
        return self.shap_values
    
    def plot_feature_importance(self, feature_names: List[str]):
        """Plot feature importance using SHAP"""
        shap.summary_plot(self.shap_values, feature_names=feature_names)
    
    def plot_waterfall(self, instance_idx: int, feature_names: List[str]):
        """Plot waterfall chart for single prediction"""
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.explainer.expected_value,
                feature_names=feature_names
            )
        )