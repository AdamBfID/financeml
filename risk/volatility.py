from arch import arch_model
import pandas as pd

class VolatilityForecaster:
    """GARCH-based volatility forecasting"""
    
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        
    def fit(self, returns: pd.Series):
        """Fit GARCH model"""
        # Scale returns to percentage
        returns_pct = returns * 100
        
        # Fit GARCH model
        self.model = arch_model(returns_pct, vol='Garch', p=self.p, q=self.q)
        self.fitted_model = self.model.fit(disp='off')
        
        return self.fitted_model
    
    def forecast(self, horizon: int = 5) -> pd.DataFrame:
        """Forecast volatility"""
        forecast = self.fitted_model.forecast(horizon=horizon)
        return forecast.variance.iloc[-1] / 100  # Scale back
    
    def get_conditional_volatility(self) -> pd.Series:
        """Get conditional volatility estimates"""
        return self.fitted_model.conditional_volatility / 100
