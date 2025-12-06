import ta
from typing import List
import pandas as pd
import numpy as np


import ta
from typing import List
import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Generate technical indicators from price data"""
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        # Ensure columns are 1D Series
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()
        
        # Trend indicators
        df['sma_20'] = ta.trend.sma_indicator(close, window=20)
        df['sma_50'] = ta.trend.sma_indicator(close, window=50)
        df['ema_12'] = ta.trend.ema_indicator(close, window=12)
        df['ema_26'] = ta.trend.ema_indicator(close, window=26)
        df['macd'] = ta.trend.macd(close)
        df['macd_signal'] = ta.trend.macd_signal(close)
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(close, window=14)
        df['stoch'] = ta.momentum.stoch(high, low, close)
        df['williams_r'] = ta.momentum.williams_r(high, low, close)
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(close)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['atr'] = ta.volatility.average_true_range(high, low, close)
        
        # Volume indicators
        df['obv'] = ta.volume.on_balance_volume(close, volume)
        df['vwap'] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
        
        # Price-based features
        df['returns'] = close.pct_change()
        df['log_returns'] = np.log(close / close.shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df
    
    @staticmethod
    def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom engineered features"""
        # Ensure columns are 1D Series
        close = df['Close'].squeeze()
        volume = df['Volume'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        open_ = df['Open'].squeeze()
        
        # Price momentum
        df['momentum_5'] = close / close.shift(5) - 1
        df['momentum_10'] = close / close.shift(10) - 1
        df['momentum_20'] = close / close.shift(20) - 1
        
        # Volume analysis
        df['volume_change'] = volume.pct_change()
        df['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        
        # Price patterns
        df['high_low_ratio'] = high / low
        df['close_open_ratio'] = close / open_
        
        return df
