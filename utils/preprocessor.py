
import pandas as pd
from typing import List, Tuple


class DataPreprocessor:
    """Data preprocessing utilities"""
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, 
                             method: str = 'forward') -> pd.DataFrame:
        """Handle missing values"""
        if method == 'forward':
            return df.fillna(method='ffill')
        elif method == 'backward':
            return df.fillna(method='bfill')
        elif method == 'interpolate':
            return df.interpolate(method='linear')
        else:
            return df.dropna()
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, columns: List[str],
                       n_std: float = 3) -> pd.DataFrame:
        """Remove outliers using z-score method"""
        df_clean = df.copy()
        for col in columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            df_clean = df_clean[
                (df_clean[col] >= mean - n_std * std) &
                (df_clean[col] <= mean + n_std * std)
            ]
        return df_clean
    
    @staticmethod
    def create_train_test_split(df: pd.DataFrame, 
                               test_size: float = 0.2) -> Tuple:
        """Create time-series aware train-test split"""
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test
