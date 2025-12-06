import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd

class DataLoader:
    """Load and manage financial data"""
    
    @staticmethod
    def load_stock_data(ticker: str, start_date: str, 
                       end_date: str = None) -> pd.DataFrame:
        """Load stock data from Yahoo Finance"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return data
    
    @staticmethod
    def load_multiple_stocks(tickers: List[str], start_date: str,
                            end_date: str = None) -> Dict[str, pd.DataFrame]:
        """Load data for multiple stocks"""
        data = {}
        for ticker in tickers:
            data[ticker] = DataLoader.load_stock_data(ticker, start_date, end_date)
        return data
    
    @staticmethod
    def get_sp500_tickers() -> List[str]:
        """Get S&P 500 tickers"""
        import pandas as pd
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        return table[0]['Symbol'].tolist()