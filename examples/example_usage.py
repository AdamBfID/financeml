import pandas as pd


"""
Example usage of FinanceML library
"""

from financeml import (
    DataLoader, TechnicalIndicators, LSTMPredictor,
    TransformerPredictor, EnsemblePredictor,
    PortfolioOptimizer,
    Backtester, #PerformanceMetrics, 
    VolatilityForecaster
)

def main():
    # 1. Load data
    print("Loading data...")
    loader = DataLoader()
    data = loader.load_stock_data('AAPL', '2020-01-01', '2024-01-01')
    print(data.head())
    
    # 2. Add technical indicators
    print("Engineering features...")
    data = TechnicalIndicators.add_all_indicators(data)
    data = TechnicalIndicators.add_custom_features(data)
    data = data.dropna()
    
    # 3. Prepare features and target
    feature_cols = [col for col in data.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]
    X = data[feature_cols].values
    y = data['Close'].values
    
    # 4. Train LSTM model
    print("Training LSTM model...")
    lstm_model = LSTMPredictor(lookback=60, forecast_horizon=30, epochs=50)
    X_seq, y_seq = lstm_model.prepare_sequences(X, y)
    
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test))
    predictions = lstm_model.predict(X_test)
    
    # 5. Portfolio optimization
    print("Optimizing portfolio...")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    portfolio_data = loader.load_multiple_stocks(tickers, '2020-01-01')
    
    returns = pd.DataFrame({
        ticker: portfolio_data[ticker].pct_change()
        for ticker in tickers
    }).dropna()
    
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    optimizer.fit(returns)
    weights = optimizer.optimize_sharpe()
    metrics = optimizer.get_portfolio_metrics()
    
    print(f"\nOptimal Portfolio:")
    print(f"Expected Return: {metrics['expected_return']:.4f}")
    print(f"Volatility: {metrics['volatility']:.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"\nWeights:")
    for ticker, weight in metrics['weights'].items():
        print(f"  {ticker}: {weight:.4f}")
    
    # 6. Volatility forecasting
    print("\nForecasting volatility...")
    vol_forecaster = VolatilityForecaster(p=1, q=1)
    vol_forecaster.fit(returns['AAPL'])
    vol_forecast = vol_forecaster.forecast(horizon=5)
    print(f"5-day volatility forecast: {vol_forecast}")
    
    # 7. Backtesting
    print("\nRunning backtest...")
    signals = pd.Series(predictions[:, 0], index=data.index[split_idx+60:])
    prices = data['Close'].loc[signals.index]
    
    backtester = Backtester(initial_capital=100000, commission=0.001)
    backtest_results = backtester.run_backtest(signals, prices)
    
    print(f"\nBacktest Results:")
    print(f"Total Return: {backtest_results['total_return']:.4f}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.4f}")
    print(f"Number of Trades: {backtest_results['num_trades']}")

if __name__ == '__main__':
    main()