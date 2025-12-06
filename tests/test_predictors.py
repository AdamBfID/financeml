import unittest
import numpy as np
from financeml import LSTMPredictor, TransformerPredictor, EnsemblePredictor

class TestPredictors(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(100, 60, 10)
        self.y = np.random.randn(100, 1)
    
    def test_lstm_predictor(self):
        model = LSTMPredictor(lookback=60, epochs=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X[:10])
        self.assertEqual(predictions.shape, (10, 1))
    
    def test_transformer_predictor(self):
        model = TransformerPredictor(lookback=60, epochs=5)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X[:10])
        self.assertEqual(predictions.shape, (10, 1))
    
    def test_ensemble_predictor(self):
        X_flat = self.X.reshape(100, -1)
        model = EnsemblePredictor(lookback=60)
        model.fit(X_flat, self.y.ravel())
        predictions = model.predict(X_flat[:10])
        self.assertEqual(predictions.shape, (10, 1))
