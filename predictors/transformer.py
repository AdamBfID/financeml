import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional
from financeml.predictors.base import BasePredictor

class TransformerModel(nn.Module):
    """Transformer-based prediction model"""
    
    def __init__(self, input_dim: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 4,
                 dropout: float = 0.1, output_dim: int = 1):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = self._generate_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        """Generate positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer
        x = self.transformer(x)
        
        # Take last time step
        x = x[:, -1, :]
        
        # Output layer
        return self.fc(x)

class TransformerPredictor(BasePredictor):
    """Transformer-based stock predictor"""
    
    def __init__(self, lookback: int = 60, forecast_horizon: int = 1,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 4,
                 learning_rate: float = 0.0001, epochs: int = 100,
                 batch_size: int = 32):
        super().__init__(lookback, forecast_horizon)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
        
    def build_model(self, input_dim: int):
        """Build Transformer model"""
        self.model = TransformerModel(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            output_dim=self.forecast_horizon
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = nn.MSELoss()
        
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple] = None):
        """Train the Transformer model"""
        if self.model is None:
            self.build_model(X.shape[-1])
        
        # Similar training loop as LSTM
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.history['train_loss'].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
