"""
Transformer-based Time Series Models for Financial Prediction

This module implements state-of-the-art transformer architectures specifically
designed for financial time series forecasting and trading signal generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import random
from loguru import logger


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series forecasting"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int = 1,
        max_len: int = 1000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Input projection
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = src.transpose(0, 1)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Layer normalization
        src = self.layer_norm(src)
        
        # Transformer encoding
        output = self.transformer(src, src_key_padding_mask=src_mask)
        
        # Use the last token for prediction
        output = output[:, -1, :]  # (batch_size, d_model)
        
        # Output projection
        output = self.output_projection(output)
        
        return output


class AttentionTimeSeriesTransformer(nn.Module):
    """Enhanced transformer with multi-head attention visualization"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int = 1,
        max_len: int = 1000
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.nhead = nhead
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim)
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.input_projection.bias.data.zero_()
        self.input_projection.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with attention weight collection"""
        # Input projection
        x = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Clear previous attention weights
        self.attention_weights = []
        
        # Apply attention layers
        for i, (attn_layer, ff_layer, ln1, ln2) in enumerate(
            zip(self.attention_layers, self.ff_layers, self.layer_norms1, self.layer_norms2)
        ):
            # Multi-head attention
            attn_output, attn_weights = attn_layer(x, x, x, key_padding_mask=src_mask)
            
            # Store attention weights
            self.attention_weights.append(attn_weights.detach())
            
            # Add & norm
            x = ln1(x + attn_output)
            
            # Feed-forward
            ff_output = ff_layer(x)
            
            # Add & norm
            x = ln2(x + ff_output)
        
        # Use the last token for prediction
        output = x[:, -1, :]  # (batch_size, d_model)
        
        # Output projection
        output = self.output_projection(output)
        
        return output
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Get attention weights for visualization"""
        return self.attention_weights


class TimeSeriesDataset(Dataset):
    """Dataset for time series data"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        sequence_length: int,
        prediction_length: int = 1,
        feature_columns: Optional[List[str]] = None
    ):
        self.data = data
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.feature_columns = feature_columns or [col for col in data.columns if col != target_column]
        
        # Prepare data
        self.X, self.y = self._prepare_data()
        
        # Normalize features
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.X = self.scaler_X.fit_transform(self.X.reshape(-1, self.X.shape[-1])).reshape(self.X.shape)
        self.y = self.scaler_y.fit_transform(self.y.reshape(-1, 1)).reshape(self.y.shape)
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        X, y = [], []
        
        for i in range(len(self.data) - self.sequence_length - self.prediction_length + 1):
            # Input sequence
            x_seq = self.data[self.feature_columns].iloc[i:i + self.sequence_length].values
            
            # Target sequence
            y_seq = self.data[self.target_column].iloc[
                i + self.sequence_length:i + self.sequence_length + self.prediction_length
            ].values
            
            X.append(x_seq)
            y.append(y_seq)
        
        return np.array(X), np.array(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform predictions"""
        return self.scaler_y.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)


class TransformerPredictor:
    """Main transformer predictor class"""
    
    def __init__(
        self,
        model_type: str = "standard",
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_type = model_type
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        logger.info(f"Initialized TransformerPredictor with device: {device}")
    
    def build_model(self, input_dim: int, output_dim: int = 1) -> None:
        """Build the transformer model"""
        if self.model_type == "standard":
            self.model = TimeSeriesTransformer(
                input_dim=input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                output_dim=output_dim
            )
        elif self.model_type == "attention":
            self.model = AttentionTimeSeriesTransformer(
                input_dim=input_dim,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                output_dim=output_dim
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        logger.info(f"Built {self.model_type} transformer with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the transformer model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                if len(target.shape) > 1:
                    target = target.squeeze(-1)
                
                loss = self.criterion(output.squeeze(), target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.model(data)
                        
                        if len(target.shape) > 1:
                            target = target.squeeze(-1)
                        
                        loss = self.criterion(output.squeeze(), target)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                predictions.append(output.cpu().numpy())
                targets.append(target.numpy())
        
        return np.concatenate(predictions), np.concatenate(targets)
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions, targets = self.predict(data_loader)
        
        if len(targets.shape) > 1:
            targets = targets.squeeze(-1)
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze(-1)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy
        target_direction = np.sign(targets[1:] - targets[:-1])
        pred_direction = np.sign(predictions[1:] - predictions[:-1])
        directional_accuracy = np.mean(target_direction == pred_direction)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy
        }
    
    def save_model(self, path: str) -> None:
        """Save model"""
        if self.model is None:
            raise ValueError("Model not built.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'model_type': self.model_type,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim
            }
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        config = checkpoint['model_config']
        self.build_model(config['input_dim'], config['output_dim'])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {path}")
    
    def get_attention_weights(self) -> Optional[List[torch.Tensor]]:
        """Get attention weights (for attention model)"""
        if isinstance(self.model, AttentionTimeSeriesTransformer):
            return self.model.get_attention_weights()
        else:
            logger.warning("Attention weights only available for AttentionTimeSeriesTransformer")
            return None 