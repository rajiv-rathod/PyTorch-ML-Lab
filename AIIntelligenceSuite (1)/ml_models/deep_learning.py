import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ml_models.base_model import BaseModel
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class NeuralNetwork(nn.Module):
    """
    Neural network model for classification or regression
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class DeepLearningModel(BaseModel):
    """
    Implementation of deep learning model using PyTorch
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, learning_rate=0.001, epochs=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = NeuralNetwork(input_dim, hidden_dim, output_dim)
        self.scaler = StandardScaler()
        self.is_classification = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train(self, X_train, y_train, X_val=None, y_val=None, is_classification=True):
        """
        Train the PyTorch model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            is_classification: Whether it's a classification problem
            
        Returns:
            Dictionary with training information
        """
        self.is_classification = is_classification
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        
        # Convert pandas Series to numpy array if needed
        if hasattr(y_train, 'values'):
            y_train_np = y_train.values
        else:
            y_train_np = y_train
            
        # Check if target contains non-numeric data
        if isinstance(y_train_np[0], (str, bytes)) or (hasattr(y_train_np[0], 'dtype') and np.issubdtype(y_train_np[0].dtype, np.datetime64)):
            raise ValueError("Cannot train neural network on string or datetime values. Please select a numeric target column.")
        
        # Convert to appropriate numpy data type
        if is_classification:
            if self.output_dim > 1:  # Multi-class
                y_train_np = y_train_np.astype(int)
                y_train_tensor = torch.LongTensor(y_train_np).to(self.device)
                criterion = nn.CrossEntropyLoss()
            else:  # Binary classification
                y_train_np = y_train_np.astype(float)
                y_train_tensor = torch.FloatTensor(y_train_np.reshape(-1, 1)).to(self.device)
                criterion = nn.BCEWithLogitsLoss()
        else:  # Regression
            y_train_np = y_train_np.astype(float)
            y_train_tensor = torch.FloatTensor(y_train_np.reshape(-1, 1)).to(self.device)
            criterion = nn.MSELoss()
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            
            # Convert pandas Series to numpy array if needed
            if hasattr(y_val, 'values'):
                y_val_np = y_val.values
            else:
                y_val_np = y_val
                
            # Check if target contains non-numeric data
            if isinstance(y_val_np[0], (str, bytes)) or (hasattr(y_val_np[0], 'dtype') and np.issubdtype(y_val_np[0].dtype, np.datetime64)):
                raise ValueError("Cannot validate neural network on string or datetime values. Please select a numeric target column.")
            
            # Convert to appropriate numpy data type
            if is_classification:
                if self.output_dim > 1:  # Multi-class
                    y_val_np = y_val_np.astype(int)
                    y_val_tensor = torch.LongTensor(y_val_np).to(self.device)
                else:  # Binary classification
                    y_val_np = y_val_np.astype(float)
                    y_val_tensor = torch.FloatTensor(y_val_np.reshape(-1, 1)).to(self.device)
            else:  # Regression
                y_val_np = y_val_np.astype(float)
                y_val_tensor = torch.FloatTensor(y_val_np.reshape(-1, 1)).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # For early stopping
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Training history
        train_losses = []
        val_losses = []
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.epochs):
            # Forward pass
            if is_classification and self.output_dim > 1:
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
            else:
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    if is_classification and self.output_dim > 1:
                        val_outputs = self.model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor)
                    else:
                        val_outputs = self.model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor)
                
                val_losses.append(val_loss.item())
                
                # Early stopping
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                self.model.train()
            
            # Log progress
            if (epoch+1) % 10 == 0:
                if X_val is not None and y_val is not None:
                    logger.info(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
                else:
                    logger.info(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
        
        # Create training plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Save the plot to a file
        plot_path = 'static/training_plot.png'
        plt.savefig(plot_path)
        plt.close()
        
        # Return training information
        return {
            'epochs_completed': min(epoch+1, self.epochs),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1] if val_losses else None,
            'best_val_loss': best_val_loss if val_losses else None,
            'early_stopped': patience_counter >= patience,
            'training_plot': plot_path
        }
        
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Predictions
        """
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.is_classification:
                if self.output_dim > 1:  # Multi-class
                    _, predicted = torch.max(outputs, 1)
                    return predicted.cpu().numpy()
                else:  # Binary classification
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    return predicted.cpu().numpy()
            else:  # Regression
                return outputs.cpu().numpy()
    
    def get_feature_importance(self):
        """
        Get feature importance for deep learning models
        
        For neural networks, we'll use a simple gradient-based approach to estimate importance
        """
        # Not implemented for this simple example
        # In a real-world scenario, you could use techniques like permutation importance
        # or integrated gradients
        return None
    
    def save(self, path):
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        model_info = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'is_classification': self.is_classification
        }
        torch.save(model_info, path)
    
    def load(self, path):
        """
        Load the model from disk
        
        Args:
            path: Path to load the model from
        """
        model_info = torch.load(path, map_location=self.device)
        
        self.input_dim = model_info['input_dim']
        self.hidden_dim = model_info['hidden_dim']
        self.output_dim = model_info['output_dim']
        self.is_classification = model_info['is_classification']
        
        self.model = NeuralNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.model.load_state_dict(model_info['model_state_dict'])
        self.model.to(self.device)
        
        self.scaler = model_info['scaler']
