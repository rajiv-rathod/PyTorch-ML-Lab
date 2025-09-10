from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models
    """
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, is_classification=True):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            is_classification: Whether the task is classification or regression
            
        Returns:
            Dictionary with training information
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self):
        """
        Get feature importance from the model if available
        
        Returns:
            Feature importance if available, else None
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load the model from disk
        
        Args:
            path: Path to load the model from
        """
        pass
