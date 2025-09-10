import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from ml_models.base_model import BaseModel
import joblib
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class TraditionalMLModel(BaseModel):
    """
    Implementation of traditional machine learning models using scikit-learn
    """
    def __init__(self, model_name='random_forest', params=None):
        self.model_name = model_name
        self.params = params if params is not None else {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_classification = True
        self.feature_names = None
        self.target_encoder = None
        self.target_type = None
    
    def _get_model(self, is_classification=True):
        """
        Get the appropriate model based on the model name and task
        
        Args:
            is_classification: Whether it's a classification problem
            
        Returns:
            Scikit-learn model instance
        """
        self.is_classification = is_classification
        
        if is_classification:
            if self.model_name == 'random_forest':
                return RandomForestClassifier(**self.params)
            elif self.model_name == 'gradient_boosting':
                return GradientBoostingClassifier(**self.params)
            elif self.model_name == 'logistic_regression':
                return LogisticRegression(**self.params)
            elif self.model_name == 'svm':
                return SVC(probability=True, **self.params)
            elif self.model_name == 'knn':
                return KNeighborsClassifier(**self.params)
            else:
                raise ValueError(f"Unknown classification model: {self.model_name}")
        else:
            if self.model_name == 'random_forest':
                return RandomForestRegressor(**self.params)
            elif self.model_name == 'gradient_boosting':
                return GradientBoostingRegressor(**self.params)
            elif self.model_name == 'linear_regression':
                return LinearRegression(**self.params)
            elif self.model_name == 'lasso':
                return Lasso(**self.params)
            elif self.model_name == 'ridge':
                return Ridge(**self.params)
            elif self.model_name == 'svm':
                return SVR(**self.params)
            elif self.model_name == 'knn':
                return KNeighborsRegressor(**self.params)
            else:
                raise ValueError(f"Unknown regression model: {self.model_name}")
        
    def train(self, X_train, y_train, X_val=None, y_val=None, is_classification=True):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            is_classification: Whether it's a classification problem
            
        Returns:
            Dictionary with training information
        """
        # Store feature names if available
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Get the model
        self.model = self._get_model(is_classification)
        
        # Convert pandas Series to numpy array if needed
        if hasattr(y_train, 'values'):
            y_train_np = y_train.values
        else:
            y_train_np = y_train
            
        # Handle different target data types
        if isinstance(y_train_np[0], (str, bytes)):
            if is_classification:
                # For string targets in classification, we encode them as integers
                unique_values = np.unique(y_train_np)
                value_to_idx = {val: idx for idx, val in enumerate(unique_values)}
                self.target_encoder = value_to_idx
                y_train_encoded = np.array([value_to_idx[val] for val in y_train_np])
                y_train_np = y_train_encoded
            else:
                # Cannot do regression on string targets
                raise ValueError("Cannot perform regression on string values. Please select a numeric target column.")
        elif hasattr(y_train_np[0], 'dtype') and np.issubdtype(y_train_np[0].dtype, np.datetime64):
            # Cannot train on datetime directly
            raise ValueError("Cannot train model on datetime values directly. Please extract numeric features from dates (like year, month, etc).")
        else:
            # For numeric targets, ensure they are in the right format
            if is_classification:
                y_train_np = y_train_np.astype(int)
            else:
                y_train_np = y_train_np.astype(float)
                
        # Train the model
        self.model.fit(X_train_scaled, y_train_np)
        
        # Store the type of target for later use
        self.target_type = type(y_train_np[0]).__name__
        
        # Training info
        training_info = {
            'model_type': self.model_name,
            'task': 'classification' if is_classification else 'regression',
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1]
        }
        
        # For models that have feature importances
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            training_info['feature_importance'] = feature_importance.tolist()
            
            # Create feature importance plot
            plt.figure(figsize=(10, 6))
            
            if self.feature_names:
                feature_names = self.feature_names
            else:
                feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
            
            sorted_idx = np.argsort(feature_importance)
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance (Higher is More Important)')
            
            # Save the plot to a file
            importance_path = 'static/feature_importance.png'
            plt.savefig(importance_path)
            plt.close()
            
            training_info['importance_plot'] = importance_path
            
        return training_info
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Handle X as pandas DataFrame
        if hasattr(X, 'values'):
            # Scale the features
            X_scaled = self.scaler.transform(X)
        else:
            # Scale the features
            X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Decode string targets if needed
        if self.is_classification and self.target_encoder is not None:
            # Invert the encoder mapping
            idx_to_value = {idx: val for val, idx in self.target_encoder.items()}
            # Decode the predictions
            return np.array([idx_to_value[pred] for pred in predictions])
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get probability predictions for classification
        
        Args:
            X: Features to make predictions on
            
        Returns:
            Probability predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification models")
        
        # Handle X as pandas DataFrame
        if hasattr(X, 'values'):
            # Scale the features
            X_scaled = self.scaler.transform(X)
        else:
            # Scale the features
            X_scaled = self.scaler.transform(X)
        
        # Check if the model has predict_proba method
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            raise ValueError(f"The model {self.model_name} does not support probability predictions")
    
    def get_feature_importance(self):
        """
        Get feature importance if available
        
        Returns:
            Feature importance if available, else None
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            if self.feature_names:
                return dict(zip(self.feature_names, importance))
            else:
                return importance
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            
            if self.feature_names:
                if len(coef.shape) == 1:
                    return dict(zip(self.feature_names, coef))
                else:
                    return {f"Class {i}": dict(zip(self.feature_names, coef[i])) for i in range(coef.shape[0])}
            else:
                return coef
        else:
            return None
    
    def save(self, path):
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        model_info = {
            'model': self.model,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'is_classification': self.is_classification,
            'feature_names': self.feature_names,
            'target_encoder': self.target_encoder,
            'target_type': self.target_type
        }
        
        joblib.dump(model_info, path)
    
    def load(self, path):
        """
        Load the model from disk
        
        Args:
            path: Path to load the model from
        """
        model_info = joblib.load(path)
        
        self.model = model_info['model']
        self.scaler = model_info['scaler']
        self.model_name = model_info['model_name']
        self.is_classification = model_info['is_classification']
        self.feature_names = model_info['feature_names']
        
        # Load target encoding information if available
        self.target_encoder = model_info.get('target_encoder', None)
        self.target_type = model_info.get('target_type', None)
