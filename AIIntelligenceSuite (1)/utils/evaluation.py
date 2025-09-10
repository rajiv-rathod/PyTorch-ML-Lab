import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score
)
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Class for evaluating machine learning models
    """
    def evaluate_model(self, model, X_test, y_test, is_classification=True):
        """
        Evaluate the model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            is_classification: Whether it's a classification problem
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics based on the task
        if is_classification:
            return self._evaluate_classification(y_test, y_pred, model)
        else:
            return self._evaluate_regression(y_test, y_pred)
    
    def _evaluate_classification(self, y_true, y_pred, model):
        """
        Evaluate classification model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model: Trained model
            
        Returns:
            Dictionary with classification metrics
        """
        # Initialize metrics dictionary
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Check if binary or multiclass
        is_binary = len(np.unique(y_true)) == 2
        
        if is_binary:
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1'] = f1_score(y_true, y_pred, average='binary')
            
            # ROC AUC if the model supports probability predictions
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                except:
                    # Some models or implementation might not have predict_proba
                    logger.warning("Could not calculate ROC AUC score")
        else:
            # Multiclass metrics
            metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
            metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
            metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
            metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        logger.info(f"Classification metrics: {metrics}")
        
        return metrics
    
    def _evaluate_regression(self, y_true, y_pred):
        """
        Evaluate regression model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with regression metrics
        """
        # Initialize metrics dictionary
        metrics = {}
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        logger.info(f"Regression metrics: {metrics}")
        
        return metrics
