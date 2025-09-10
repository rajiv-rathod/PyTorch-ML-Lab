import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance
import os
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Class for creating visualizations for machine learning models
    """
    def __init__(self):
        # Create the static directory if it doesn't exist
        os.makedirs('static', exist_ok=True)
        
        # Use dark background for plots to match the Bootstrap theme
        plt.style.use('dark_background')
    
    def create_visualizations(self, model, X_test, y_test, is_classification=True):
        """
        Create visualizations for model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            is_classification: Whether it's a classification problem
            
        Returns:
            Dictionary with paths to the visualization files
        """
        plot_paths = {}
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Create visualizations based on the task
        if is_classification:
            plot_paths.update(self._create_classification_plots(model, X_test, y_test, y_pred))
        else:
            plot_paths.update(self._create_regression_plots(y_test, y_pred))
        
        # Feature importance plot if available
        if hasattr(model, 'get_feature_importance') and model.get_feature_importance() is not None:
            plot_paths['feature_importance'] = self._create_feature_importance_plot(model, X_test)
        
        return plot_paths
    
    def _create_classification_plots(self, model, X_test, y_test, y_pred):
        """
        Create plots for classification models
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with paths to the classification plots
        """
        plots = {}
        
        # Confusion matrix plot
        cm_path = 'static/confusion_matrix.png'
        plots['confusion_matrix'] = self._create_confusion_matrix_plot(y_test, y_pred, cm_path)
        
        # Check if binary classification
        if len(np.unique(y_test)) == 2:
            # ROC curve if the model supports probability predictions
            if hasattr(model, 'predict_proba'):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    roc_path = 'static/roc_curve.png'
                    plots['roc_curve'] = self._create_roc_curve_plot(y_test, y_prob, roc_path)
                    
                    # Precision-recall curve
                    pr_path = 'static/precision_recall_curve.png'
                    plots['precision_recall_curve'] = self._create_precision_recall_curve_plot(y_test, y_prob, pr_path)
                except:
                    logger.warning("Could not create ROC or PR curve plots")
        
        return plots
    
    def _create_regression_plots(self, y_test, y_pred):
        """
        Create plots for regression models
        
        Args:
            y_test: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with paths to the regression plots
        """
        plots = {}
        
        # Predicted vs Actual plot
        pred_actual_path = 'static/predicted_vs_actual.png'
        plots['predicted_vs_actual'] = self._create_predicted_vs_actual_plot(y_test, y_pred, pred_actual_path)
        
        # Residuals plot
        residuals_path = 'static/residuals.png'
        plots['residuals'] = self._create_residuals_plot(y_test, y_pred, residuals_path)
        
        # Residuals distribution plot
        residuals_dist_path = 'static/residuals_distribution.png'
        plots['residuals_distribution'] = self._create_residuals_distribution_plot(y_test, y_pred, residuals_dist_path)
        
        return plots
    
    def _create_confusion_matrix_plot(self, y_true, y_pred, output_path):
        """
        Create confusion matrix plot
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        # Get the number of classes
        n_classes = cm.shape[0]
        
        # Create a heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add class labels if few classes
        if n_classes <= 10:
            plt.xticks(np.arange(n_classes) + 0.5, range(n_classes))
            plt.yticks(np.arange(n_classes) + 0.5, range(n_classes))
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _create_roc_curve_plot(self, y_true, y_prob, output_path):
        """
        Create ROC curve plot
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities for the positive class
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(8, 6))
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _create_precision_recall_curve_plot(self, y_true, y_prob, output_path):
        """
        Create precision-recall curve plot
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities for the positive class
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(8, 6))
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        # Plot precision-recall curve
        plt.plot(recall, precision, label='Precision-Recall curve')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _create_predicted_vs_actual_plot(self, y_true, y_pred, output_path):
        """
        Create predicted vs actual plot for regression
        
        Args:
            y_true: True values
            y_pred: Predicted values
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(8, 6))
        
        # Create scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _create_residuals_plot(self, y_true, y_pred, output_path):
        """
        Create residuals plot for regression
        
        Args:
            y_true: True values
            y_pred: Predicted values
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(8, 6))
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create scatter plot
        plt.scatter(y_pred, residuals, alpha=0.6)
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual')
        
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _create_residuals_distribution_plot(self, y_true, y_pred, output_path):
        """
        Create residuals distribution plot for regression
        
        Args:
            y_true: True values
            y_pred: Predicted values
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(8, 6))
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Create histogram with KDE
        sns.histplot(residuals, kde=True)
        
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _create_feature_importance_plot(self, model, X_test, output_path='static/feature_importance.png'):
        """
        Create feature importance plot
        
        Args:
            model: Trained model
            X_test: Test features
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(10, 8))
        
        # Try to get feature importance
        feature_importance = model.get_feature_importance()
        
        if feature_importance is None:
            # Try permutation importance instead
            try:
                result = permutation_importance(model.model, X_test, y_test, n_repeats=10, random_state=42)
                feature_importance = result.importances_mean
            except:
                logger.warning("Could not calculate feature importance")
                return None
        
        # If feature_importance is a dict, convert to array
        if isinstance(feature_importance, dict):
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())
        else:
            features = [f"Feature {i}" for i in range(len(feature_importance))]
            importance = feature_importance
        
        # Sort by importance
        sorted_idx = np.argsort(importance)
        
        # Plot top 20 features
        if len(sorted_idx) > 20:
            sorted_idx = sorted_idx[-20:]
        
        plt.barh([features[i] for i in sorted_idx], [importance[i] for i in sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance (Higher is More Important)')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
