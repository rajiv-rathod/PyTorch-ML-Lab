import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class for preprocessing data for machine learning models
    """
    def __init__(self):
        self.label_encoders = {}
        self.one_hot_encoders = {}
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
    
    def preprocess_data(self, df, feature_cols, target_col, test_size=0.2, random_state=42):
        """
        Preprocess the data for model training
        
        Args:
            df: Pandas DataFrame with the data
            feature_cols: List of feature column names
            target_col: Target column name
            test_size: Size of the test set
            random_state: Random state for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Preprocessing data with {len(feature_cols)} features and target: {target_col}")
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values in features
        X = self._handle_missing_values(X)
        
        # Encode categorical features
        X = self._encode_categorical_features(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def _handle_missing_values(self, df):
        """
        Handle missing values in the DataFrame
        
        Args:
            df: Pandas DataFrame with features
            
        Returns:
            DataFrame with missing values handled
        """
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # Handle missing values in numeric columns
        if numeric_cols:
            df[numeric_cols] = self.numeric_imputer.fit_transform(df[numeric_cols])
        
        # Handle missing values in categorical columns
        if categorical_cols:
            df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        
        return df
    
    def _encode_categorical_features(self, df):
        """
        Encode categorical features
        
        Args:
            df: Pandas DataFrame with features
            
        Returns:
            DataFrame with encoded categorical features
        """
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            # For columns with fewer unique values, use one-hot encoding
            if df[col].nunique() < 10:
                # Create a one-hot encoder for this column
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                
                # Fit and transform the column
                encoded = encoder.fit_transform(df[[col]])
                
                # Create new column names
                encoded_cols = [f"{col}_{val}" for val in encoder.categories_[0]]
                
                # Create a new DataFrame with the encoded columns
                encoded_df = pd.DataFrame(encoded, index=df.index, columns=encoded_cols)
                
                # Drop the original column and add the encoded columns
                df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
                
                # Store the encoder
                self.one_hot_encoders[col] = encoder
            else:
                # For columns with many unique values, use label encoding
                encoder = LabelEncoder()
                
                # Fit and transform the column
                df[col] = encoder.fit_transform(df[col])
                
                # Store the encoder
                self.label_encoders[col] = encoder
        
        return df
    
    def transform_data(self, df, feature_cols):
        """
        Transform new data using the same preprocessing steps
        
        Args:
            df: Pandas DataFrame with features
            feature_cols: List of feature column names
            
        Returns:
            Preprocessed features
        """
        # Extract features
        X = df[feature_cols].copy()
        
        # Handle missing values in features
        X = self._transform_missing_values(X)
        
        # Encode categorical features
        X = self._transform_categorical_features(X)
        
        return X
    
    def _transform_missing_values(self, df):
        """
        Transform missing values in new data
        
        Args:
            df: Pandas DataFrame with features
            
        Returns:
            DataFrame with missing values handled
        """
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # Handle missing values in numeric columns
        if numeric_cols:
            df[numeric_cols] = self.numeric_imputer.transform(df[numeric_cols])
        
        # Handle missing values in categorical columns
        if categorical_cols:
            df[categorical_cols] = self.categorical_imputer.transform(df[categorical_cols])
        
        return df
    
    def _transform_categorical_features(self, df):
        """
        Transform categorical features in new data
        
        Args:
            df: Pandas DataFrame with features
            
        Returns:
            DataFrame with encoded categorical features
        """
        # Transform one-hot encoded columns
        for col, encoder in self.one_hot_encoders.items():
            if col in df.columns:
                # Transform the column
                encoded = encoder.transform(df[[col]])
                
                # Create new column names
                encoded_cols = [f"{col}_{val}" for val in encoder.categories_[0]]
                
                # Create a new DataFrame with the encoded columns
                encoded_df = pd.DataFrame(encoded, index=df.index, columns=encoded_cols)
                
                # Drop the original column and add the encoded columns
                df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
        
        # Transform label encoded columns
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Transform the column
                df[col] = encoder.transform(df[col])
        
        return df
