"""
Data preprocessing module for Multi-Disease Prediction
Handles data loading, cleaning, feature engineering, and preparation
"""

import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .config import DATASET_CONFIG, PROCESSED_DATA_DIR, RAW_DATA_DIR, FEATURE_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing class for handling multiple disease datasets"""
    
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selectors = {}
        self.smote = None
        
    def download_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Download dataset from UCI repository or other sources"""
        try:
            config = DATASET_CONFIG[dataset_name]
            url = config["url"]
            columns = config["columns"]
            
            logger.info(f"Downloading {dataset_name} dataset from {url}")
            
            if dataset_name == "heart_disease":
                # Heart disease dataset has missing values marked as '?'
                df = pd.read_csv(url, names=columns, na_values='?')
            elif dataset_name == "diabetes":
                df = pd.read_csv(url, names=columns)
            elif dataset_name == "liver_disease":
                df = pd.read_csv(url, names=columns)
            
            # Save raw data
            raw_file = RAW_DATA_DIR / f"{dataset_name}_raw.csv"
            df.to_csv(raw_file, index=False)
            logger.info(f"Raw data saved to {raw_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_name} dataset: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
        logger.info(f"Cleaning {dataset_name} dataset")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        
        if dataset_name == "heart_disease":
            # For heart disease, replace missing values with median for numerical columns
            numerical_cols = DATASET_CONFIG[dataset_name]["numerical_columns"]
            for col in numerical_cols:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
        
        elif dataset_name == "diabetes":
            # For diabetes, replace zeros with NaN for certain columns, then fill with median
            zero_to_nan_cols = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
            for col in zero_to_nan_cols:
                if col in df.columns:
                    df[col] = df[col].replace(0, np.nan)
            
            # Fill NaN with median
            numerical_cols = DATASET_CONFIG[dataset_name]["numerical_columns"]
            for col in numerical_cols:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
        
        elif dataset_name == "liver_disease":
            # For liver disease, fill missing values with median
            numerical_cols = DATASET_CONFIG[dataset_name]["numerical_columns"]
            for col in numerical_cols:
                if col in df.columns and df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Handled {missing_before - missing_after} missing values")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Encode categorical features using Label Encoding"""
        config = DATASET_CONFIG[dataset_name]
        categorical_columns = config["categorical_columns"]
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[f"{dataset_name}_{col}"] = le
                logger.info(f"Encoded categorical column: {col}")
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, dataset_name: str, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using StandardScaler"""
        config = DATASET_CONFIG[dataset_name]
        numerical_columns = config["numerical_columns"]
        
        df_scaled = df.copy()
        
        if fit:
            scaler = StandardScaler()
            df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
            self.scalers[dataset_name] = scaler
            logger.info(f"Fitted scaler for {dataset_name}")
        else:
            if dataset_name in self.scalers:
                df_scaled[numerical_columns] = self.scalers[dataset_name].transform(df_scaled[numerical_columns])
                logger.info(f"Applied existing scaler for {dataset_name}")
        
        return df_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, dataset_name: str, fit: bool = True) -> pd.DataFrame:
        """Select top features using mutual information"""
        if not FEATURE_CONFIG["use_feature_selection"]:
            return X
        
        n_features = min(FEATURE_CONFIG["n_features"], X.shape[1])
        
        if fit:
            if FEATURE_CONFIG["feature_selection_method"] == "mutual_info":
                selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            else:
                selector = SelectKBest(score_func=f_classif, k=n_features)
            
            X_selected = selector.fit_transform(X, y)
            self.feature_selectors[dataset_name] = selector
            logger.info(f"Selected {n_features} features for {dataset_name}")
        else:
            if dataset_name in self.feature_selectors:
                X_selected = self.feature_selectors[dataset_name].transform(X)
                logger.info(f"Applied existing feature selector for {dataset_name}")
            else:
                X_selected = X
        
        # Get selected feature names
        if fit and dataset_name in self.feature_selectors:
            selected_features = X.columns[self.feature_selectors[dataset_name].get_support()].tolist()
            logger.info(f"Selected features for {dataset_name}: {selected_features}")
        
        return pd.DataFrame(X_selected, columns=X.columns[:n_features] if fit else X.columns)
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE for handling class imbalance"""
        if not FEATURE_CONFIG["use_smote"]:
            return X, y
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        self.smote = smote
        
        logger.info(f"Applied SMOTE. Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    
    def prepare_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Complete pipeline for preparing a dataset"""
        logger.info(f"Starting complete preprocessing pipeline for {dataset_name}")
        
        # Download and load data
        df = self.download_dataset(dataset_name)
        
        # Clean data
        df_clean = self.clean_data(df, dataset_name)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean, dataset_name)
        
        # Separate features and target
        config = DATASET_CONFIG[dataset_name]
        target_column = config["target_column"]
        
        X = df_encoded.drop(columns=[target_column])
        y = df_encoded[target_column]
        
        # Convert target to binary (0/1) for all datasets
        if dataset_name == "heart_disease":
            y = (y > 0).astype(int)  # Convert to binary (0: no disease, 1: disease)
        elif dataset_name == "liver_disease":
            y = (y == 1).astype(int)  # Convert to binary (0: healthy, 1: liver disease)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        # Scale features
        if FEATURE_CONFIG["use_scaling"]:
            X_train_scaled = self.scale_features(X_train, dataset_name, fit=True)
            X_test_scaled = self.scale_features(X_test, dataset_name, fit=False)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Select features
        X_train_selected = self.select_features(X_train_scaled, y_train, dataset_name, fit=True)
        X_test_selected = self.select_features(X_test_scaled, y_test, dataset_name, fit=False)
        
        # Apply SMOTE
        X_train_resampled, y_train_resampled = self.apply_smote(X_train_selected, y_train)
        
        # Save processed data
        processed_file = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
        processed_df = pd.concat([X_train_resampled, y_train_resampled], axis=1)
        processed_df.to_csv(processed_file, index=False)
        logger.info(f"Processed data saved to {processed_file}")
        
        return X_train_resampled, y_train_resampled, X_test_selected, y_test
    
    def get_feature_names(self, dataset_name: str) -> list:
        """Get feature names after preprocessing"""
        config = DATASET_CONFIG[dataset_name]
        all_features = config["numerical_columns"] + config["categorical_columns"]
        
        if dataset_name in self.feature_selectors and FEATURE_CONFIG["use_feature_selection"]:
            selected_indices = self.feature_selectors[dataset_name].get_support()
            return [all_features[i] for i in range(len(all_features)) if selected_indices[i]]
        
        return all_features

def load_processed_data(dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load processed data if available, otherwise process from scratch"""
    processed_file = PROCESSED_DATA_DIR / f"{dataset_name}_processed.csv"
    
    if processed_file.exists():
        logger.info(f"Loading processed data for {dataset_name}")
        df = pd.read_csv(processed_file)
        
        # Separate features and target
        config = DATASET_CONFIG[dataset_name]
        target_column = config["target_column"]
        
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            # If target column is not in processed data, use last column
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, y_train, X_test, y_test
    else:
        logger.info(f"Processed data not found for {dataset_name}, processing from scratch")
        preprocessor = DataPreprocessor()
        return preprocessor.prepare_dataset(dataset_name) 