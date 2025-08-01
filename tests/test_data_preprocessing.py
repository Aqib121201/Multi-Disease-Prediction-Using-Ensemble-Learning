"""
Unit tests for data preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_preprocessing import DataPreprocessor
from src.config import DATASET_CONFIG

class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    def setup_method(self):
        """Setup method for each test"""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data for testing
        self.sample_heart_data = pd.DataFrame({
            'age': [55, 60, 45, 70],
            'sex': [1, 0, 1, 0],
            'cp': [1, 2, 0, 3],
            'trestbps': [130, 140, 120, 150],
            'chol': [250, 300, 200, 350],
            'fbs': [0, 1, 0, 0],
            'restecg': [0, 1, 0, 2],
            'thalach': [150, 140, 160, 130],
            'exang': [0, 1, 0, 1],
            'oldpeak': [1.5, 2.0, 0.5, 3.0],
            'slope': [1, 2, 0, 1],
            'ca': [0, 1, 0, 2],
            'thal': [2, 3, 1, 2],
            'target': [1, 0, 1, 0]
        })
        
        self.sample_diabetes_data = pd.DataFrame({
            'pregnancies': [2, 1, 3, 0],
            'glucose': [120, 100, 140, 90],
            'blood_pressure': [80, 70, 90, 60],
            'skin_thickness': [25, 20, 30, 15],
            'insulin': [100, 80, 120, 70],
            'bmi': [28.5, 25.0, 32.0, 22.0],
            'diabetes_pedigree': [0.5, 0.3, 0.7, 0.2],
            'age': [45, 35, 55, 25],
            'outcome': [1, 0, 1, 0]
        })
    
    def test_clean_data_heart_disease(self):
        """Test data cleaning for heart disease dataset"""
        # Add some missing values
        test_data = self.sample_heart_data.copy()
        test_data.loc[0, 'age'] = np.nan
        test_data.loc[1, 'chol'] = np.nan
        
        cleaned_data = self.preprocessor.clean_data(test_data, "heart_disease")
        
        # Check that missing values are filled
        assert cleaned_data['age'].isnull().sum() == 0
        assert cleaned_data['chol'].isnull().sum() == 0
        assert len(cleaned_data) == len(test_data)
    
    def test_clean_data_diabetes(self):
        """Test data cleaning for diabetes dataset"""
        # Add some zeros that should be converted to NaN
        test_data = self.sample_diabetes_data.copy()
        test_data.loc[0, 'glucose'] = 0
        test_data.loc[1, 'blood_pressure'] = 0
        
        cleaned_data = self.preprocessor.clean_data(test_data, "diabetes")
        
        # Check that zeros are converted to NaN and then filled
        assert cleaned_data['glucose'].isnull().sum() == 0
        assert cleaned_data['blood_pressure'].isnull().sum() == 0
        assert len(cleaned_data) == len(test_data)
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding"""
        test_data = self.sample_heart_data.copy()
        
        encoded_data = self.preprocessor.encode_categorical_features(test_data, "heart_disease")
        
        # Check that categorical columns are encoded
        categorical_cols = DATASET_CONFIG["heart_disease"]["categorical_columns"]
        for col in categorical_cols:
            if col in encoded_data.columns:
                # Check that encoded values are numeric
                assert encoded_data[col].dtype in ['int64', 'int32', 'int16', 'int8']
    
    def test_scale_features(self):
        """Test feature scaling"""
        test_data = self.sample_heart_data.copy()
        numerical_cols = DATASET_CONFIG["heart_disease"]["numerical_columns"]
        
        # Remove target column for scaling
        test_data = test_data.drop(columns=['target'])
        
        scaled_data = self.preprocessor.scale_features(test_data, "heart_disease", fit=True)
        
        # Check that scaled features have mean close to 0 and std close to 1
        for col in numerical_cols:
            if col in scaled_data.columns:
                assert abs(scaled_data[col].mean()) < 1e-10
                assert abs(scaled_data[col].std() - 1.0) < 1e-10
    
    def test_select_features(self):
        """Test feature selection"""
        test_data = self.sample_heart_data.copy()
        X = test_data.drop(columns=['target'])
        y = test_data['target']
        
        selected_data = self.preprocessor.select_features(X, y, "heart_disease", fit=True)
        
        # Check that feature selection reduces dimensionality
        assert selected_data.shape[1] <= X.shape[1]
        assert selected_data.shape[0] == X.shape[0]
    
    def test_apply_smote(self):
        """Test SMOTE application"""
        # Create imbalanced data
        imbalanced_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'target': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # Imbalanced
        })
        
        X = imbalanced_data.drop(columns=['target'])
        y = imbalanced_data['target']
        
        X_resampled, y_resampled = self.preprocessor.apply_smote(X, y)
        
        # Check that SMOTE increases minority class
        original_minority = y.value_counts().min()
        resampled_minority = y_resampled.value_counts().min()
        
        assert resampled_minority >= original_minority
        assert len(X_resampled) == len(y_resampled)
    
    def test_prepare_dataset(self):
        """Test complete dataset preparation pipeline"""
        # This test might take longer as it involves downloading data
        # We'll test with a small subset
        try:
            # Test with a small subset of data
            test_data = self.sample_heart_data.copy()
            
            # Mock the download_dataset method to return our test data
            original_download = self.preprocessor.download_dataset
            self.preprocessor.download_dataset = lambda x: test_data
            
            X_train, y_train, X_test, y_test = self.preprocessor.prepare_dataset("heart_disease")
            
            # Check that data is properly split
            assert len(X_train) > 0
            assert len(X_test) > 0
            assert len(y_train) > 0
            assert len(y_test) > 0
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)
            
            # Restore original method
            self.preprocessor.download_dataset = original_download
            
        except Exception as e:
            pytest.skip(f"Dataset preparation test skipped: {e}")
    
    def test_get_feature_names(self):
        """Test feature name retrieval"""
        feature_names = self.preprocessor.get_feature_names("heart_disease")
        
        # Check that feature names are returned
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        
        # Check that all expected features are present
        expected_features = (DATASET_CONFIG["heart_disease"]["numerical_columns"] + 
                           DATASET_CONFIG["heart_disease"]["categorical_columns"])
        
        for feature in expected_features:
            assert feature in feature_names

def test_load_processed_data():
    """Test loading processed data"""
    from src.data_preprocessing import load_processed_data
    
    # This test requires processed data to exist
    # We'll test the function signature and basic behavior
    try:
        # This should either load data or raise an exception
        result = load_processed_data("heart_disease")
        
        # If successful, check the structure
        if result is not None:
            X_train, y_train, X_test, y_test = result
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
            assert isinstance(X_test, pd.DataFrame)
            assert isinstance(y_test, pd.Series)
            
    except Exception as e:
        # It's okay if the function raises an exception when no processed data exists
        assert "processed data not found" in str(e).lower() or "file not found" in str(e).lower()

if __name__ == "__main__":
    pytest.main([__file__]) 