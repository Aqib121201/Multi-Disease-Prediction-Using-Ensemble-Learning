"""
Configuration file for Multi-Disease Prediction Using Ensemble Learning
Centralized hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, VISUALIZATIONS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "heart_disease": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "columns": [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ],
        "target_column": "target",
        "categorical_columns": ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
        "numerical_columns": ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    },
    "diabetes": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "columns": [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin',
            'bmi', 'diabetes_pedigree', 'age', 'outcome'
        ],
        "target_column": "outcome",
        "categorical_columns": [],
        "numerical_columns": ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                            'insulin', 'bmi', 'diabetes_pedigree', 'age']
    },
    "liver_disease": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
        "columns": [
            'age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase',
            'alamine_aminotransferase', 'aspartate_aminotransferase', 'total_proteins',
            'albumin', 'albumin_and_globulin_ratio', 'dataset'
        ],
        "target_column": "dataset",
        "categorical_columns": ['gender'],
        "numerical_columns": ['age', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase',
                            'alamine_aminotransferase', 'aspartate_aminotransferase', 
                            'total_proteins', 'albumin', 'albumin_and_globulin_ratio']
    }
}

# Model configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss"
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "random_state": 42,
        "probability": True
    }
}

# Training configuration
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "scoring": "roc_auc",
    "n_jobs": -1
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "use_smote": True,
    "use_scaling": True,
    "use_feature_selection": True,
    "feature_selection_method": "mutual_info",
    "n_features": 10
}

# SHAP configuration
SHAP_CONFIG = {
    "max_display": 20,
    "plot_type": "bar",
    "save_plots": True
}

# Streamlit app configuration
STREAMLIT_CONFIG = {
    "page_title": "Multi-Disease Prediction",
    "page_icon": "🏥",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "app.log"
} 