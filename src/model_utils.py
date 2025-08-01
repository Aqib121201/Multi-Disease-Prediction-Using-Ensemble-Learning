"""
Model utilities for Multi-Disease Prediction
Helper functions for model loading, prediction, and evaluation
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .config import MODELS_DIR, DATASET_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manager class for loading and using trained models"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_models = {}
        self.model_scores = {}
        
    def load_models(self, dataset_name: str) -> Dict:
        """Load all trained models for a dataset"""
        logger.info(f"Loading models for {dataset_name}")
        
        models = {}
        
        # Load individual models
        for model_type in ["random_forest", "xgboost", "svm"]:
            model_path = MODELS_DIR / f"{dataset_name}_{model_type}.joblib"
            if model_path.exists():
                try:
                    models[model_type] = joblib.load(model_path)
                    logger.info(f"Loaded {model_type} model for {dataset_name}")
                except Exception as e:
                    logger.error(f"Error loading {model_type} model: {e}")
        
        # Load ensemble model
        ensemble_path = MODELS_DIR / f"{dataset_name}_ensemble.joblib"
        if ensemble_path.exists():
            try:
                ensemble_model = joblib.load(ensemble_path)
                models["ensemble"] = ensemble_model
                logger.info(f"Loaded ensemble model for {dataset_name}")
            except Exception as e:
                logger.error(f"Error loading ensemble model: {e}")
        
        # Load model scores
        scores_path = MODELS_DIR / f"{dataset_name}_scores.json"
        if scores_path.exists():
            try:
                with open(scores_path, 'r') as f:
                    self.model_scores[dataset_name] = json.load(f)
                logger.info(f"Loaded model scores for {dataset_name}")
            except Exception as e:
                logger.error(f"Error loading model scores: {e}")
        
        self.models[dataset_name] = models
        return models
    
    def predict(self, dataset_name: str, X: pd.DataFrame, model_type: str = "ensemble") -> Dict:
        """Make predictions using a specific model"""
        if dataset_name not in self.models:
            self.load_models(dataset_name)
        
        if dataset_name not in self.models:
            raise ValueError(f"No models found for dataset: {dataset_name}")
        
        models = self.models[dataset_name]
        
        if model_type not in models:
            raise ValueError(f"Model type {model_type} not found for dataset {dataset_name}")
        
        model = models[model_type]
        
        try:
            # Make prediction
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else None
            
            return {
                'prediction': int(prediction),
                'probability': float(probability) if probability is not None else None,
                'model_type': model_type,
                'dataset': dataset_name
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {'error': str(e)}
    
    def predict_batch(self, dataset_name: str, X: pd.DataFrame, model_type: str = "ensemble") -> Dict:
        """Make batch predictions"""
        if dataset_name not in self.models:
            self.load_models(dataset_name)
        
        if dataset_name not in self.models:
            raise ValueError(f"No models found for dataset: {dataset_name}")
        
        models = self.models[dataset_name]
        
        if model_type not in models:
            raise ValueError(f"Model type {model_type} not found for dataset {dataset_name}")
        
        model = models[model_type]
        
        try:
            # Make predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
            
            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None,
                'model_type': model_type,
                'dataset': dataset_name
            }
            
        except Exception as e:
            logger.error(f"Error making batch prediction: {e}")
            return {'error': str(e)}
    
    def get_model_performance(self, dataset_name: str) -> Dict:
        """Get performance metrics for all models"""
        if dataset_name not in self.model_scores:
            # Try to load scores
            scores_path = MODELS_DIR / f"{dataset_name}_scores.json"
            if scores_path.exists():
                with open(scores_path, 'r') as f:
                    self.model_scores[dataset_name] = json.load(f)
        
        if dataset_name in self.model_scores:
            return self.model_scores[dataset_name]
        else:
            return {}
    
    def get_best_model(self, dataset_name: str, metric: str = "auc") -> str:
        """Get the best performing model based on a metric"""
        performance = self.get_model_performance(dataset_name)
        
        if not performance:
            return None
        
        best_model = None
        best_score = -1
        
        for model_name, metrics in performance.items():
            if metric in metrics and metrics[metric] is not None:
                if metrics[metric] > best_score:
                    best_score = metrics[metric]
                    best_model = model_name
        
        return best_model
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets with trained models"""
        available_datasets = []
        
        for model_file in MODELS_DIR.glob("*_ensemble.joblib"):
            dataset_name = model_file.stem.replace("_ensemble", "")
            available_datasets.append(dataset_name)
        
        return available_datasets

def preprocess_input_data(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Preprocess input data for prediction"""
    config = DATASET_CONFIG[dataset_name]
    
    # Ensure all required columns are present
    required_columns = config["numerical_columns"] + config["categorical_columns"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns for {dataset_name}: {missing_columns}")
    
    # Select only required columns
    data = data[required_columns].copy()
    
    # Handle missing values
    numerical_columns = config["numerical_columns"]
    for col in numerical_columns:
        if col in data.columns and data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
    
    # Encode categorical variables
    categorical_columns = config["categorical_columns"]
    for col in categorical_columns:
        if col in data.columns:
            # Simple label encoding (0, 1, 2, ...)
            data[col] = pd.Categorical(data[col]).codes
    
    return data

def create_sample_data(dataset_name: str) -> pd.DataFrame:
    """Create sample data for demonstration"""
    config = DATASET_CONFIG[dataset_name]
    
    if dataset_name == "heart_disease":
        sample_data = {
            'age': [55],
            'sex': [1],  # 1 for male, 0 for female
            'cp': [1],   # chest pain type
            'trestbps': [130],  # resting blood pressure
            'chol': [250],      # cholesterol
            'fbs': [0],         # fasting blood sugar
            'restecg': [0],     # resting ECG
            'thalach': [150],   # max heart rate
            'exang': [0],       # exercise induced angina
            'oldpeak': [1.5],   # ST depression
            'slope': [1],       # slope of peak exercise ST
            'ca': [0],          # number of major vessels
            'thal': [2]         # thalassemia
        }
    
    elif dataset_name == "diabetes":
        sample_data = {
            'pregnancies': [2],
            'glucose': [120],
            'blood_pressure': [80],
            'skin_thickness': [25],
            'insulin': [100],
            'bmi': [28.5],
            'diabetes_pedigree': [0.5],
            'age': [45]
        }
    
    elif dataset_name == "liver_disease":
        sample_data = {
            'age': [45],
            'gender': [1],  # 1 for male, 0 for female
            'total_bilirubin': [1.2],
            'direct_bilirubin': [0.3],
            'alkaline_phosphotase': [120],
            'alamine_aminotransferase': [25],
            'aspartate_aminotransferase': [30],
            'total_proteins': [7.0],
            'albumin': [4.0],
            'albumin_and_globulin_ratio': [1.2]
        }
    
    return pd.DataFrame(sample_data)

def evaluate_model_performance(y_true: pd.Series, y_pred: np.ndarray, 
                             y_prob: np.ndarray = None, model_name: str = "") -> Dict:
    """Evaluate model performance with comprehensive metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    # Add classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['classification_report'] = report
    
    return metrics

def create_performance_comparison_plot(dataset_name: str, save_plot: bool = True):
    """Create performance comparison plot for all models"""
    scores_path = MODELS_DIR / f"{dataset_name}_scores.json"
    
    if not scores_path.exists():
        logger.warning(f"No scores file found for {dataset_name}")
        return
    
    with open(scores_path, 'r') as f:
        scores = json.load(f)
    
    # Prepare data for plotting
    models = list(scores.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [scores[model].get(metric, 0) for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = MODELS_DIR / f"performance_comparison_{dataset_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance comparison plot to {plot_path}")
    
    plt.show()

def get_feature_descriptions(dataset_name: str) -> Dict[str, str]:
    """Get feature descriptions for a dataset"""
    descriptions = {
        "heart_disease": {
            "age": "Age in years",
            "sex": "Sex (1 = male; 0 = female)",
            "cp": "Chest pain type (1-4)",
            "trestbps": "Resting blood pressure (mm Hg)",
            "chol": "Serum cholesterol (mg/dl)",
            "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
            "restecg": "Resting electrocardiographic results (0-2)",
            "thalach": "Maximum heart rate achieved",
            "exang": "Exercise induced angina (1 = yes; 0 = no)",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "Slope of the peak exercise ST segment (1-3)",
            "ca": "Number of major vessels colored by fluoroscopy (0-3)",
            "thal": "Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)"
        },
        "diabetes": {
            "pregnancies": "Number of times pregnant",
            "glucose": "Plasma glucose concentration (mg/dl)",
            "blood_pressure": "Diastolic blood pressure (mm Hg)",
            "skin_thickness": "Triceps skin fold thickness (mm)",
            "insulin": "2-Hour serum insulin (mu U/ml)",
            "bmi": "Body mass index (weight in kg/(height in m)^2)",
            "diabetes_pedigree": "Diabetes pedigree function",
            "age": "Age in years"
        },
        "liver_disease": {
            "age": "Age in years",
            "gender": "Gender (1 = male; 0 = female)",
            "total_bilirubin": "Total bilirubin (mg/dl)",
            "direct_bilirubin": "Direct bilirubin (mg/dl)",
            "alkaline_phosphotase": "Alkaline phosphotase (IU/L)",
            "alamine_aminotransferase": "Alamine aminotransferase (IU/L)",
            "aspartate_aminotransferase": "Aspartate aminotransferase (IU/L)",
            "total_proteins": "Total proteins (g/dl)",
            "albumin": "Albumin (g/dl)",
            "albumin_and_globulin_ratio": "Albumin and globulin ratio"
        }
    }
    
    return descriptions.get(dataset_name, {}) 