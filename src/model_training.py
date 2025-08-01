"""
Model training module for Multi-Disease Prediction
Implements ensemble learning with Random Forest, XGBoost, and SVM
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from .config import MODEL_CONFIG, TRAINING_CONFIG, MODELS_DIR, VISUALIZATIONS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleModelTrainer:
    """Ensemble model trainer for disease prediction"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.best_model = None
        self.model_scores = {}
        self.feature_importance = {}
        
    def create_individual_models(self) -> Dict[str, Any]:
        """Create individual models with configured parameters"""
        models = {}
        
        # Random Forest
        rf_config = MODEL_CONFIG["random_forest"]
        models["random_forest"] = RandomForestClassifier(**rf_config)
        
        # XGBoost
        xgb_config = MODEL_CONFIG["xgboost"]
        models["xgboost"] = xgb.XGBClassifier(**xgb_config)
        
        # SVM
        svm_config = MODEL_CONFIG["svm"]
        models["svm"] = SVC(**svm_config)
        
        logger.info("Created individual models: Random Forest, XGBoost, SVM")
        return models
    
    def train_individual_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """Train individual models and evaluate performance"""
        models = self.create_individual_models()
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
        
        self.models = models
        self.model_scores = results
        return results
    
    def create_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> VotingClassifier:
        """Create and train ensemble model using voting classifier"""
        logger.info("Creating ensemble model...")
        
        # Get trained individual models
        estimators = []
        for name, model in self.models.items():
            estimators.append((name, model))
        
        # Create voting classifier (soft voting for probabilities)
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        self.ensemble_model = ensemble
        
        logger.info("Ensemble model trained successfully")
        return ensemble
    
    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate ensemble model performance"""
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained yet")
        
        # Make predictions
        y_pred = self.ensemble_model.predict(X_test)
        y_pred_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        logger.info(f"Ensemble - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
        return metrics
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def perform_hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                    model_name: str) -> Any:
        """Perform hyperparameter tuning for a specific model"""
        logger.info(f"Performing hyperparameter tuning for {model_name}")
        
        if model_name == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == "xgboost":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42)
            
        elif model_name == "svm":
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
            model = SVC(probability=True, random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_models(self, dataset_name: str):
        """Save trained models to disk"""
        # Save individual models
        for name, model in self.models.items():
            model_path = MODELS_DIR / f"{dataset_name}_{name}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")
        
        # Save ensemble model
        if self.ensemble_model is not None:
            ensemble_path = MODELS_DIR / f"{dataset_name}_ensemble.joblib"
            joblib.dump(self.ensemble_model, ensemble_path)
            logger.info(f"Saved ensemble model to {ensemble_path}")
        
        # Save model scores
        scores_path = MODELS_DIR / f"{dataset_name}_scores.json"
        import json
        scores_dict = {}
        for name, result in self.model_scores.items():
            scores_dict[name] = {
                'accuracy': result['metrics']['accuracy'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'f1': result['metrics']['f1'],
                'auc': result['metrics'].get('auc', None)
            }
        
        with open(scores_path, 'w') as f:
            json.dump(scores_dict, f, indent=4)
        logger.info(f"Saved model scores to {scores_path}")
    
    def load_models(self, dataset_name: str):
        """Load trained models from disk"""
        # Load individual models
        for name in ["random_forest", "xgboost", "svm"]:
            model_path = MODELS_DIR / f"{dataset_name}_{name}.joblib"
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} model from {model_path}")
        
        # Load ensemble model
        ensemble_path = MODELS_DIR / f"{dataset_name}_ensemble.joblib"
        if ensemble_path.exists():
            self.ensemble_model = joblib.load(ensemble_path)
            logger.info(f"Loaded ensemble model from {ensemble_path}")
    
    def create_visualizations(self, X_test: pd.DataFrame, y_test: pd.Series, 
                            dataset_name: str):
        """Create and save model performance visualizations"""
        logger.info(f"Creating visualizations for {dataset_name}")
        
        # Create ROC curves
        plt.figure(figsize=(12, 8))
        
        for name, result in self.model_scores.items():
            if result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                auc = result['metrics']['auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        # Add ensemble ROC if available
        if self.ensemble_model is not None:
            y_ensemble_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_ensemble_proba)
            auc = roc_auc_score(y_test, y_ensemble_proba)
            plt.plot(fpr, tpr, label=f'Ensemble (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {dataset_name.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save ROC plot
        roc_path = VISUALIZATIONS_DIR / f"roc_curves_{dataset_name}.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ROC curves to {roc_path}")
        
        # Create confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (name, result) in enumerate(self.model_scores.items()):
            cm = result['metrics']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name.replace("_", " ").title()} - Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Add ensemble confusion matrix
        if self.ensemble_model is not None:
            y_ensemble_pred = self.ensemble_model.predict(X_test)
            cm = confusion_matrix(y_test, y_ensemble_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[3])
            axes[3].set_title('Ensemble - Confusion Matrix')
            axes[3].set_xlabel('Predicted')
            axes[3].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save confusion matrix plot
        cm_path = VISUALIZATIONS_DIR / f"confusion_matrices_{dataset_name}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrices to {cm_path}")
        
        # Create feature importance plot (for tree-based models)
        if self.feature_importance:
            plt.figure(figsize=(12, 8))
            
            for name, importance in self.feature_importance.items():
                if importance is not None:
                    # Get feature names (assuming they're the same as X_test columns)
                    feature_names = X_test.columns
                    indices = np.argsort(importance)[::-1]
                    
                    plt.subplot(1, len(self.feature_importance), list(self.feature_importance.keys()).index(name) + 1)
                    plt.bar(range(len(indices)), importance[indices])
                    plt.title(f'{name.replace("_", " ").title()} - Feature Importance')
                    plt.xlabel('Features')
                    plt.ylabel('Importance')
                    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            
            plt.tight_layout()
            
            # Save feature importance plot
            fi_path = VISUALIZATIONS_DIR / f"feature_importance_{dataset_name}.png"
            plt.savefig(fi_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved feature importance to {fi_path}")
    
    def train_complete_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series, 
                              dataset_name: str) -> Dict:
        """Complete training pipeline for a dataset"""
        logger.info(f"Starting complete training pipeline for {dataset_name}")
        
        # Train individual models
        individual_results = self.train_individual_models(X_train, y_train, X_test, y_test)
        
        # Create and train ensemble
        ensemble_model = self.create_ensemble_model(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_metrics = self.evaluate_ensemble(X_test, y_test)
        
        # Create visualizations
        self.create_visualizations(X_test, y_test, dataset_name)
        
        # Save models
        self.save_models(dataset_name)
        
        # Compile final results
        final_results = {
            'individual_models': individual_results,
            'ensemble_metrics': ensemble_metrics,
            'best_model': self.get_best_model()
        }
        
        logger.info(f"Training pipeline completed for {dataset_name}")
        return final_results
    
    def get_best_model(self) -> str:
        """Determine the best performing model"""
        if not self.model_scores:
            return None
        
        best_score = 0
        best_model = None
        
        for name, result in self.model_scores.items():
            score = result['metrics']['auc'] if 'auc' in result['metrics'] else result['metrics']['accuracy']
            if score > best_score:
                best_score = score
                best_model = name
        
        # Check if ensemble is better
        if self.ensemble_model is not None:
            # This would need to be calculated separately
            pass
        
        return best_model

def train_dataset(dataset_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Convenience function to train models for a dataset"""
    trainer = EnsembleModelTrainer()
    return trainer.train_complete_pipeline(X_train, y_train, X_test, y_test, dataset_name) 