"""
Explainability module for Multi-Disease Prediction
Implements SHAP for model interpretability and feature importance analysis
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

from .config import SHAP_CONFIG, VISUALIZATIONS_DIR, MODELS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """SHAP-based model explainer for disease prediction models"""
    
    def __init__(self):
        self.explainers = {}
        self.shap_values = {}
        self.feature_names = {}
        
    def create_explainer(self, model, X_train: pd.DataFrame, model_name: str):
        """Create SHAP explainer for a model"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping explainer creation.")
            return None
        
        try:
            logger.info(f"Creating SHAP explainer for {model_name}")
            
            # Choose appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For models with probability output (Random Forest, XGBoost)
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.KernelExplainer(model.predict_proba, X_train)
            else:
                # For models without probability output (SVM without probability)
                explainer = shap.KernelExplainer(model.predict, X_train)
            
            self.explainers[model_name] = explainer
            self.feature_names[model_name] = X_train.columns.tolist()
            
            logger.info(f"SHAP explainer created for {model_name}")
            return explainer
            
        except Exception as e:
            logger.error(f"Error creating SHAP explainer for {model_name}: {e}")
            return None
    
    def calculate_shap_values(self, model, X_test: pd.DataFrame, model_name: str):
        """Calculate SHAP values for test data"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP values calculation.")
            return None
        
        try:
            if model_name not in self.explainers:
                self.create_explainer(model, X_test, model_name)
            
            explainer = self.explainers[model_name]
            
            logger.info(f"Calculating SHAP values for {model_name}")
            
            # Calculate SHAP values
            if hasattr(explainer, 'shap_values'):
                # For TreeExplainer
                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):
                    # For binary classification, take the positive class
                    shap_values = shap_values[1]
            else:
                # For KernelExplainer
                shap_values = explainer.shap_values(X_test)
            
            self.shap_values[model_name] = shap_values
            logger.info(f"SHAP values calculated for {model_name}")
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values for {model_name}: {e}")
            return None
    
    def create_summary_plot(self, model_name: str, X_test: pd.DataFrame, save_plot: bool = True):
        """Create SHAP summary plot"""
        if not SHAP_AVAILABLE or model_name not in self.shap_values:
            logger.warning(f"SHAP values not available for {model_name}")
            return None
        
        try:
            logger.info(f"Creating SHAP summary plot for {model_name}")
            
            shap_values = self.shap_values[model_name]
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                X_test,
                max_display=SHAP_CONFIG["max_display"],
                show=False
            )
            plt.title(f'SHAP Summary Plot - {model_name.replace("_", " ").title()}')
            plt.tight_layout()
            
            if save_plot:
                plot_path = VISUALIZATIONS_DIR / f"shap_summary_{model_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP summary plot to {plot_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot for {model_name}: {e}")
    
    def create_waterfall_plot(self, model_name: str, X_test: pd.DataFrame, 
                            sample_idx: int = 0, save_plot: bool = True):
        """Create SHAP waterfall plot for a specific sample"""
        if not SHAP_AVAILABLE or model_name not in self.shap_values:
            logger.warning(f"SHAP values not available for {model_name}")
            return None
        
        try:
            logger.info(f"Creating SHAP waterfall plot for {model_name}, sample {sample_idx}")
            
            shap_values = self.shap_values[model_name]
            
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=shap_values[sample_idx].sum(),
                    data=X_test.iloc[sample_idx],
                    feature_names=X_test.columns
                ),
                show=False
            )
            plt.title(f'SHAP Waterfall Plot - {model_name.replace("_", " ").title()} (Sample {sample_idx})')
            plt.tight_layout()
            
            if save_plot:
                plot_path = VISUALIZATIONS_DIR / f"shap_waterfall_{model_name}_sample_{sample_idx}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP waterfall plot to {plot_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating SHAP waterfall plot for {model_name}: {e}")
    
    def create_dependence_plot(self, model_name: str, X_test: pd.DataFrame, 
                             feature_name: str, save_plot: bool = True):
        """Create SHAP dependence plot for a specific feature"""
        if not SHAP_AVAILABLE or model_name not in self.shap_values:
            logger.warning(f"SHAP values not available for {model_name}")
            return None
        
        try:
            logger.info(f"Creating SHAP dependence plot for {model_name}, feature {feature_name}")
            
            shap_values = self.shap_values[model_name]
            
            plt.figure(figsize=(10, 8))
            shap.dependence_plot(
                feature_name,
                shap_values,
                X_test,
                show=False
            )
            plt.title(f'SHAP Dependence Plot - {model_name.replace("_", " ").title()} ({feature_name})')
            plt.tight_layout()
            
            if save_plot:
                plot_path = VISUALIZATIONS_DIR / f"shap_dependence_{model_name}_{feature_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP dependence plot to {plot_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot for {model_name}: {e}")
    
    def create_feature_importance_plot(self, model_name: str, save_plot: bool = True):
        """Create feature importance plot based on SHAP values"""
        if not SHAP_AVAILABLE or model_name not in self.shap_values:
            logger.warning(f"SHAP values not available for {model_name}")
            return None
        
        try:
            logger.info(f"Creating feature importance plot for {model_name}")
            
            shap_values = self.shap_values[model_name]
            feature_names = self.feature_names[model_name]
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(0)
            
            # Sort features by importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=True)
            
            # Create horizontal bar plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(feature_importance)), feature_importance['importance'])
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'Feature Importance (SHAP) - {model_name.replace("_", " ").title()}')
            plt.tight_layout()
            
            if save_plot:
                plot_path = VISUALIZATIONS_DIR / f"shap_feature_importance_{model_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP feature importance plot to {plot_path}")
            
            plt.close()
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot for {model_name}: {e}")
            return None
    
    def create_force_plot(self, model_name: str, X_test: pd.DataFrame, 
                         sample_idx: int = 0, save_plot: bool = True):
        """Create SHAP force plot for a specific sample"""
        if not SHAP_AVAILABLE or model_name not in self.shap_values:
            logger.warning(f"SHAP values not available for {model_name}")
            return None
        
        try:
            logger.info(f"Creating SHAP force plot for {model_name}, sample {sample_idx}")
            
            shap_values = self.shap_values[model_name]
            
            plt.figure(figsize=(12, 6))
            shap.force_plot(
                shap_values[sample_idx],
                X_test.iloc[sample_idx],
                show=False
            )
            plt.title(f'SHAP Force Plot - {model_name.replace("_", " ").title()} (Sample {sample_idx})')
            plt.tight_layout()
            
            if save_plot:
                plot_path = VISUALIZATIONS_DIR / f"shap_force_{model_name}_sample_{sample_idx}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP force plot to {plot_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating SHAP force plot for {model_name}: {e}")
    
    def explain_prediction(self, model, X_sample: pd.DataFrame, model_name: str) -> Dict:
        """Explain a single prediction"""
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not available"}
        
        try:
            # Create explainer if not exists
            if model_name not in self.explainers:
                self.create_explainer(model, X_sample, model_name)
            
            explainer = self.explainers[model_name]
            
            # Calculate SHAP values for the sample
            if hasattr(explainer, 'shap_values'):
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                shap_values = explainer.shap_values(X_sample)
            
            # Get feature contributions
            feature_contributions = {}
            for i, feature in enumerate(X_sample.columns):
                feature_contributions[feature] = {
                    'value': float(X_sample.iloc[0, i]),
                    'shap_value': float(shap_values[0, i])
                }
            
            # Sort by absolute SHAP value
            sorted_features = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]['shap_value']),
                reverse=True
            )
            
            return {
                'prediction': float(model.predict(X_sample)[0]),
                'probability': float(model.predict_proba(X_sample)[0, 1]) if hasattr(model, 'predict_proba') else None,
                'feature_contributions': dict(sorted_features),
                'total_shap': float(shap_values.sum())
            }
            
        except Exception as e:
            logger.error(f"Error explaining prediction for {model_name}: {e}")
            return {"error": str(e)}
    
    def create_comprehensive_analysis(self, model, X_train: pd.DataFrame, 
                                   X_test: pd.DataFrame, model_name: str):
        """Create comprehensive SHAP analysis for a model"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping comprehensive analysis.")
            return
        
        logger.info(f"Creating comprehensive SHAP analysis for {model_name}")
        
        # Calculate SHAP values
        self.calculate_shap_values(model, X_test, model_name)
        
        # Create various plots
        self.create_summary_plot(model_name, X_test)
        self.create_feature_importance_plot(model_name)
        
        # Create waterfall plot for first sample
        self.create_waterfall_plot(model_name, X_test, sample_idx=0)
        
        # Create force plot for first sample
        self.create_force_plot(model_name, X_test, sample_idx=0)
        
        # Create dependence plots for top features
        feature_importance = self.create_feature_importance_plot(model_name, save_plot=False)
        if feature_importance is not None:
            top_features = feature_importance.tail(3)['feature'].tolist()
            for feature in top_features:
                self.create_dependence_plot(model_name, X_test, feature)
        
        logger.info(f"Comprehensive SHAP analysis completed for {model_name}")

def analyze_model_explainability(models: Dict, X_train: pd.DataFrame, 
                               X_test: pd.DataFrame, dataset_name: str):
    """Analyze explainability for all models in a dataset"""
    explainer = ModelExplainer()
    
    for model_name, model in models.items():
        logger.info(f"Analyzing explainability for {model_name}")
        explainer.create_comprehensive_analysis(model, X_train, X_test, model_name)
    
    return explainer

def explain_single_prediction(model, X_sample: pd.DataFrame, model_name: str) -> Dict:
    """Explain a single prediction using SHAP"""
    explainer = ModelExplainer()
    return explainer.explain_prediction(model, X_sample, model_name) 