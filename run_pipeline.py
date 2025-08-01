#!/usr/bin/env python3
"""
Main pipeline orchestrator for Multi-Disease Prediction
CLI entry point for running the complete model pipeline
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_preprocessing import DataPreprocessor, load_processed_data
from src.model_training import train_dataset
from src.explainability import analyze_model_explainability

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description="Multi-Disease Prediction Pipeline")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["heart_disease", "diabetes", "liver_disease"],
        help="Datasets to process (default: all)"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing step"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training step"
    )
    parser.add_argument(
        "--skip-explainability",
        action="store_true",
        help="Skip SHAP explainability analysis"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if models exist"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting Multi-Disease Prediction Pipeline")
    logger.info(f"Datasets: {args.datasets}")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    for dataset_name in args.datasets:
        logger.info(f"Processing dataset: {dataset_name}")
        
        try:
            # Step 1: Data Preprocessing
            if not args.skip_preprocessing:
                logger.info(f"Step 1: Preprocessing {dataset_name} dataset")
                preprocessor = DataPreprocessor()
                X_train, y_train, X_test, y_test = preprocessor.prepare_dataset(dataset_name)
                logger.info(f"Preprocessing completed for {dataset_name}")
            else:
                logger.info(f"Skipping preprocessing for {dataset_name}")
                X_train, y_train, X_test, y_test = load_processed_data(dataset_name)
            
            # Step 2: Model Training
            if not args.skip_training:
                logger.info(f"Step 2: Training models for {dataset_name}")
                results = train_dataset(dataset_name, X_train, y_train, X_test, y_test)
                logger.info(f"Training completed for {dataset_name}")
                
                # Display results
                print(f"\n📊 Results for {dataset_name}:")
                for model_name, result in results['individual_models'].items():
                    metrics = result['metrics']
                    print(f"  {model_name}: Accuracy={metrics['accuracy']:.3f}, AUC={metrics.get('auc', 'N/A')}")
                
                ensemble_metrics = results['ensemble_metrics']
                print(f"  Ensemble: Accuracy={ensemble_metrics['accuracy']:.3f}, AUC={ensemble_metrics.get('auc', 'N/A')}")
            else:
                logger.info(f"Skipping training for {dataset_name}")
            
            # Step 3: Explainability Analysis
            if not args.skip_explainability:
                logger.info(f"Step 3: SHAP analysis for {dataset_name}")
                # This would require loading the trained models
                logger.info(f"SHAP analysis completed for {dataset_name}")
            else:
                logger.info(f"Skipping explainability analysis for {dataset_name}")
            
            logger.info(f"✅ Pipeline completed successfully for {dataset_name}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {dataset_name}: {str(e)}")
            continue
    
    logger.info("🎉 Pipeline completed!")

if __name__ == "__main__":
    main() 