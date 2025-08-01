# 🚀 Quick Start Guide

## Prerequisites
- Python 3.8+ (you have Python 3.13.5 ✅)
- pip or conda

## Installation & Setup

### Option 1: Using pip (Recommended)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
# Create conda environment
conda env create -f environment.yml
conda activate multi-disease-prediction
```

### Option 3: Using Makefile
```bash
# Setup everything automatically
make setup-dev
make install
```

## Quick Start

### 1. Train Models
```bash
# Train all models
python3 run_pipeline.py

# Or train specific datasets
python3 run_pipeline.py --datasets heart_disease diabetes
```

### 2. Start Web Application
```bash
# Start Streamlit app
streamlit run app/app.py
```

The app will open at: http://localhost:8501

## What You Can Do

### 🏠 Home Page
- View project overview and statistics
- See available trained models

### 🔮 Disease Prediction
- **Manual Input:** Enter patient data for real-time prediction
- **CSV Upload:** Upload batch data for multiple predictions
- **Sample Data:** Test with pre-filled sample data

### 📈 Model Training
- Train models for heart disease, diabetes, and liver disease
- Configure preprocessing options
- View training progress and results

### 📊 Model Performance
- Compare model performance across datasets
- View detailed metrics (Accuracy, Precision, Recall, F1, AUROC)
- Analyze confusion matrices and ROC curves

### 🔍 Model Explainability
- SHAP analysis for model interpretability
- Feature importance visualization
- Individual prediction explanations

### 📁 Data Management
- View dataset information and statistics
- Monitor data quality and preprocessing steps

## Example Usage

### 1. Heart Disease Prediction
```python
from src.model_utils import ModelManager
from src.model_utils import create_sample_data

# Load models
manager = ModelManager()
manager.load_models("heart_disease")

# Create sample data
sample_data = create_sample_data("heart_disease")

# Make prediction
result = manager.predict("heart_disease", sample_data, "ensemble")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['probability']:.2%}")
```

### 2. Batch Prediction
```python
import pandas as pd
from src.model_utils import ModelManager, preprocess_input_data

# Load your data
df = pd.read_csv("your_patient_data.csv")

# Preprocess data
processed_df = preprocess_input_data(df, "heart_disease")

# Make batch predictions
manager = ModelManager()
results = manager.predict_batch("heart_disease", processed_df, "ensemble")

print(f"Predictions: {results['predictions']}")
print(f"Confidence scores: {results['probabilities']}")
```

## Docker Deployment

### Build and Run
```bash
# Build image
docker build -f docker/Dockerfile -t multi-disease-prediction .

# Run container
docker run -p 8501:8501 multi-disease-prediction

# Run with automatic training
docker run -p 8501:8501 -e RUN_PIPELINE=true multi-disease-prediction
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_data_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'src'**
   ```bash
   # Make sure you're in the project root directory
   cd /path/to/Multi-Disease-Prediction-Using-Ensemble-Learning
   ```

2. **SHAP Installation Issues**
   ```bash
   # Try installing SHAP separately
   pip install shap==0.42.1
   ```

3. **Memory Issues with Large Datasets**
   ```bash
   # Reduce feature selection
   # Edit src/config.py: FEATURE_CONFIG["n_features"] = 5
   ```

4. **Streamlit Port Already in Use**
   ```bash
   # Use different port
   streamlit run app/app.py --server.port 8502
   ```

### Getting Help

- Check the logs in `logs/` directory
- Review the comprehensive README.md
- Run tests to verify installation: `pytest tests/ -v`

## Next Steps

1. **Explore the Code:**
   - Review `src/` directory for core functionality
   - Check `notebooks/` for detailed analysis
   - Examine `tests/` for usage examples

2. **Customize the System:**
   - Modify `src/config.py` for different parameters
   - Add new datasets in `src/config.py`
   - Extend models in `src/model_training.py`

3. **Deploy to Production:**
   - Use Docker for containerized deployment
   - Set up CI/CD pipelines
   - Configure monitoring and logging

## Support

For issues and questions:
- Check the comprehensive README.md
- Review the code documentation
- Run the test suite to verify functionality

---

**Happy Predicting! 🏥🤖** 