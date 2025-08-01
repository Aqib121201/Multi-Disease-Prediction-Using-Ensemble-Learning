"""
Streamlit Web Application for Multi-Disease Prediction
Main application file with comprehensive UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.model_utils import ModelManager, preprocess_input_data, create_sample_data, get_feature_descriptions
from src.data_preprocessing import DataPreprocessor, load_processed_data
from src.model_training import train_dataset
from src.explainability import explain_single_prediction

# Page configuration
st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .positive {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .negative {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()

if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Multi-Disease Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### Ensemble Learning with Random Forest, XGBoost, and SVM")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "🔮 Disease Prediction", "📈 Model Training", "📊 Model Performance", "🔍 Model Explainability", "📁 Data Management"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Available Datasets")
        
        # List available datasets
        available_datasets = st.session_state.model_manager.list_available_datasets()
        if available_datasets:
            st.success(f"✅ {len(available_datasets)} trained models available")
            for dataset in available_datasets:
                st.write(f"• {dataset.replace('_', ' ').title()}")
        else:
            st.warning("⚠️ No trained models found. Please train models first.")
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Disease Prediction":
        show_prediction_page()
    elif page == "📈 Model Training":
        show_training_page()
    elif page == "📊 Model Performance":
        show_performance_page()
    elif page == "🔍 Model Explainability":
        show_explainability_page()
    elif page == "📁 Data Management":
        show_data_management_page()

def show_home_page():
    """Display home page with project overview"""
    st.header("🏠 Welcome to Multi-Disease Prediction System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This system uses **Ensemble Learning** to predict multiple diseases with high accuracy and interpretability.
        
        **🔬 Supported Diseases:**
        - **Heart Disease** - Cardiovascular disease prediction
        - **Diabetes** - Diabetes mellitus prediction  
        - **Liver Disease** - Liver disease prediction
        
        **🤖 Machine Learning Models:**
        - **Random Forest** - Ensemble of decision trees
        - **XGBoost** - Gradient boosting framework
        - **SVM** - Support Vector Machine
        - **Ensemble** - Voting classifier combining all models
        
        **🔍 Explainability Features:**
        - **SHAP Analysis** - Model interpretability
        - **Feature Importance** - Understanding key factors
        - **Individual Predictions** - Detailed explanations
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Key Features
        
        ✅ **Multi-Disease Support**
        ✅ **Ensemble Learning**
        ✅ **SHAP Explainability**
        ✅ **Real-time Predictions**
        ✅ **Model Performance Analysis**
        ✅ **Data Visualization**
        ✅ **Batch Processing**
        ✅ **Model Retraining**
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📈 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    available_datasets = st.session_state.model_manager.list_available_datasets()
    
    with col1:
        st.metric("Trained Models", len(available_datasets))
    
    with col2:
        st.metric("Supported Diseases", 3)
    
    with col3:
        st.metric("ML Algorithms", 4)
    
    with col4:
        st.metric("Explainability Tools", 1)

def show_prediction_page():
    """Display disease prediction page"""
    st.header("🔮 Disease Prediction")
    
    # Dataset selection
    available_datasets = st.session_state.model_manager.list_available_datasets()
    
    if not available_datasets:
        st.error("❌ No trained models available. Please train models first in the Model Training page.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        dataset_name = st.selectbox(
            "Select Disease Type:",
            available_datasets,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        model_type = st.selectbox(
            "Select Model:",
            ["ensemble", "random_forest", "xgboost", "svm"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        prediction_method = st.radio(
            "Prediction Method:",
            ["📝 Manual Input", "📁 Upload CSV", "🎲 Sample Data"]
        )
    
    st.markdown("---")
    
    # Prediction interface
    if prediction_method == "📝 Manual Input":
        show_manual_prediction(dataset_name, model_type)
    elif prediction_method == "📁 Upload CSV":
        show_csv_prediction(dataset_name, model_type)
    elif prediction_method == "🎲 Sample Data":
        show_sample_prediction(dataset_name, model_type)

def show_manual_prediction(dataset_name, model_type):
    """Show manual input prediction interface"""
    st.subheader("📝 Manual Input Prediction")
    
    # Get feature descriptions
    feature_descriptions = get_feature_descriptions(dataset_name)
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        input_data = {}
        
        with col1:
            if dataset_name == "heart_disease":
                input_data['age'] = st.number_input("Age", min_value=20, max_value=100, value=55)
                input_data['sex'] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                input_data['cp'] = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                                              help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
                input_data['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=130)
                input_data['chol'] = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
                input_data['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                input_data['restecg'] = st.selectbox("Resting ECG Results", [0, 1, 2], 
                                                   help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
            
            elif dataset_name == "diabetes":
                input_data['pregnancies'] = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=2)
                input_data['glucose'] = st.number_input("Glucose (mg/dl)", min_value=50, max_value=300, value=120)
                input_data['blood_pressure'] = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=150, value=80)
                input_data['skin_thickness'] = st.number_input("Skin Thickness (mm)", min_value=10, max_value=100, value=25)
                input_data['insulin'] = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=100)
                input_data['bmi'] = st.number_input("BMI", min_value=15.0, max_value=70.0, value=28.5, step=0.1)
                input_data['diabetes_pedigree'] = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
                input_data['age'] = st.number_input("Age", min_value=20, max_value=100, value=45)
            
            elif dataset_name == "liver_disease":
                input_data['age'] = st.number_input("Age", min_value=10, max_value=100, value=45)
                input_data['gender'] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                input_data['total_bilirubin'] = st.number_input("Total Bilirubin (mg/dl)", min_value=0.0, max_value=30.0, value=1.2, step=0.1)
                input_data['direct_bilirubin'] = st.number_input("Direct Bilirubin (mg/dl)", min_value=0.0, max_value=20.0, value=0.3, step=0.1)
                input_data['alkaline_phosphotase'] = st.number_input("Alkaline Phosphotase (IU/L)", min_value=50, max_value=500, value=120)
                input_data['alamine_aminotransferase'] = st.number_input("Alamine Aminotransferase (IU/L)", min_value=10, max_value=200, value=25)
                input_data['aspartate_aminotransferase'] = st.number_input("Aspartate Aminotransferase (IU/L)", min_value=10, max_value=300, value=30)
                input_data['total_proteins'] = st.number_input("Total Proteins (g/dl)", min_value=2.0, max_value=10.0, value=7.0, step=0.1)
                input_data['albumin'] = st.number_input("Albumin (g/dl)", min_value=1.0, max_value=6.0, value=4.0, step=0.1)
                input_data['albumin_and_globulin_ratio'] = st.number_input("Albumin/Globulin Ratio", min_value=0.5, max_value=3.0, value=1.2, step=0.1)
        
        with col2:
            st.markdown("### 📋 Feature Descriptions")
            for feature, description in list(feature_descriptions.items())[:5]:  # Show first 5
                st.markdown(f"**{feature.replace('_', ' ').title()}:** {description}")
        
        submitted = st.form_submit_button("🔮 Make Prediction")
        
        if submitted:
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Preprocess input
            try:
                processed_input = preprocess_input_data(input_df, dataset_name)
                
                # Make prediction
                result = st.session_state.model_manager.predict(dataset_name, processed_input, model_type)
                
                if 'error' not in result:
                    display_prediction_result(result, dataset_name)
                else:
                    st.error(f"Prediction error: {result['error']}")
                    
            except Exception as e:
                st.error(f"Error processing input: {str(e)}")

def show_csv_prediction(dataset_name, model_type):
    """Show CSV upload prediction interface"""
    st.subheader("📁 CSV Upload Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with patient data:",
        type=['csv'],
        help="CSV should contain the required features for the selected disease"
    )
    
    if uploaded_file is not None:
        try:
            # Load CSV
            df = pd.read_csv(uploaded_file)
            st.write("📊 Uploaded Data Preview:")
            st.dataframe(df.head())
            
            # Preprocess data
            processed_df = preprocess_input_data(df, dataset_name)
            
            # Make predictions
            if st.button("🔮 Predict All"):
                with st.spinner("Making predictions..."):
                    result = st.session_state.model_manager.predict_batch(dataset_name, processed_df, model_type)
                
                if 'error' not in result:
                    display_batch_prediction_result(result, df)
                else:
                    st.error(f"Prediction error: {result['error']}")
                    
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

def show_sample_prediction(dataset_name, model_type):
    """Show sample data prediction"""
    st.subheader("🎲 Sample Data Prediction")
    
    # Create sample data
    sample_df = create_sample_data(dataset_name)
    
    st.write("📊 Sample Data:")
    st.dataframe(sample_df)
    
    if st.button("🔮 Predict Sample"):
        try:
            # Preprocess sample data
            processed_sample = preprocess_input_data(sample_df, dataset_name)
            
            # Make prediction
            result = st.session_state.model_manager.predict(dataset_name, processed_sample, model_type)
            
            if 'error' not in result:
                display_prediction_result(result, dataset_name)
            else:
                st.error(f"Prediction error: {result['error']}")
                
        except Exception as e:
            st.error(f"Error processing sample data: {str(e)}")

def display_prediction_result(result, dataset_name):
    """Display prediction result with styling"""
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prediction = result['prediction']
        probability = result['probability']
        
        if prediction == 1:
            st.markdown('<div class="prediction-result positive">', unsafe_allow_html=True)
            st.error("🚨 **HIGH RISK** - Disease Detected")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-result negative">', unsafe_allow_html=True)
            st.success("✅ **LOW RISK** - No Disease Detected")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Prediction", "Positive" if prediction == 1 else "Negative")
    
    with col3:
        if probability is not None:
            st.metric("Confidence", f"{probability:.1%}")
            
            # Create confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Level"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)

def display_batch_prediction_result(result, original_df):
    """Display batch prediction results"""
    st.markdown("---")
    st.subheader("📊 Batch Prediction Results")
    
    # Add predictions to original dataframe
    result_df = original_df.copy()
    result_df['Prediction'] = result['predictions']
    if result['probabilities']:
        result_df['Confidence'] = result['probabilities']
    
    # Display results
    st.dataframe(result_df)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive_count = sum(result['predictions'])
        st.metric("Positive Predictions", positive_count)
    
    with col2:
        negative_count = len(result['predictions']) - positive_count
        st.metric("Negative Predictions", negative_count)
    
    with col3:
        if result['probabilities']:
            avg_confidence = np.mean(result['probabilities'])
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
    
    # Download results
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name=f"prediction_results_{result['dataset']}.csv",
        mime="text/csv"
    )

def show_training_page():
    """Display model training page"""
    st.header("📈 Model Training")
    
    # Dataset selection for training
    datasets = ["heart_disease", "diabetes", "liver_disease"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_dataset = st.selectbox(
            "Select Dataset to Train:",
            datasets,
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        st.markdown("### 📊 Dataset Information")
        if selected_dataset == "heart_disease":
            st.write("**Heart Disease Dataset**")
            st.write("- Source: UCI Machine Learning Repository")
            st.write("- Features: 13 clinical parameters")
            st.write("- Target: Binary classification")
        elif selected_dataset == "diabetes":
            st.write("**Diabetes Dataset**")
            st.write("- Source: Pima Indians Diabetes Database")
            st.write("- Features: 8 medical parameters")
            st.write("- Target: Binary classification")
        elif selected_dataset == "liver_disease":
            st.write("**Liver Disease Dataset**")
            st.write("- Source: Indian Liver Patient Dataset")
            st.write("- Features: 10 clinical parameters")
            st.write("- Target: Binary classification")
    
    st.markdown("---")
    
    # Training options
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_smote = st.checkbox("Use SMOTE for Class Imbalance", value=True)
        use_scaling = st.checkbox("Use Feature Scaling", value=True)
        use_feature_selection = st.checkbox("Use Feature Selection", value=True)
    
    with col2:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    # Start training
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                # Load and preprocess data
                preprocessor = DataPreprocessor()
                X_train, y_train, X_test, y_test = preprocessor.prepare_dataset(selected_dataset)
                
                # Train models
                results = train_dataset(selected_dataset, X_train, y_train, X_test, y_test)
                
                st.success("✅ Training completed successfully!")
                
                # Display results
                st.subheader("📊 Training Results")
                
                # Individual model results
                for model_name, result in results['individual_models'].items():
                    metrics = result['metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.3f}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.3f}")
                    with col4:
                        st.metric("F1-Score", f"{metrics['f1']:.3f}")
                
                # Ensemble results
                st.markdown("### 🎯 Ensemble Model Results")
                ensemble_metrics = results['ensemble_metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{ensemble_metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{ensemble_metrics['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{ensemble_metrics['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{ensemble_metrics['f1']:.3f}")
                
                # Refresh model manager
                st.session_state.model_manager.load_models(selected_dataset)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

def show_performance_page():
    """Display model performance page"""
    st.header("📊 Model Performance")
    
    available_datasets = st.session_state.model_manager.list_available_datasets()
    
    if not available_datasets:
        st.error("❌ No trained models available. Please train models first.")
        return
    
    # Dataset selection
    selected_dataset = st.selectbox(
        "Select Dataset:",
        available_datasets,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Load performance data
    performance = st.session_state.model_manager.get_model_performance(selected_dataset)
    
    if not performance:
        st.warning("⚠️ No performance data available for this dataset.")
        return
    
    # Performance overview
    st.subheader("📈 Performance Overview")
    
    # Create performance comparison
    models = list(performance.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Performance table
    performance_df = pd.DataFrame(performance).T
    st.dataframe(performance_df.round(3))
    
    # Performance charts
    st.subheader("📊 Performance Charts")
    
    # Bar chart comparison
    fig = go.Figure()
    
    for metric in metrics:
        values = [performance[model].get(metric, 0) for model in models]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=models,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title=f"Model Performance Comparison - {selected_dataset.replace('_', ' ').title()}",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model
    best_model = st.session_state.model_manager.get_best_model(selected_dataset, "auc")
    if best_model:
        st.success(f"🏆 Best performing model: **{best_model.replace('_', ' ').title()}**")

def show_explainability_page():
    """Display model explainability page"""
    st.header("🔍 Model Explainability")
    
    available_datasets = st.session_state.model_manager.list_available_datasets()
    
    if not available_datasets:
        st.error("❌ No trained models available. Please train models first.")
        return
    
    # Dataset and model selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_dataset = st.selectbox(
            "Select Dataset:",
            available_datasets,
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        model_type = st.selectbox(
            "Select Model:",
            ["ensemble", "random_forest", "xgboost", "svm"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    st.markdown("---")
    
    # Load model
    models = st.session_state.model_manager.load_models(selected_dataset)
    
    if model_type not in models:
        st.error(f"Model {model_type} not found for dataset {selected_dataset}")
        return
    
    model = models[model_type]
    
    # Explainability options
    st.subheader("🔍 Explainability Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["📊 SHAP Summary Plot", "💧 SHAP Waterfall Plot", "📈 Feature Importance", "🎯 Individual Prediction Explanation"]
    )
    
    if analysis_type == "📊 SHAP Summary Plot":
        st.info("SHAP Summary Plot shows the distribution of feature impacts on model predictions.")
        # This would require loading test data and calculating SHAP values
        st.warning("SHAP analysis requires additional computation. Please use the training page to generate SHAP plots.")
    
    elif analysis_type == "💧 SHAP Waterfall Plot":
        st.info("SHAP Waterfall Plot shows how each feature contributes to a specific prediction.")
        st.warning("SHAP analysis requires additional computation. Please use the training page to generate SHAP plots.")
    
    elif analysis_type == "📈 Feature Importance":
        if hasattr(model, 'feature_importances_'):
            # Get feature names
            feature_names = get_feature_descriptions(selected_dataset).keys()
            
            # Create feature importance plot
            importance_df = pd.DataFrame({
                'Feature': list(feature_names),
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Feature Importance - {model_type.replace('_', ' ').title()}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available for this model type.")
    
    elif analysis_type == "🎯 Individual Prediction Explanation":
        st.info("Explain a specific prediction using SHAP values.")
        
        # Create sample input
        sample_df = create_sample_data(selected_dataset)
        st.write("Sample input data:")
        st.dataframe(sample_df)
        
        if st.button("🔍 Explain Prediction"):
            try:
                processed_sample = preprocess_input_data(sample_df, selected_dataset)
                explanation = explain_single_prediction(model, processed_sample, model_type)
                
                if 'error' not in explanation:
                    st.subheader("🎯 Prediction Explanation")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Prediction", "Positive" if explanation['prediction'] == 1 else "Negative")
                        if explanation['probability']:
                            st.metric("Probability", f"{explanation['probability']:.1%}")
                    
                    with col2:
                        st.write("**Feature Contributions:**")
                        for feature, contrib in list(explanation['feature_contributions'].items())[:5]:
                            color = "red" if contrib['shap_value'] > 0 else "green"
                            st.markdown(f"- **{feature}**: <span style='color:{color}'>{contrib['shap_value']:.3f}</span>", unsafe_allow_html=True)
                else:
                    st.error(f"Explanation error: {explanation['error']}")
                    
            except Exception as e:
                st.error(f"Error explaining prediction: {str(e)}")

def show_data_management_page():
    """Display data management page"""
    st.header("📁 Data Management")
    
    st.subheader("📊 Dataset Information")
    
    datasets_info = {
        "heart_disease": {
            "name": "Heart Disease Dataset",
            "source": "UCI Machine Learning Repository",
            "features": 13,
            "samples": "~300",
            "description": "Clinical parameters for heart disease prediction"
        },
        "diabetes": {
            "name": "Diabetes Dataset", 
            "source": "Pima Indians Diabetes Database",
            "features": 8,
            "samples": "~768",
            "description": "Medical parameters for diabetes prediction"
        },
        "liver_disease": {
            "name": "Liver Disease Dataset",
            "source": "Indian Liver Patient Dataset", 
            "features": 10,
            "samples": "~583",
            "description": "Clinical parameters for liver disease prediction"
        }
    }
    
    for dataset_key, info in datasets_info.items():
        with st.expander(f"📋 {info['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Source:** {info['source']}")
                st.write(f"**Features:** {info['features']}")
                st.write(f"**Samples:** {info['samples']}")
            
            with col2:
                st.write(f"**Description:** {info['description']}")
                
                # Check if model exists
                available_datasets = st.session_state.model_manager.list_available_datasets()
                if dataset_key in available_datasets:
                    st.success("✅ Model trained")
                else:
                    st.warning("⚠️ Model not trained")
    
    st.markdown("---")
    
    st.subheader("🔄 Data Processing Pipeline")
    
    st.markdown("""
    ### 📋 Processing Steps:
    
    1. **📥 Data Download** - Fetch from UCI repository
    2. **🧹 Data Cleaning** - Handle missing values and duplicates
    3. **🔢 Feature Encoding** - Convert categorical variables
    4. **⚖️ Feature Scaling** - Normalize numerical features
    5. **🎯 Feature Selection** - Select most important features
    6. **⚖️ Class Balancing** - Apply SMOTE if needed
    7. **✂️ Train-Test Split** - Split data for evaluation
    """)
    
    st.markdown("---")
    
    st.subheader("📈 Data Statistics")
    
    # Show statistics for available datasets
    available_datasets = st.session_state.model_manager.list_available_datasets()
    
    if available_datasets:
        for dataset in available_datasets:
            try:
                # Load processed data
                X_train, y_train, X_test, y_test = load_processed_data(dataset)
                
                with st.expander(f"📊 {dataset.replace('_', ' ').title()} Statistics"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Training Samples", len(X_train))
                        st.metric("Test Samples", len(X_test))
                    
                    with col2:
                        st.metric("Features", X_train.shape[1])
                        st.metric("Positive Class %", f"{y_train.mean():.1%}")
                    
                    with col3:
                        st.write("**Feature Names:**")
                        for feature in X_train.columns[:5]:  # Show first 5
                            st.write(f"• {feature}")
                        if len(X_train.columns) > 5:
                            st.write(f"• ... and {len(X_train.columns) - 5} more")
                            
            except Exception as e:
                st.error(f"Error loading statistics for {dataset}: {str(e)}")

if __name__ == "__main__":
    main() 