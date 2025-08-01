"""
Utility functions for Streamlit app
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any

def display_prediction_result(result: Dict[str, Any], dataset_name: str):
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

def display_batch_prediction_result(result: Dict[str, Any], original_df: pd.DataFrame):
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

def create_input_form(dataset_name: str) -> Dict[str, Any]:
    """Create input form for manual prediction"""
    input_data = {}
    
    if dataset_name == "heart_disease":
        col1, col2 = st.columns(2)
        
        with col1:
            input_data['age'] = st.number_input("Age", min_value=20, max_value=100, value=55)
            input_data['sex'] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            input_data['cp'] = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                                          help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
            input_data['trestbps'] = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=130)
            input_data['chol'] = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=250)
            input_data['fbs'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            input_data['restecg'] = st.selectbox("Resting ECG Results", [0, 1, 2], 
                                               help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
        
        with col2:
            input_data['thalach'] = st.number_input("Max Heart Rate", min_value=60, max_value=200, value=150)
            input_data['exang'] = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            input_data['oldpeak'] = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
            input_data['slope'] = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2], 
                                             help="0: Upsloping, 1: Flat, 2: Downsloping")
            input_data['ca'] = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
            input_data['thal'] = st.selectbox("Thalassemia", [0, 1, 2, 3], 
                                            help="0: Normal, 1: Fixed defect, 2: Reversable defect, 3: Other")
    
    elif dataset_name == "diabetes":
        col1, col2 = st.columns(2)
        
        with col1:
            input_data['pregnancies'] = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=2)
            input_data['glucose'] = st.number_input("Glucose (mg/dl)", min_value=50, max_value=300, value=120)
            input_data['blood_pressure'] = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=150, value=80)
            input_data['skin_thickness'] = st.number_input("Skin Thickness (mm)", min_value=10, max_value=100, value=25)
        
        with col2:
            input_data['insulin'] = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=100)
            input_data['bmi'] = st.number_input("BMI", min_value=15.0, max_value=70.0, value=28.5, step=0.1)
            input_data['diabetes_pedigree'] = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
            input_data['age'] = st.number_input("Age", min_value=20, max_value=100, value=45)
    
    elif dataset_name == "liver_disease":
        col1, col2 = st.columns(2)
        
        with col1:
            input_data['age'] = st.number_input("Age", min_value=10, max_value=100, value=45)
            input_data['gender'] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            input_data['total_bilirubin'] = st.number_input("Total Bilirubin (mg/dl)", min_value=0.0, max_value=30.0, value=1.2, step=0.1)
            input_data['direct_bilirubin'] = st.number_input("Direct Bilirubin (mg/dl)", min_value=0.0, max_value=20.0, value=0.3, step=0.1)
            input_data['alkaline_phosphotase'] = st.number_input("Alkaline Phosphotase (IU/L)", min_value=50, max_value=500, value=120)
        
        with col2:
            input_data['alamine_aminotransferase'] = st.number_input("Alamine Aminotransferase (IU/L)", min_value=10, max_value=200, value=25)
            input_data['aspartate_aminotransferase'] = st.number_input("Aspartate Aminotransferase (IU/L)", min_value=10, max_value=300, value=30)
            input_data['total_proteins'] = st.number_input("Total Proteins (g/dl)", min_value=2.0, max_value=10.0, value=7.0, step=0.1)
            input_data['albumin'] = st.number_input("Albumin (g/dl)", min_value=1.0, max_value=6.0, value=4.0, step=0.1)
            input_data['albumin_and_globulin_ratio'] = st.number_input("Albumin/Globulin Ratio", min_value=0.5, max_value=3.0, value=1.2, step=0.1)
    
    return input_data 