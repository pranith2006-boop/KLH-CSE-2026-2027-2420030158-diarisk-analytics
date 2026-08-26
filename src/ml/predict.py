import os
import pandas as pd
import numpy as np
import joblib

def get_model_pipeline(dataset_name, model_name):
    """
    Loads a saved ML pipeline (which contains both ColumnTransformer and estimator).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'models', dataset_name, f"{model_name}.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model pipeline not found at {model_path}")
        
    return joblib.load(model_path)

def predict_single_dataset1(input_dict, model_name='xgboost'):
    """
    Generates prediction for a single record using a Dataset 1 model.
    """
    pipeline = get_model_pipeline('dataset1', model_name)
    
    # Convert input dict to DataFrame (single row)
    df_input = pd.DataFrame([input_dict])
    
    # Predict
    pred = pipeline.predict(df_input)[0]
    prob = pipeline.predict_proba(df_input)[0]
    
    # Class map
    class_map = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
    
    return {
        'predicted_class_code': int(pred),
        'predicted_label': class_map.get(int(pred), "Unknown"),
        'class_probabilities': {
            'No Diabetes': float(prob[0]),
            'Prediabetes': float(prob[1]),
            'Diabetes': float(prob[2])
        }
    }

def predict_single_dataset2(input_dict, model_name='xgboost'):
    """
    Generates prediction for a single record using a Dataset 2 model.
    """
    pipeline = get_model_pipeline('dataset2', model_name)
    
    # Convert input dict to DataFrame (single row)
    df_input = pd.DataFrame([input_dict])
    
    # Predict
    pred = pipeline.predict(df_input)[0]
    prob = pipeline.predict_proba(df_input)[0]
    
    return {
        'predicted_class_code': int(pred),
        'predicted_label': "Diabetes Diagnosed" if int(pred) == 1 else "No Diabetes",
        'class_probabilities': {
            'No Diabetes': float(prob[0]),
            'Diabetes': float(prob[1])
        }
    }

if __name__ == '__main__':
    # Test cases to verify the prediction function
    print("Testing ML Predict Module...")
    
    # Mock input for Dataset 1
    mock_input_1 = {
        'HighBP': 1.0, 'HighChol': 1.0, 'CholCheck': 1.0, 'BMI': 32.0, 
        'Smoker': 0.0, 'Stroke': 0.0, 'HeartDiseaseorAttack': 1.0, 
        'PhysActivity': 0.0, 'Fruits': 1.0, 'Veggies': 1.0, 'HvyAlcoholConsump': 0.0, 
        'AnyHealthcare': 1.0, 'NoDocbcCost': 0.0, 'GenHlth': 4.0, 'MentHlth': 10.0, 
        'PhysHlth': 15.0, 'DiffWalk': 1.0, 'Sex': 1.0, 'Age': 9.0, 
        'Education': 4.0, 'Income': 5.0
    }
    
    # Mock input for Dataset 2
    mock_input_2 = {
        'Age': 58, 'gender': 'Male', 'ethnicity': 'White', 'education_level': 'Bachelor\'s Degree',
        'income_level': 'High', 'employment_status': 'Employed', 'smoking_status': 'Never',
        'alcohol_consumption_per_week': 2, 'physical_activity_minutes_per_week': 120, 'diet_score': 6.0,
        'sleep_hours_per_day': 7.0, 'screen_time_hours_per_day': 5.0, 'family_history_diabetes': 1,
        'hypertension_history': 1, 'cardiovascular_history': 0, 'bmi': 28.5, 'waist_to_hip_ratio': 0.88,
        'systolic_bp': 130, 'diastolic_bp': 85, 'heart_rate': 72, 'cholesterol_total': 210,
        'hdl_cholesterol': 50, 'ldl_cholesterol': 130, 'triglycerides': 150, 'glucose_fasting': 125,
        'glucose_postprandial': 180, 'insulin_level': 12.5, 'hba1c': 6.6
    }
    
    # We will test running predictions after models are trained in execution steps.
    print("Mock inputs defined. Run train.py first to enable tests.")
