import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def get_column_lists(target_col='diagnosed_diabetes'):
    """
    Returns lists of numerical and categorical feature columns for Dataset 2,
    excluding target candidates to prevent data leakage.
    """
    target_candidates = ['diagnosed_diabetes', 'diabetes_stage', 'diabetes_risk_score']
    
    numerical_cols = [
        'Age', 'physical_activity_minutes_per_week', 'diet_score', 
        'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 
        'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
        'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 
        'triglycerides', 'glucose_fasting', 'glucose_postprandial', 
        'insulin_level', 'hba1c'
    ]
    
    categorical_cols = [
        'gender', 'ethnicity', 'education_level', 'income_level', 
        'employment_status', 'smoking_status', 'alcohol_consumption_per_week', 
        'family_history_diabetes', 'hypertension_history', 'cardiovascular_history'
    ]
    
    # Filter columns to make sure they exist in the feature set (excluding all targets)
    numerical_cols = [c for c in numerical_cols if c not in target_candidates and c != target_col]
    categorical_cols = [c for c in categorical_cols if c not in target_candidates and c != target_col]
    
    return numerical_cols, categorical_cols

def build_preprocessor(target_col='diagnosed_diabetes'):
    """
    Builds and returns the ColumnTransformer for Dataset 2 based on the selected target.
    """
    num_cols, cat_cols = get_column_lists(target_col)
    
    # Numerical pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    
    return preprocessor

def handle_duplicates(df):
    """
    Identifies and removes exact duplicates from Dataset 2.
    """
    dup_before = df.duplicated().sum()
    df_clean = df.drop_duplicates().reset_index(drop=True)
    dup_after = df_clean.duplicated().sum()
    return df_clean, dup_before, dup_after

def detect_outliers_iqr(df, target_col='diagnosed_diabetes'):
    """
    Detects outliers in numerical continuous features.
    """
    num_cols, _ = get_column_lists(target_col)
    outlier_counts = {}
    total_rows = len(df)
    
    for col in num_cols:
        # Check if column is numeric in the dataframe
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts[col] = {
                'count': int(count),
                'percentage': float(count / total_rows * 100)
            }
    return outlier_counts
