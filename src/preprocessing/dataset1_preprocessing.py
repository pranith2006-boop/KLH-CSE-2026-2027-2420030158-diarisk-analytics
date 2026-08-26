import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def get_column_lists():
    """
    Returns lists of numerical and categorical feature columns for Dataset 1.
    """
    numerical_cols = ['BMI', 'MentHlth', 'PhysHlth']
    
    categorical_cols = [
        'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
        'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    
    return numerical_cols, categorical_cols

def build_preprocessor():
    """
    Builds and returns the ColumnTransformer for Dataset 1.
    """
    num_cols, cat_cols = get_column_lists()
    
    # Numerical pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline (binary indicators and ordinal codes treated as nominal categories)
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
    Identifies and removes exact duplicates from Dataset 1.
    """
    dup_before = df.duplicated().sum()
    df_clean = df.drop_duplicates().reset_index(drop=True)
    dup_after = df_clean.duplicated().sum()
    return df_clean, dup_before, dup_after

def detect_outliers_iqr(df):
    """
    Detects outliers in numerical continuous features.
    """
    num_cols, _ = get_column_lists()
    outlier_counts = {}
    total_rows = len(df)
    
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Outliers count
        count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_counts[col] = {
            'count': int(count),
            'percentage': float(count / total_rows * 100)
        }
    return outlier_counts
