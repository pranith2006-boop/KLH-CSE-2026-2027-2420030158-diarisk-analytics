import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Add preprocessing directory to Python path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'src', 'preprocessing'))

import dataset1_preprocessing
import dataset2_preprocessing

def train_and_save_dataset1(data_path, models_dir):
    """
    Trains and saves all 6 models for Dataset 1.
    """
    print("\n" + "="*50)
    print("TRAINING MODELS FOR DATASET 1 (BRFSS 2015)")
    print("="*50)
    
    # Load raw data
    df = pd.read_csv(data_path)
    
    # Clean duplicates (same as Phase 2)
    df_clean, _, _ = dataset1_preprocessing.handle_duplicates(df)
    
    target_col = 'Diabetes_012'
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Imbalance parameters for Dataset 1
    # Target distribution ratios
    class_counts = y_train.value_counts()
    print("Train Class distribution:")
    for k, v in class_counts.items():
        print(f"  Class {k}: {v} ({v/len(y_train)*100:.2f}%)")
        
    # Calculate scale_pos_weight for XGBoost (approximated for multi-class or binary subset)
    # For multiclass, scale_pos_weight is not directly supported, but we will define standard baselines.
    # For CatBoost, we will use auto_class_weights='Balanced'.
    # For SVC/RandomForest/LogisticRegression, we use class_weight='balanced'.
    
    # Preprocessor
    preprocessor = dataset1_preprocessing.build_preprocessor()
    
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced', n_jobs=-1
        ),
        'decision_tree': DecisionTreeClassifier(
            random_state=42, max_depth=8, class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            random_state=42, n_estimators=100, max_depth=8, class_weight='balanced', n_jobs=-1
        ),
        'svm': SVC(
            probability=True, random_state=42, class_weight='balanced', max_iter=5000 # Limit iterations to prevent hanging
        ),
        'xgboost': XGBClassifier(
            random_state=42, eval_metric='mlogloss', n_jobs=-1
        ),
        'catboost': CatBoostClassifier(
            random_state=42, verbose=0, thread_count=-1, auto_class_weights='Balanced'
        )
    }
    
    # SVM on large datasets is extremely slow. We will train it on a stratified subsample of 12,000 rows
    # to ensure the execution completes in a reasonable time.
    sub_size = 12000
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=sub_size, stratify=y_train, random_state=42
    )
    
    # Train each model and save the pipeline
    os.makedirs(os.path.join(models_dir, 'dataset1'), exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train on appropriate data
        if name == 'svm':
            print(f"  (Training SVM on a stratified subset of {sub_size} records to prevent hanging)")
            pipeline.fit(X_train_sub, y_train_sub)
        else:
            pipeline.fit(X_train, y_train)
            
        # Save pipeline
        model_path = os.path.join(models_dir, 'dataset1', f"{name}.joblib")
        joblib.dump(pipeline, model_path)
        print(f"  Saved to {model_path}")

def train_and_save_dataset2(data_path, models_dir):
    """
    Trains and saves all 6 models for Dataset 2.
    """
    print("\n" + "="*50)
    print("TRAINING MODELS FOR DATASET 2 (CLINICAL & LIFESTYLE)")
    print("="*50)
    
    # Load raw data
    df = pd.read_csv(data_path)
    
    # Clean duplicates (same as Phase 2)
    df_clean, _, _ = dataset2_preprocessing.handle_duplicates(df)
    
    target_col = 'diagnosed_diabetes'
    other_targets = ['diabetes_stage', 'diabetes_risk_score']
    
    # Separate features and target, dropping other target candidates to prevent leakage
    X = df_clean.drop(columns=[target_col] + other_targets)
    y = df_clean[target_col].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Preprocessor
    preprocessor = dataset2_preprocessing.build_preprocessor(target_col)
    
    # Dataset 2 is relatively balanced (60% pos, 40% neg), so standard class weighting is not strictly necessary,
    # but we can configure models normally.
    # Calculate scale_pos_weight for XGBoost: negative_count / positive_count = 40 / 60 = 0.67
    scale_pos = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, max_iter=1000, n_jobs=-1
        ),
        'decision_tree': DecisionTreeClassifier(
            random_state=42, max_depth=8
        ),
        'random_forest': RandomForestClassifier(
            random_state=42, n_estimators=100, max_depth=10, n_jobs=-1
        ),
        'svm': SVC(
            probability=True, random_state=42, max_iter=5000 # Limit iterations to prevent hanging
        ),
        'xgboost': XGBClassifier(
            random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos, n_jobs=-1
        ),
        'catboost': CatBoostClassifier(
            random_state=42, verbose=0, thread_count=-1
        )
    }
    
    # SVM on 77k rows with probability=True is slow. We will train it on a stratified subsample of 15,000 rows
    sub_size = 15000
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=sub_size, stratify=y_train, random_state=42
    )
    
    # Train each model and save the pipeline
    os.makedirs(os.path.join(models_dir, 'dataset2'), exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train on appropriate data
        if name == 'svm':
            print(f"  (Training SVM on a stratified subset of {sub_size} records to prevent hanging)")
            pipeline.fit(X_train_sub, y_train_sub)
        else:
            pipeline.fit(X_train, y_train)
            
        # Save pipeline
        model_path = os.path.join(models_dir, 'dataset2', f"{name}.joblib")
        joblib.dump(pipeline, model_path)
        print(f"  Saved to {model_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    
    dataset1_path = os.path.join(data_dir, 'diabetes_012_health_indicators_BRFSS2015.csv')
    dataset2_path = os.path.join(data_dir, 'Diabetes_and_LifeStyle_Dataset .csv')
    
    if os.path.exists(dataset1_path):
        train_and_save_dataset1(dataset1_path, models_dir)
    else:
        print(f"Error: Dataset 1 not found at {dataset1_path}")
        
    if os.path.exists(dataset2_path):
        train_and_save_dataset2(dataset2_path, models_dir)
    else:
        print(f"Error: Dataset 2 not found at {dataset2_path}")
        
    print("\n" + "="*50)
    print("ALL ML MODELS TRAINED AND SAVED SUCCESSFULLY")
    print("="*50)
