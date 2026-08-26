import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import dataset-specific preprocessing functions
import dataset1_preprocessing
import dataset2_preprocessing

def preprocess_dataset1(data_path, output_dir):
    """
    Orchestrates the preprocessing pipeline for Dataset 1.
    """
    print("--------------------------------------------------")
    print("Processing Dataset 1: BRFSS 2015 Health Indicators")
    print("--------------------------------------------------")
    
    # Load dataset
    df = pd.read_csv(data_path)
    orig_rows, orig_cols = df.shape
    
    # Check for missing values
    missing_count = int(df.isnull().sum().sum())
    
    # Handle duplicates
    df_clean, dup_before, dup_after = dataset1_preprocessing.handle_duplicates(df)
    removed_duplicates = int(dup_before)
    
    # Outliers detection (on clean full data for reporting)
    outliers = dataset1_preprocessing.detect_outliers_iqr(df_clean)
    outliers_str = ", ".join([f"{col}: {info['count']} ({info['percentage']:.2f}%)" for col, info in outliers.items()])
    
    # Target column (identified in Phase 1)
    target_col = 'Diabetes_012'
    
    # Class distribution and imbalance
    class_dist = df_clean[target_col].value_counts().to_dict()
    class_dist_str = ", ".join([f"Class {int(k)}: {v} ({v/len(df_clean)*100:.2f}%)" for k, v in class_dist.items()])
    
    # Separate features and target
    X = df_clean.drop(columns=[target_col])
    # Target is float in CSV, cast to integer
    y = df_clean[target_col].astype(int)
    
    # Train/Test Split (80% train, 20% test, stratified by target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Fit preprocessing pipeline on training data
    preprocessor = dataset1_preprocessing.build_preprocessor()
    preprocessor.fit(X_train)
    
    # Get processed column names
    num_cols, cat_cols = dataset1_preprocessing.get_column_lists()
    
    # OneHotEncoder feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    # Extract feature names safely across different scikit-learn versions
    if hasattr(cat_encoder, 'get_feature_names_out'):
        cat_features_out = cat_encoder.get_feature_names_out(cat_cols)
    else:
        cat_features_out = cat_encoder.get_feature_names(cat_cols)
        
    processed_feature_names = list(num_cols) + list(cat_features_out)
    
    # Transform train and test features
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to DataFrames for saving
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)
    
    # Save processed splits
    os.makedirs(output_dir, exist_ok=True)
    X_train_processed_df.to_csv(os.path.join(output_dir, "dataset1_X_train.csv"), index=False)
    X_test_processed_df.to_csv(os.path.join(output_dir, "dataset1_X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "dataset1_y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "dataset1_y_test.csv"), index=False)
    
    # Create preprocessing summary
    summary_data = {
        'Metric': [
            'Original Records', 'Original Features', 'Missing Values', 
            'Duplicate Records Before', 'Duplicate Records After', 'Removed Duplicates',
            'Numerical Columns', 'Categorical Columns', 'Target Column', 
            'Class Distribution', 'Train Records', 'Test Records', 
            'Processed Features Count', 'Scaling Method', 'Encoding Method'
        ],
        'Value': [
            orig_rows, orig_cols, missing_count, 
            dup_before, dup_after, removed_duplicates,
            len(num_cols), len(cat_cols), target_col, 
            class_dist_str, len(X_train), len(X_test), 
            len(processed_feature_names), 'StandardScaler', 'OneHotEncoder'
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "dataset1_preprocessing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Print console summary
    print(f"Original Records:    {orig_rows:,}")
    print(f"Original Features:   {orig_cols}")
    print(f"Missing Values:      {missing_count}")
    print(f"Duplicate Records:   {dup_before:,} (Removed)")
    print(f"Outliers Detected:   {outliers_str}")
    print(f"Final Records:       {len(df_clean):,}")
    print(f"Final Features:      {orig_cols - 1}")
    print(f"Target:              {target_col}")
    print(f"Class Distribution:  {class_dist_str}")
    print(f"Train/Test split:    Train={len(X_train):,}, Test={len(X_test):,}")
    print(f"Final Shape (X_train processed): {X_train_processed.shape}")
    print(f"Saved summary to:    {summary_path}\n")
    
    return summary_df

def preprocess_dataset2(data_path, output_dir):
    """
    Orchestrates the preprocessing pipeline for Dataset 2.
    """
    print("--------------------------------------------------")
    print("Processing Dataset 2: Lifestyle & Clinical Dataset")
    print("--------------------------------------------------")
    
    # Load dataset
    df = pd.read_csv(data_path)
    orig_rows, orig_cols = df.shape
    
    # Check for missing values
    missing_count = int(df.isnull().sum().sum())
    
    # Handle duplicates
    df_clean, dup_before, dup_after = dataset2_preprocessing.handle_duplicates(df)
    removed_duplicates = int(dup_before)
    
    # Primary classification target variable (diagnosed_diabetes)
    target_col = 'diagnosed_diabetes'
    other_targets = ['diabetes_stage', 'diabetes_risk_score']
    
    # Outliers detection (on clean full data for reporting)
    outliers = dataset2_preprocessing.detect_outliers_iqr(df_clean, target_col)
    outliers_str = ", ".join([f"{col}: {info['count']} ({info['percentage']:.2f}%)" for col, info in list(outliers.items())[:5]]) + " (showing first 5)"
    
    # Class distribution and imbalance
    class_dist = df_clean[target_col].value_counts().to_dict()
    class_dist_str = ", ".join([f"Class {int(k)}: {v} ({v/len(df_clean)*100:.2f}%)" for k, v in class_dist.items()])
    
    # Separate features and target, dropping other target candidates to prevent data leakage
    X = df_clean.drop(columns=[target_col] + other_targets)
    y = df_clean[target_col].astype(int)
    
    # Train/Test Split (80% train, 20% test, stratified by target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Fit preprocessing pipeline on training data
    preprocessor = dataset2_preprocessing.build_preprocessor(target_col)
    preprocessor.fit(X_train)
    
    # Get processed column names
    num_cols, cat_cols = dataset2_preprocessing.get_column_lists(target_col)
    
    # OneHotEncoder feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    if hasattr(cat_encoder, 'get_feature_names_out'):
        cat_features_out = cat_encoder.get_feature_names_out(cat_cols)
    else:
        cat_features_out = cat_encoder.get_feature_names(cat_cols)
        
    processed_feature_names = list(num_cols) + list(cat_features_out)
    
    # Transform train and test features
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to DataFrames for saving
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)
    
    # Save processed splits
    os.makedirs(output_dir, exist_ok=True)
    X_train_processed_df.to_csv(os.path.join(output_dir, "dataset2_X_train.csv"), index=False)
    X_test_processed_df.to_csv(os.path.join(output_dir, "dataset2_X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "dataset2_y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "dataset2_y_test.csv"), index=False)
    
    # Create preprocessing summary
    summary_data = {
        'Metric': [
            'Original Records', 'Original Features', 'Missing Values', 
            'Duplicate Records Before', 'Duplicate Records After', 'Removed Duplicates',
            'Numerical Columns', 'Categorical Columns', 'Target Column', 
            'Class Distribution', 'Train Records', 'Test Records', 
            'Processed Features Count', 'Scaling Method', 'Encoding Method',
            'Dropped Target Leakage Columns'
        ],
        'Value': [
            orig_rows, orig_cols, missing_count, 
            dup_before, dup_after, removed_duplicates,
            len(num_cols), len(cat_cols), target_col, 
            class_dist_str, len(X_train), len(X_test), 
            len(processed_feature_names), 'StandardScaler', 'OneHotEncoder',
            ", ".join(other_targets)
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "dataset2_preprocessing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Print console summary
    print(f"Original Records:    {orig_rows:,}")
    print(f"Original Features:   {orig_cols}")
    print(f"Missing Values:      {missing_count}")
    print(f"Duplicate Records:   {dup_before:,} (Removed)")
    print(f"Outliers Detected:   {outliers_str}")
    print(f"Final Records:       {len(df_clean):,}")
    print(f"Final Features:      {len(X.columns)}")
    print(f"Target:              {target_col}")
    print(f"Class Distribution:  {class_dist_str}")
    print(f"Train/Test split:    Train={len(X_train):,}, Test={len(X_test):,}")
    print(f"Final Shape (X_train processed): {X_train_processed.shape}")
    print(f"Saved summary to:    {summary_path}\n")
    
    return summary_df

if __name__ == '__main__':
    # Determine base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "results", "processed_data")
    
    # Dataset paths
    dataset1_path = os.path.join(data_dir, "diabetes_012_health_indicators_BRFSS2015.csv")
    dataset2_path = os.path.join(data_dir, "Diabetes_and_LifeStyle_Dataset .csv")
    
    print("==================================================")
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("==================================================\n")
    
    # Process Dataset 1
    if os.path.exists(dataset1_path):
        preprocess_dataset1(dataset1_path, output_dir)
    else:
        print(f"Error: Dataset 1 not found at {dataset1_path}")
        
    # Process Dataset 2
    if os.path.exists(dataset2_path):
        preprocess_dataset2(dataset2_path, output_dir)
    else:
        print(f"Error: Dataset 2 not found at {dataset2_path}")
        
    print("==================================================")
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("==================================================")
