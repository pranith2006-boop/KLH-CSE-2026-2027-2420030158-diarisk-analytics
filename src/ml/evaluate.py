import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import plotly.express as px
import plotly.graph_objects as go

# Add preprocessing directory to Python path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, 'src', 'preprocessing'))

import dataset1_preprocessing
import dataset2_preprocessing

MODEL_NAMES = ['logistic_regression', 'decision_tree', 'random_forest', 'svm', 'xgboost', 'catboost']
METRIC_COLUMNS = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']


def _feature_names(preprocessor):
    """Return output feature names from a fitted preprocessing transformer."""
    return list(preprocessor.get_feature_names_out())


def _save_metric_charts(results_df, output_dir, dataset_name):
    chart_dir = os.path.join(output_dir, 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    for metric in METRIC_COLUMNS:
        figure = px.bar(
            results_df, x='Model', y=metric, text_auto='.4f',
            title=f'{metric} Comparison - {dataset_name.title()}',
            labels={metric: metric, 'Model': 'Model'}
        )
        figure.update_yaxes(range=[0, 1])
        figure.write_html(os.path.join(chart_dir, f'{dataset_name}_{metric.lower().replace("-", "_")}.html'))


def _save_ranking(results_df, output_dir, dataset_name):
    ranking = results_df.sort_values(
        ['ROC-AUC', 'Recall', 'F1', 'Accuracy'], ascending=False
    ).reset_index(drop=True)
    ranking.insert(0, 'Rank', range(1, len(ranking) + 1))
    ranking = ranking[['Rank', 'Model', 'ROC-AUC', 'Recall', 'F1', 'Accuracy']]
    ranking.to_csv(os.path.join(output_dir, f'{dataset_name}_model_ranking.csv'), index=False, float_format='%.4f')
    figure = px.bar(
        ranking, x='Model', y='ROC-AUC', color='Rank', text='Rank',
        title=f'Model Ranking - {dataset_name.title()}',
        labels={'ROC-AUC': 'ROC-AUC', 'Model': 'Model'}
    )
    figure.write_html(os.path.join(output_dir, f'{dataset_name}_model_ranking.html'))
    return ranking


def _save_feature_chart(output_dir, dataset_name):
    feature_dir = os.path.join(output_dir, dataset_name, 'feature_importance')
    chart_dir = os.path.join(output_dir, 'charts')
    os.makedirs(chart_dir, exist_ok=True)
    frames = []
    for model_name in MODEL_NAMES:
        path = os.path.join(feature_dir, f'{model_name}_feature_importance.csv')
        if os.path.exists(path):
            frame = pd.read_csv(path).head(20)
            frame['Model'] = model_name
            frames.append(frame)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        figure = px.bar(
            combined.sort_values('Importance'), x='Importance', y='Feature',
            color='Model', facet_col='Model', facet_col_wrap=2,
            title=f'Top Feature Importance - {dataset_name.title()}', orientation='h'
        )
        figure.write_html(os.path.join(chart_dir, f'{dataset_name}_feature_importance.html'))


def _evaluate_saved_dataset(data_path, models_dir, output_dir, dataset_name):
    if dataset_name == 'dataset1':
        preprocessing = dataset1_preprocessing
        target_col = 'Diabetes_012'
        df, _, _ = preprocessing.handle_duplicates(pd.read_csv(data_path))
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)
        cv_metric = 'roc_auc_ovr'
    else:
        preprocessing = dataset2_preprocessing
        target_col = 'diagnosed_diabetes'
        df, _, _ = preprocessing.handle_duplicates(pd.read_csv(data_path))
        X = df.drop(columns=[target_col, 'diabetes_stage', 'diabetes_risk_score'])
        y = df[target_col].astype(int)
        cv_metric = 'roc_auc'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    labels = sorted(y.unique().tolist())
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(os.path.join(dataset_dir, 'confusion_matrices'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'roc_curves'), exist_ok=True)

    results = []
    roc_figure = go.Figure()
    predictions_by_model = {}
    for model_name in MODEL_NAMES:
        model_path = os.path.join(models_dir, dataset_name, f'{model_name}.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Missing trained model: {model_path}')
        pipeline = joblib.load(model_path)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        if len(labels) == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            positive_mask = (y_test == labels[-1]).astype(int)
            fpr, tpr, _ = roc_curve(positive_mask, y_prob[:, -1])

        results.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'ROC-AUC': roc_auc
        })
        roc_figure.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC={roc_auc:.4f})'))

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f'Actual {label}' for label in labels], columns=[f'Predicted {label}' for label in labels])
        cm_df.to_csv(os.path.join(dataset_dir, 'confusion_matrices', f'{model_name}_confusion_matrix.csv'))
        report = classification_report(y_test, y_pred, labels=labels, target_names=[str(label) for label in labels], output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(os.path.join(dataset_dir, 'confusion_matrices', f'{model_name}_classification_report.csv'))
        predictions_by_model[model_name] = (y_pred, y_prob)

    results_df = pd.DataFrame(results)
    cv_path = os.path.join(output_dir, f'{dataset_name}_model_comparison.csv')
    old_results = pd.read_csv(cv_path) if os.path.exists(cv_path) else pd.DataFrame()
    if not old_results.empty and {'Model', 'CV Mean', 'CV Std'}.issubset(old_results.columns):
        results_df = results_df.merge(old_results[['Model', 'CV Mean', 'CV Std']], on='Model', how='left')
    else:
        results_df['CV Mean'] = np.nan
        results_df['CV Std'] = np.nan
    results_df.to_csv(cv_path, index=False, float_format='%.4f')
    results_df.to_csv(os.path.join(dataset_dir, 'model_comparison.csv'), index=False, float_format='%.4f')
    results_df[['Model', 'CV Mean', 'CV Std']].to_csv(os.path.join(output_dir, f'{dataset_name}_cv_results.csv'), index=False, float_format='%.4f')

    ranking = _save_ranking(results_df, output_dir, dataset_name)
    best = ranking.iloc[0]
    second = ranking.iloc[1]
    best_metrics = results_df[results_df['Model'] == best['Model']].iloc[0]
    best_y_pred, _ = predictions_by_model[best['Model']]
    roc_figure.update_layout(title=f'ROC Curves - {dataset_name.title()}', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    roc_figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guessing', line=dict(dash='dash')))
    roc_figure.write_html(os.path.join(dataset_dir, 'roc_curves', 'model_roc_comparison.html'))
    _save_metric_charts(results_df, output_dir, dataset_name)
    _save_feature_chart(output_dir, dataset_name)

    positive_label = labels[-1]
    binary_actual = (y_test == positive_label).astype(int)
    binary_pred = (best_y_pred == positive_label).astype(int)
    tn, fp, fn, tp = confusion_matrix(binary_actual, binary_pred, labels=[0, 1]).ravel()
    total_negative = tn + fp
    total_positive = fn + tp
    error_analysis = {
        'positive_label_for_error_analysis': positive_label,
        'true_negative': int(tn), 'false_positive': int(fp),
        'false_negative': int(fn), 'true_positive': int(tp),
        'false_positive_rate': float(fp / total_negative) if total_negative else None,
        'false_negative_rate': float(fn / total_positive) if total_positive else None
    }
    with open(os.path.join(output_dir, f'{dataset_name}_error_analysis.json'), 'w', encoding='utf-8') as file:
        json.dump(error_analysis, file, indent=2)
    with open(os.path.join(output_dir, f'{dataset_name}_best_model.json'), 'w', encoding='utf-8') as file:
        json.dump({
            'dataset': dataset_name, 'best_model': best['Model'],
            'model_path': os.path.relpath(os.path.join(models_dir, dataset_name, f"{best['Model']}.joblib"), base_dir),
            'preprocessing_path': 'embedded in model pipeline',
            'feature_names': _feature_names(joblib.load(os.path.join(models_dir, dataset_name, f"{best['Model']}.joblib")) .named_steps['preprocessor']),
            'target_name': target_col, 'selection_rule': 'ROC-AUC, then Recall, F1, Accuracy'
        }, file, indent=2)
    with open(os.path.join(output_dir, f'{dataset_name}_summary.txt'), 'w', encoding='utf-8') as file:
        file.write(f'{dataset_name.upper()}\nBest Model: {best["Model"]}\n')
        for metric in METRIC_COLUMNS:
            file.write(f'{metric}: {best_metrics[metric]:.4f}\n')
        file.write(f'\nSecond Best Model: {second["Model"]}\n')
        file.write('Selection: ROC-AUC primary, Recall/F1/Accuracy tie-breakers.\n')
        file.write(f'Error analysis positive label: {positive_label}\n')
        file.write(f'False Positive Rate: {error_analysis["false_positive_rate"]:.4f}\n')
        file.write(f'False Negative Rate: {error_analysis["false_negative_rate"]:.4f}\n')
    return results_df, best, error_analysis

def evaluate_dataset1(data_path, models_dir, output_dir):
    """
    Evaluates all 6 models for Dataset 1 and saves results.
    """
    print("\n" + "="*50)
    print("EVALUATING MODELS FOR DATASET 1")
    print("="*50)
    
    # Load raw data
    df = pd.read_csv(data_path)
    df_clean, _, _ = dataset1_preprocessing.handle_duplicates(df)
    
    target_col = 'Diabetes_012'
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Stratified K-Fold for CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model_names = ['logistic_regression', 'decision_tree', 'random_forest', 'svm', 'xgboost', 'catboost']
    results = []
    
    # Plot configuration for joint ROC curve
    plt.figure(figsize=(10, 8))
    
    # Create subdirectories
    d1_out_dir = os.path.join(output_dir, 'dataset1')
    os.makedirs(os.path.join(d1_out_dir, 'confusion_matrices'), exist_ok=True)
    os.makedirs(os.path.join(d1_out_dir, 'roc_curves'), exist_ok=True)
    os.makedirs(os.path.join(d1_out_dir, 'feature_importance'), exist_ok=True)
    
    for name in model_names:
        model_path = os.path.join(models_dir, 'dataset1', f"{name}.joblib")
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Skipping.")
            continue
            
        print(f"Evaluating {name}...")
        pipeline = joblib.load(model_path)
        
        # 1. Cross Validation on training data
        # To avoid hanging on SVM, run CV on the same subset
        if name == 'svm':
            sub_size = 12000
            X_train_sub, _, y_train_sub, _ = train_test_split(
                X_train, y_train, train_size=sub_size, stratify=y_train, random_state=42
            )
            cv_scores = cross_val_score(pipeline, X_train_sub, y_train_sub, cv=cv, scoring='roc_auc_ovr', n_jobs=-1)
        else:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc_ovr', n_jobs=-1)
            
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # 2. Test predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)
        
        # 3. Compute metrics (macro average for multi-class)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': roc_auc,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
        
        # 4. Save Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'])
        plt.title(f"Confusion Matrix: {name} (Dataset 1)")
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        cm_path = os.path.join(d1_out_dir, 'confusion_matrices', f"{name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=100)
        plt.close()

        # Save the full test-set classification report alongside the matrix.
        report = classification_report(
            y_test, y_pred,
            labels=[0, 1, 2],
            target_names=['No Diabetes', 'Prediabetes', 'Diabetes'],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(
            os.path.join(d1_out_dir, 'confusion_matrices', f"{name}_classification_report.csv")
        )
        
        # 5. Plot ROC Curve for Diabetes class (Class 2) vs others
        # Binarize labels: Class 2 vs rest
        y_test_bin = (y_test == 2).astype(int)
        y_prob_class2 = y_prob[:, 2] # Probability of Class 2
        fpr, tpr, _ = roc_curve(y_test_bin, y_prob_class2)
        roc_auc_class2 = roc_auc_score(y_test_bin, y_prob_class2)
        
        # Add to the joint plot (active figure is set to the joint one at start)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"{name} (AUC Class 2 = {roc_auc_class2:.3f})")
        
        # 6. Save Feature Importance (if supported)
        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Extract processed feature names
        num_cols, cat_cols = dataset1_preprocessing.get_column_lists()
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_features_out = cat_encoder.get_feature_names_out(cat_cols)
        else:
            cat_features_out = cat_encoder.get_feature_names(cat_cols)
        feature_names = list(num_cols) + list(cat_features_out)
        
        feat_imp_df = None
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        elif hasattr(classifier, 'coef_'):
            # Multiclass coefficients
            coefs = classifier.coef_
            if coefs.ndim > 1:
                # Average coefficient magnitude across classes
                mean_coefs = np.mean(np.abs(coefs), axis=0)
                feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_coefs})
            else:
                feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coefs)})
                
        if feat_imp_df is not None:
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
            feat_imp_path = os.path.join(d1_out_dir, 'feature_importance', f"{name}_feature_importance.csv")
            feat_imp_df.to_csv(feat_imp_path, index=False)
            
    # Save the joint ROC plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves (Diabetes Class 2 vs others): Dataset 1')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, 'dataset1_roc_comparison.png')
    plt.savefig(roc_plot_path, dpi=120)
    plt.close()
    
    # Create comparison dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(d1_out_dir, 'model_comparison.csv'), index=False)
    results_df.to_csv(os.path.join(output_dir, 'dataset1_model_comparison.csv'), index=False)
    
    # Print comparison
    print("\nDataset 1 Model Comparison:")
    print(results_df.to_string(index=False))
    
    # Select by ROC-AUC, then recall and F1 as deterministic tie-breakers.
    best_model_idx = (
        results_df.sort_values(['ROC-AUC', 'Recall', 'F1'], ascending=False)
        .index[0]
    )
    best_model_row = results_df.loc[best_model_idx]
    print(f"\nBest Model for Dataset 1: {best_model_row['Model']} (ROC-AUC = {best_model_row['ROC-AUC']:.4f})")
    
    return results_df, best_model_row['Model']

def evaluate_dataset2(data_path, models_dir, output_dir):
    """
    Evaluates all 6 models for Dataset 2 and saves results.
    """
    print("\n" + "="*50)
    print("EVALUATING MODELS FOR DATASET 2")
    print("="*50)
    
    # Load raw data
    df = pd.read_csv(data_path)
    df_clean, _, _ = dataset2_preprocessing.handle_duplicates(df)
    
    target_col = 'diagnosed_diabetes'
    other_targets = ['diabetes_stage', 'diabetes_risk_score']
    
    # Separate features and target
    X = df_clean.drop(columns=[target_col] + other_targets)
    y = df_clean[target_col].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Stratified K-Fold for CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model_names = ['logistic_regression', 'decision_tree', 'random_forest', 'svm', 'xgboost', 'catboost']
    results = []
    
    # Plot configuration for joint ROC curve
    plt.figure(figsize=(10, 8))
    
    # Create subdirectories
    d2_out_dir = os.path.join(output_dir, 'dataset2')
    os.makedirs(os.path.join(d2_out_dir, 'confusion_matrices'), exist_ok=True)
    os.makedirs(os.path.join(d2_out_dir, 'roc_curves'), exist_ok=True)
    os.makedirs(os.path.join(d2_out_dir, 'feature_importance'), exist_ok=True)
    
    for name in model_names:
        model_path = os.path.join(models_dir, 'dataset2', f"{name}.joblib")
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Skipping.")
            continue
            
        print(f"Evaluating {name}...")
        pipeline = joblib.load(model_path)
        
        # 1. Cross Validation on training data
        if name == 'svm':
            sub_size = 15000
            X_train_sub, _, y_train_sub, _ = train_test_split(
                X_train, y_train, train_size=sub_size, stratify=y_train, random_state=42
            )
            cv_scores = cross_val_score(pipeline, X_train_sub, y_train_sub, cv=cv, scoring='roc_auc', n_jobs=-1)
        else:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # 2. Test predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)
        
        # 3. Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': roc_auc,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
        
        # 4. Save Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.title(f"Confusion Matrix: {name} (Dataset 2)")
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        cm_path = os.path.join(d2_out_dir, 'confusion_matrices', f"{name}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=100)
        plt.close()

        report = classification_report(
            y_test, y_pred,
            labels=[0, 1],
            target_names=['No Diabetes', 'Diabetes'],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(
            os.path.join(d2_out_dir, 'confusion_matrices', f"{name}_classification_report.csv")
        )
        
        # 5. Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        
        # Add to the joint plot
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
        
        # 6. Save Feature Importance (if supported)
        classifier = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Extract processed feature names
        num_cols, cat_cols = dataset2_preprocessing.get_column_lists(target_col)
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_features_out = cat_encoder.get_feature_names_out(cat_cols)
        else:
            cat_features_out = cat_encoder.get_feature_names(cat_cols)
        feature_names = list(num_cols) + list(cat_features_out)
        
        feat_imp_df = None
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        elif hasattr(classifier, 'coef_'):
            coefs = classifier.coef_
            if coefs.ndim > 1:
                mean_coefs = np.mean(np.abs(coefs), axis=0)
                feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_coefs})
            else:
                feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coefs)})
                
        if feat_imp_df is not None:
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
            feat_imp_path = os.path.join(d2_out_dir, 'feature_importance', f"{name}_feature_importance.csv")
            feat_imp_df.to_csv(feat_imp_path, index=False)
            
    # Save the joint ROC plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves (Diabetes Diagnosis): Dataset 2')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, 'dataset2_roc_comparison.png')
    plt.savefig(roc_plot_path, dpi=120)
    plt.close()
    
    # Create comparison dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(d2_out_dir, 'model_comparison.csv'), index=False)
    results_df.to_csv(os.path.join(output_dir, 'dataset2_model_comparison.csv'), index=False)
    
    # Print comparison
    print("\nDataset 2 Model Comparison:")
    print(results_df.to_string(index=False))
    
    # Select by ROC-AUC, then recall and F1 as deterministic tie-breakers.
    best_model_idx = (
        results_df.sort_values(['ROC-AUC', 'Recall', 'F1'], ascending=False)
        .index[0]
    )
    best_model_row = results_df.loc[best_model_idx]
    print(f"\nBest Model for Dataset 2: {best_model_row['Model']} (ROC-AUC = {best_model_row['ROC-AUC']:.4f})")
    
    return results_df, best_model_row['Model']

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    output_dir = os.path.join(base_dir, 'results', 'model_metrics')
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset1_path = os.path.join(data_dir, 'diabetes_012_health_indicators_BRFSS2015.csv')
    dataset2_path = os.path.join(data_dir, 'Diabetes_and_LifeStyle_Dataset .csv')
    
    dataset_results = []
    if os.path.exists(dataset1_path):
        results1, best1, errors1 = _evaluate_saved_dataset(dataset1_path, models_dir, output_dir, 'dataset1')
        best_metrics1 = results1[results1['Model'] == best1['Model']].iloc[0]
        dataset_results.append({'Dataset': 'dataset1', 'Best Model': best1['Model'], **{metric: best_metrics1[metric] for metric in METRIC_COLUMNS}})
    else:
        print(f"Error: Dataset 1 not found at {dataset1_path}")

    if os.path.exists(dataset2_path):
        results2, best2, errors2 = _evaluate_saved_dataset(dataset2_path, models_dir, output_dir, 'dataset2')
        best_metrics2 = results2[results2['Model'] == best2['Model']].iloc[0]
        dataset_results.append({'Dataset': 'dataset2', 'Best Model': best2['Model'], **{metric: best_metrics2[metric] for metric in METRIC_COLUMNS}})
    else:
        print(f"Error: Dataset 2 not found at {dataset2_path}")

    if dataset_results:
        pd.DataFrame(dataset_results).to_csv(
            os.path.join(output_dir, 'dataset_comparison.csv'), index=False, float_format='%.4f'
        )

    print("\n" + "="*50)
    print("BEST MODELS SELECTION REPORT")
    print("="*50)
    for row in dataset_results:
        print(f"{row['Dataset']}: {row['Best Model']} (ROC-AUC={row['ROC-AUC']:.4f})")
    print("="*50 + "\n")
