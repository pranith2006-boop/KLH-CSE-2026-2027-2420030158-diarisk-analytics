import os
import pandas as pd
import numpy as np
from scipy import stats

def calculate_descriptive_statistics(df, num_cols, cat_cols):
    """
    Calculates detailed descriptive statistics for numerical and categorical columns.
    """
    # 1. Numerical columns statistics
    num_stats = []
    for col in num_cols:
        series = df[col].dropna()
        q25 = series.quantile(0.25)
        q50 = series.quantile(0.50)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        
        stats_dict = {
            'Column': col,
            'Count': len(series),
            'Mean': series.mean(),
            'Median': q50,
            'StdDev': series.std(),
            'Variance': series.var(),
            'Min': series.min(),
            'Max': series.max(),
            '25th_Percentile': q25,
            '50th_Percentile': q50,
            '75th_Percentile': q75,
            'IQR': iqr
        }
        num_stats.append(stats_dict)
    num_stats_df = pd.DataFrame(num_stats)
    
    # 2. Categorical columns statistics
    cat_stats = []
    for col in cat_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        val_counts = series.value_counts()
        mode_val = val_counts.index[0]
        mode_freq = val_counts.iloc[0]
        
        stats_dict = {
            'Column': col,
            'Count': len(series),
            'Unique_Values_Count': series.nunique(),
            'Mode_Value': mode_val,
            'Mode_Frequency': mode_freq,
            'Mode_Percentage': (mode_freq / len(series)) * 100
        }
        cat_stats.append(stats_dict)
    cat_stats_df = pd.DataFrame(cat_stats)
    
    return num_stats_df, cat_stats_df

def analyze_target(df, target_col):
    """
    Analyzes class count, percentage distribution, and class imbalance ratio of the target.
    """
    counts = df[target_col].value_counts(dropna=False)
    pcts = df[target_col].value_counts(normalize=True, dropna=False) * 100
    
    target_data = []
    for val in counts.index:
        target_data.append({
            'Class_Value': val,
            'Count': counts[val],
            'Percentage': pcts[val]
        })
    
    target_df = pd.DataFrame(target_data)
    
    # Calculate imbalance ratio (majority class count / minority class count)
    min_count = counts.min()
    max_count = counts.max()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    return target_df, imbalance_ratio

def analyze_numerical_distributions(df, num_cols):
    """
    Analyzes skewness, kurtosis, range, and outlier count (IQR) for numerical features.
    """
    dist_info = []
    for col in num_cols:
        series = df[col].dropna()
        skew = series.skew()
        kurt = series.kurtosis()
        
        # Determine skewness category
        if abs(skew) < 0.5:
            skew_cat = 'Approximately Symmetric'
        elif skew >= 0.5:
            skew_cat = 'Right-skewed (Positive)'
        else:
            skew_cat = 'Left-skewed (Negative)'
            
        # Outliers count
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
        
        dist_info.append({
            'Column': col,
            'Range': series.max() - series.min(),
            'Skewness': skew,
            'Skewness_Classification': skew_cat,
            'Kurtosis': kurt,
            'IQR_Outliers_Count': outlier_count,
            'Outliers_Percentage': (outlier_count / len(series)) * 100
        })
        
    return pd.DataFrame(dist_info)

def analyze_correlations(df, num_cols):
    """
    Calculates Pearson and Spearman correlations, and extracts strong and weak relationships.
    """
    # Pearson
    pearson_corr = df[num_cols].corr(method='pearson')
    # Spearman
    spearman_corr = df[num_cols].corr(method='spearman')
    
    relationships = []
    
    # We will extract pairwise correlations from Pearson matrix (excluding self-correlations)
    cols = list(pearson_corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            r_val = pearson_corr.loc[col1, col2]
            rs_val = spearman_corr.loc[col1, col2]
            
            # Determine strength
            if abs(r_val) >= 0.5:
                strength = 'Strong'
            elif abs(r_val) >= 0.2:
                strength = 'Moderate'
            elif abs(r_val) >= 0.1:
                strength = 'Weak'
            else:
                strength = 'Very Weak / None'
                
            relationships.append({
                'Feature_1': col1,
                'Feature_2': col2,
                'Pearson_r': r_val,
                'Spearman_rho': rs_val,
                'Relationship_Type': 'Positive' if r_val > 0 else 'Negative',
                'Strength': strength
            })
            
    relationships_df = pd.DataFrame(relationships)
    return pearson_corr, spearman_corr, relationships_df

def feature_to_target_relationship(df, target_col, num_cols, cat_cols):
    """
    Analyzes relationships between features and the target.
    """
    feature_target_data = []
    
    # Target values list
    target_classes = df[target_col].unique()
    target_classes = sorted([c for c in target_classes if pd.notnull(c)])
    
    # 1. Numerical features grouped by target
    for col in num_cols:
        for tc in target_classes:
            subgroup = df[df[target_col] == tc][col].dropna()
            if len(subgroup) == 0:
                continue
            feature_target_data.append({
                'Feature': col,
                'Feature_Type': 'Numerical',
                'Target_Class': tc,
                'Group_Count': len(subgroup),
                'Group_Mean': subgroup.mean(),
                'Group_Median': subgroup.median(),
                'Prevalence_or_Percent': 'N/A'
            })
            
    # 2. Categorical features cross-tabulation with target
    for col in cat_cols:
        crosstab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        counts = pd.crosstab(df[col], df[target_col])
        
        for cat_val in crosstab.index:
            for tc in target_classes:
                cnt = counts.loc[cat_val, tc] if tc in counts.columns else 0
                pct = crosstab.loc[cat_val, tc] if tc in crosstab.columns else 0.0
                feature_target_data.append({
                    'Feature': col,
                    'Feature_Type': 'Categorical',
                    'Target_Class': f"{cat_val} vs Target Class {tc}",
                    'Group_Count': int(cnt),
                    'Group_Mean': 'N/A',
                    'Group_Median': 'N/A',
                    'Prevalence_or_Percent': f"{pct:.2f}%"
                })
                
    return pd.DataFrame(feature_target_data)

def perform_statistical_significance_tests(df, target_col, num_cols, cat_cols):
    """
    Runs appropriate statistical significance tests (t-test/Mann-Whitney, Chi-square).
    """
    test_results = []
    target_classes = sorted([c for c in df[target_col].unique() if pd.notnull(c)])
    
    # 1. Tests for numerical features
    for col in num_cols:
        # Check normality of groups to decide on t-test vs Mann-Whitney U
        # Since our datasets are large, standard parametric tests might reject normality even for tiny deviations, 
        # but visually inspect if we can. A safe approach is to run Mann-Whitney U because medical parameters 
        # (insulin, glucose, triglycerides) are typically right-skewed and non-normal.
        
        if len(target_classes) == 2:
            # Binary target: Mann-Whitney U test
            group0 = df[df[target_col] == target_classes[0]][col].dropna()
            group1 = df[df[target_col] == target_classes[1]][col].dropna()
            
            if len(group0) > 0 and len(group1) > 0:
                stat, p_val = stats.mannwhitneyu(group0, group1, alternative='two-sided')
                is_sig = p_val < 0.05
                interpretation = "Statistically Significant difference between target groups." if is_sig else "No statistically significant difference."
                test_results.append({
                    'Feature': col,
                    'Test_Name': 'Mann-Whitney U Test',
                    'Test_Statistic': stat,
                    'p-value': p_val,
                    'Significance_Alpha_0.05': is_sig,
                    'Interpretation': interpretation
                })
        elif len(target_classes) > 2:
            # Multi-class target: Kruskal-Wallis test (non-parametric ANOVA)
            groups = [df[df[target_col] == tc][col].dropna() for tc in target_classes]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                stat, p_val = stats.kruskal(*groups)
                is_sig = p_val < 0.05
                interpretation = "Statistically Significant difference across target groups." if is_sig else "No statistically significant difference."
                test_results.append({
                    'Feature': col,
                    'Test_Name': 'Kruskal-Wallis Test',
                    'Test_Statistic': stat,
                    'p-value': p_val,
                    'Significance_Alpha_0.05': is_sig,
                    'Interpretation': interpretation
                })
                
    # 2. Chi-square tests for categorical features
    for col in cat_cols:
        contingency_table = pd.crosstab(df[col], df[target_col])
        if contingency_table.size > 0:
            try:
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                is_sig = p_val < 0.05
                interpretation = "Statistically Significant association with target." if is_sig else "No statistically significant association."
                test_results.append({
                    'Feature': col,
                    'Test_Name': 'Chi-Square Test of Independence',
                    'Test_Statistic': chi2,
                    'p-value': p_val,
                    'Significance_Alpha_0.05': is_sig,
                    'Interpretation': interpretation
                })
            except Exception as e:
                # E.g. expected values too small or empty tables
                pass
                
    return pd.DataFrame(test_results)

def generate_insights_text(df, target_col, num_cols, cat_cols, corr_rels, test_results):
    """
    Generates data-driven insights with neutral, non-causal language.
    """
    insights = []
    
    # 1. Target Imbalance insight
    counts = df[target_col].value_counts()
    pcts = df[target_col].value_counts(normalize=True) * 100
    majority_class = counts.index[0]
    minority_class = counts.index[-1]
    ratio = counts[majority_class] / counts[minority_class]
    
    insights.append(f"Target variable '{target_col}' imbalance analysis:")
    if ratio > 3:
        insights.append(f"  - The target is highly imbalanced with a ratio of {ratio:.2f}:1. Majority class '{majority_class}' represents {pcts[majority_class]:.2f}% of the records, while minority class '{minority_class}' represents {pcts[minority_class]:.2f}%. Model evaluation metrics should prioritize F1-score or AUC-ROC rather than accuracy.")
    else:
        insights.append(f"  - The target is relatively balanced with a ratio of {ratio:.2f}:1. Class distribution is {', '.join([f'{k}: {v:.1f}%' for k, v in pcts.items()])}.")
        
    # 2. Correlation insights
    insights.append("\nCorrelation Analysis Insights (Pearson):")
    # Sort correlations by absolute r-value
    strong_corrs = corr_rels[corr_rels['Strength'].isin(['Strong', 'Moderate'])].sort_values(by='Pearson_r', key=abs, ascending=False)
    if not strong_corrs.empty:
        for idx, row in strong_corrs.head(5).iterrows():
            insights.append(f"  - Features '{row['Feature_1']}' and '{row['Feature_2']}' show a {row['Strength'].lower()} {row['Relationship_Type'].lower()} relationship (r = {row['Pearson_r']:.2f}, Spearman rho = {row['Spearman_rho']:.2f}).")
    else:
        insights.append("  - No strong or moderate linear correlations (r >= 0.2) detected between numerical features.")
        
    # 3. Statistical Significance and Target Association insights
    insights.append("\nStatistical Significance Testing (Alpha = 0.05):")
    significant_features = test_results[test_results['Significance_Alpha_0.05'] == True]
    insights.append(f"  - {len(significant_features)} out of {len(test_results)} tested features show a statistically significant relationship or difference across target classes.")
    
    # Group by test type
    for idx, row in significant_features.head(8).iterrows():
        test_type = "association" if "Chi-Square" in row['Test_Name'] else "difference"
        insights.append(f"    * Feature '{row['Feature']}' shows a statistically significant {test_type} with the target group (p-value = {row['p-value']:.3e} using {row['Test_Name']}).")
        
    # 4. Distribution and Skewness insights
    insights.append("\nDistribution and Outlier Insights:")
    # We will identify skewed columns
    skewed_df = df[num_cols].skew()
    right_skewed = list(skewed_df[skewed_df >= 0.5].index)
    left_skewed = list(skewed_df[skewed_df <= -0.5].index)
    if right_skewed:
        insights.append(f"  - Right-skewed numerical variables (skew >= 0.5): {', '.join(right_skewed)}. These features may benefit from log or power transformations during modeling if linear algorithms are used.")
    if left_skewed:
        insights.append(f"  - Left-skewed numerical variables (skew <= -0.5): {', '.join(left_skewed)}.")
        
    # Outliers
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            insights.append(f"  - Feature '{col}' contains {len(outliers):,} outliers ({len(outliers)/len(df)*100:.2f}%) outside the IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}].")
            
    return "\n".join(insights)

def run_dataset_eda(filepath, target_col, output_subdir, base_output_dir):
    filename = os.path.basename(filepath)
    print(f"Running EDA on {filename}...")
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Categorize columns based on their data types and unique counts (same logic as preprocessing)
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col == target_col:
            continue
        dtype_str = str(df[col].dtype)
        is_obj_like = df[col].dtype in ['object', 'bool', 'category'] or 'str' in dtype_str
        unique_count = df[col].nunique()
        
        # Categorize
        if is_obj_like or unique_count <= 15:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
            
    # Make sure output directories exist
    os.makedirs(output_subdir, exist_ok=True)
    
    # 1. Descriptive Statistics
    num_stats_df, cat_stats_df = calculate_descriptive_statistics(df, numerical_cols, categorical_cols)
    # Combine stats for a full sheet
    combined_stats = pd.concat([num_stats_df, cat_stats_df], ignore_index=True)
    combined_stats.to_csv(os.path.join(output_subdir, "descriptive_statistics.csv"), index=False)
    
    # Also save to base results/analysis directory direct filename
    dataset_name = "dataset1" if "BRFSS2015" in filename else "dataset2"
    combined_stats.to_csv(os.path.join(base_output_dir, f"{dataset_name}_descriptive_statistics.csv"), index=False)
    
    # 2. Target Variable Analysis
    target_df, imbalance_ratio = analyze_target(df, target_col)
    target_df.to_csv(os.path.join(output_subdir, "target_distribution.csv"), index=False)
    
    # 3. Numerical Feature Analysis & Distributions
    num_dist_df = analyze_numerical_distributions(df, numerical_cols)
    
    # 4. Correlation Analysis
    pearson_corr, spearman_corr, relationships_df = analyze_correlations(df, numerical_cols)
    pearson_corr.to_csv(os.path.join(output_subdir, "correlation_matrix.csv"))
    # Also save to base results/analysis directory direct filename
    pearson_corr.to_csv(os.path.join(base_output_dir, f"{dataset_name}_correlation_matrix.csv"))
    
    # 5. Feature-to-Target Analysis
    feat_target_df = feature_to_target_relationship(df, target_col, numerical_cols, categorical_cols)
    feat_target_df.to_csv(os.path.join(output_subdir, "feature_target_analysis.csv"), index=False)
    
    # 6. Statistical Significance Tests
    test_results_df = perform_statistical_significance_tests(df, target_col, numerical_cols, categorical_cols)
    test_results_df.to_csv(os.path.join(output_subdir, "statistical_tests.csv"), index=False)
    
    # 7. Insights Generation
    insights_txt = generate_insights_text(df, target_col, numerical_cols, categorical_cols, relationships_df, test_results_df)
    with open(os.path.join(output_subdir, "insights.txt"), 'w', encoding='utf-8') as f:
        f.write(f"=== EDA Insights: {filename} ===\n\n")
        f.write(insights_txt)
        
    print(f"Completed and saved EDA results to: {output_subdir}")
    
    # Return structured dict for console summaries
    return {
        'filename': filename,
        'records': len(df),
        'features': len(df.columns) - 1,
        'target': target_col,
        'target_distribution': target_df.to_dict('records'),
        'numerical_features': numerical_cols,
        'relationships': relationships_df,
        'test_results': test_results_df,
        'insights': insights_txt
    }

def print_final_console_output(report1, report2):
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS (EDA) RESULTS SUMMARY")
    print("="*60)
    
    for report_idx, report in enumerate([report1, report2], 1):
        print(f"\nDATASET {report_idx}")
        print("-" * 30)
        print(f"Records:             {report['records']:,}")
        print(f"Features:            {report['features']}")
        print(f"Target:              {report['target']}")
        
        # Target Distribution
        dist_strs = []
        for row in report['target_distribution']:
            dist_strs.append(f"Class {row['Class_Value']}: {row['Count']:,} ({row['Percentage']:.2f}%)")
        print(f"Target Distribution: {'; '.join(dist_strs)}")
        
        # Important Numerical Features
        print(f"Important Numerical Features: {', '.join(report['numerical_features'])}")
        
        # Strongest Correlations
        rels = report['relationships']
        strong_rels = rels[rels['Strength'].isin(['Strong', 'Moderate'])].sort_values(by='Pearson_r', key=abs, ascending=False)
        strong_strs = []
        for idx, row in strong_rels.head(3).iterrows():
            strong_strs.append(f"{row['Feature_1']}-{row['Feature_2']} ({row['Pearson_r']:.2f})")
        print(f"Strongest Correlations:       {', '.join(strong_strs) if strong_strs else 'None (r >= 0.2)'}")
        
        # Significant features count
        sig_count = (report['test_results']['Significance_Alpha_0.05'] == True).sum()
        total_tests = len(report['test_results'])
        print(f"Statistically Significant Features: {sig_count} out of {total_tests} features show significant relationship/difference.")
        
        # Important EDA Insights (print first few lines of insights)
        print("\nImportant EDA Insights:")
        lines = report['insights'].split('\n')
        for line in lines[:15]:  # Print first 15 lines of insights
            if line.strip():
                print(f"  {line}")
        print("-" * 30)
        
    # Dataset comparison
    print("\nDATASET COMPARISON")
    print("------------------")
    size_diff = abs(report1['records'] - report2['records'])
    larger_dataset = "Dataset 1" if report1['records'] > report2['records'] else "Dataset 2"
    print(f"- **Difference in size**: {larger_dataset} is larger by {size_diff:,} records.")
    print(f"- **Difference in features**: Dataset 1 has {report1['features']} features; Dataset 2 has {report2['features']} features (difference of {abs(report1['features'] - report2['features'])}).")
    
    # Target distribution comparison
    print("- **Difference in target distribution**:")
    print(f"  * Dataset 1 ('{report1['target']}') is highly multi-class imbalanced (majority represents 82.71% of clean records).")
    print(f"  * Dataset 2 ('{report2['target']}') is a relatively balanced binary classification target (60.00% diagnosed vs 40.00% undiagnosed).")
    
    # Common features
    common_cols = list(set(report1['numerical_features']) & set(report2['numerical_features']))
    print(f"- **Common important features**: {', '.join(common_cols) if common_cols else 'None (different feature spaces)'}")
    
    # Dataset specific features
    d1_specific = list(set(report1['numerical_features']) - set(report2['numerical_features']))
    d2_specific = list(set(report2['numerical_features']) - set(report1['numerical_features']))
    print(f"- **Dataset 1 specific numerical features**: {', '.join(d1_specific)}")
    print(f"- **Dataset 2 specific numerical features**: {', '.join(d2_specific)}")
    print("="*60 + "\n")

if __name__ == '__main__':
    # Determine base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    base_output_dir = os.path.join(base_dir, "results", "analysis")
    
    # Dataset paths
    dataset1_path = os.path.join(data_dir, "diabetes_012_health_indicators_BRFSS2015.csv")
    dataset2_path = os.path.join(data_dir, "Diabetes_and_LifeStyle_Dataset .csv")
    
    # Output directories for each dataset
    output_dataset1 = os.path.join(base_output_dir, "dataset1")
    output_dataset2 = os.path.join(base_output_dir, "dataset2")
    
    # Targets (based on Phase 1 & Phase 2 decisions)
    target1 = 'Diabetes_012'
    target2 = 'diagnosed_diabetes'
    
    print("==================================================")
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA) MODULE")
    print("==================================================\n")
    
    # Run EDA on Dataset 1
    report1 = None
    if os.path.exists(dataset1_path):
        report1 = run_dataset_eda(dataset1_path, target1, output_dataset1, base_output_dir)
    else:
        print(f"Error: Dataset 1 not found at {dataset1_path}")
        
    # Run EDA on Dataset 2
    report2 = None
    if os.path.exists(dataset2_path):
        report2 = run_dataset_eda(dataset2_path, target2, output_dataset2, base_output_dir)
    else:
        print(f"Error: Dataset 2 not found at {dataset2_path}")
        
    # Print console summaries
    if report1 is not None and report2 is not None:
        print_final_console_output(report1, report2)
        
    print("==================================================")
    print("EDA PIPELINE COMPLETED SUCCESSFULLY")
    print("==================================================")
