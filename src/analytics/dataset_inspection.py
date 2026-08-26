import os
import pandas as pd
import numpy as np

def identify_target_candidates(df):
    """
    Identifies potential target/output columns based on column names.
    """
    keywords = ['diabetes', 'target', 'class', 'label', 'outcome', 'diagnos']
    candidates = []
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            # Exclude obvious predictor-only columns like family history or diet
            if 'history' not in col.lower() and 'family' not in col.lower():
                candidates.append(col)
    return candidates

def detect_outliers_iqr(df, numerical_cols):
    """
    Detects outliers using the Interquartile Range (IQR) method.
    Returns a dictionary of outlier counts and percentages for each continuous column.
    """
    outliers_info = {}
    for col in numerical_cols:
        # Only run outlier detection on continuous variables (more than 15 unique values)
        if df[col].nunique() > 15:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_series = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outliers_info[col] = {
                'count': len(outlier_series),
                'percentage': (len(outlier_series) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    return outliers_info

def check_invalid_suspicious_values(df):
    """
    Checks for biologically suspicious or invalid values in common medical variables.
    """
    warnings = []
    
    # 1. Check for negative values in any column (since all variables in these datasets should be non-negative)
    for col in df.select_dtypes(include=[np.number]).columns:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            warnings.append(f"Column '{col}' contains {neg_count} negative value(s).")
            
    # 2. Check for suspicious zero values in clinical variables where zero is physiologically impossible/unlikely
    impossible_zero_fields = [
        'bmi', 'BMI', 
        'systolic_bp', 'diastolic_bp', 
        'heart_rate', 
        'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
        'glucose_fasting', 'glucose_postprandial', 
        'hba1c'
    ]
    for field in impossible_zero_fields:
        for col in df.columns:
            if field.lower() in col.lower():
                # Avoid binary flags or indicators
                if df[col].nunique() > 2:
                    zero_count = (df[col] == 0).sum()
                    if zero_count > 0:
                        warnings.append(f"Column '{col}' contains {zero_count} zero value(s) which may represent missing data.")
                        
    # 3. Check for extreme BMI values
    for col in df.columns:
        if 'bmi' in col.lower():
            very_low_bmi = (df[col] < 10).sum()
            very_high_bmi = (df[col] > 80).sum()
            if very_low_bmi > 0:
                warnings.append(f"Column '{col}' has {very_low_bmi} value(s) below BMI of 10.")
            if very_high_bmi > 0:
                warnings.append(f"Column '{col}' has {very_high_bmi} value(s) above BMI of 80.")
                
    # 4. Check for extreme HbA1c values
    for col in df.columns:
        if 'hba1c' in col.lower():
            very_high_hba1c = (df[col] > 20).sum()
            if very_high_hba1c > 0:
                warnings.append(f"Column '{col}' has {very_high_hba1c} extreme value(s) above HbA1c of 20%.")
                
    return warnings

def generate_dataset_report(filepath):
    filename = os.path.basename(filepath)
    print(f"Inspecting {filename}...")
    
    # Load dataset
    df = pd.read_csv(filepath)
    
    rows, cols = df.shape
    column_names = list(df.columns)
    
    # Data type of every column
    col_dtypes = {col: str(df[col].dtype) for col in df.columns}
    
    # Determine Numerical vs Categorical
    numerical_cols = []
    categorical_cols = []
    
    # In Dataset 1, many columns are float64 but represent binary flags or ordinal scales (like Smoker, Stroke, GenHlth).
    # We classify them semantically:
    # - If data type is object, string, category, or boolean -> Categorical
    # - If numeric but unique values <= 15 -> Categorical (discrete codes, binary flags, ordinal scales)
    # - Otherwise -> Numerical
    for col in df.columns:
        dtype_str = col_dtypes[col]
        is_obj_like = df[col].dtype in ['object', 'bool', 'category'] or 'str' in dtype_str
        unique_count = df[col].nunique()
        
        if is_obj_like or unique_count <= 15:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
            
    # Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / rows) * 100
    
    # Duplicate rows
    num_duplicates = df.duplicated().sum()
    
    # Unique values for categorical columns
    categorical_uniques = {}
    for col in categorical_cols:
        uniques = df[col].dropna().unique()
        values_clean = []
        for val in uniques[:15]:
            if hasattr(val, 'item'):
                values_clean.append(val.item())
            else:
                values_clean.append(val)
        categorical_uniques[col] = {
            'num_unique': len(uniques),
            'values': values_clean
        }
        
    # Target candidate identification
    target_candidates = identify_target_candidates(df)
    
    # Class distribution and imbalance
    target_distributions = {}
    for cand in target_candidates:
        dist = df[cand].value_counts(dropna=False)
        pct = df[cand].value_counts(normalize=True, dropna=False) * 100
        counts_dict = {}
        pct_dict = {}
        for val, count in dist.items():
            val_clean = val.item() if hasattr(val, 'item') else val
            counts_dict[val_clean] = int(count)
            pct_dict[val_clean] = float(pct[val])
        target_distributions[cand] = {
            'counts': counts_dict,
            'percentages': pct_dict
        }
        
    # Numerical mins and maxs
    numerical_min_max = {}
    for col in numerical_cols:
        min_v = df[col].min()
        max_v = df[col].max()
        numerical_min_max[col] = {
            'min': min_v.item() if hasattr(min_v, 'item') else min_v,
            'max': max_v.item() if hasattr(max_v, 'item') else max_v
        }
        
    # Outliers
    outliers = detect_outliers_iqr(df, numerical_cols)
    
    # Suspicious values
    suspicious_warnings = check_invalid_suspicious_values(df)
    
    # Basic statistical summary
    stats_df = df.describe(include='all').transpose()
    
    # Format statistical summary as text
    stats_summary = []
    stats_summary.append(f"{'Column':<35} | {'count':<10} | {'mean':<10} | {'std':<10} | {'min':<10} | {'50%':<10} | {'max':<10}")
    stats_summary.append("-" * 110)
    for idx, row in stats_df.iterrows():
        cnt = f"{row.get('count', 'N/A'):.1f}" if pd.notnull(row.get('count')) else 'N/A'
        mean = f"{row.get('mean', 'N/A'):.2f}" if pd.notnull(row.get('mean')) else 'N/A'
        std = f"{row.get('std', 'N/A'):.2f}" if pd.notnull(row.get('std')) else 'N/A'
        min_v = f"{row.get('min', 'N/A'):.2f}" if pd.notnull(row.get('min')) else 'N/A'
        median = f"{row.get('50%', 'N/A'):.2f}" if pd.notnull(row.get('50%')) else 'N/A'
        max_v = f"{row.get('max', 'N/A'):.2f}" if pd.notnull(row.get('max')) else 'N/A'
        stats_summary.append(f"{str(idx):<35} | {cnt:<10} | {mean:<10} | {std:<10} | {min_v:<10} | {median:<10} | {max_v:<10}")

    # Build report dict
    report = {
        'filename': filename,
        'rows': rows,
        'cols': cols,
        'column_names': column_names,
        'col_dtypes': col_dtypes,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'missing_counts': missing_counts.to_dict(),
        'missing_pct': missing_pct.to_dict(),
        'num_duplicates': int(num_duplicates),
        'categorical_uniques': categorical_uniques,
        'target_candidates': target_candidates,
        'target_distributions': target_distributions,
        'numerical_min_max': numerical_min_max,
        'outliers': outliers,
        'suspicious_warnings': suspicious_warnings,
        'stats_summary': "\n".join(stats_summary)
    }
    
    return report

def write_report_to_file(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_filename = f"{os.path.splitext(report['filename'])[0].strip()}_inspection_report.md"
    filepath = os.path.join(output_dir, report_filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Dataset Inspection Report: {report['filename']}\n\n")
        f.write(f"## 1. Basic Metadata\n")
        f.write(f"- **Filename**: {report['filename']}\n")
        f.write(f"- **Number of Rows**: {report['rows']:,}\n")
        f.write(f"- **Number of Columns**: {report['cols']}\n")
        f.write(f"- **Number of Duplicate Rows**: {report['num_duplicates']:,} ({ (report['num_duplicates'] / report['rows'])*100:.2f}%)\n\n")
        
        f.write(f"## 2. Column Variables and Data Types\n")
        f.write(f"The dataset contains **{len(report['numerical_cols'])}** numerical columns and **{len(report['categorical_cols'])}** categorical columns (including binary flags and ordinal codes).\n\n")
        
        f.write(f"### Columns by Semantic Classification\n")
        f.write(f"- **Numerical ({len(report['numerical_cols'])}):** {', '.join(report['numerical_cols'])}\n")
        f.write(f"- **Categorical ({len(report['categorical_cols'])}):** {', '.join(report['categorical_cols'])}\n\n")
        
        f.write(f"### Complete Data Type Details\n")
        f.write(f"| Column Name | Pandas Dtype | Inferred Type |\n")
        f.write(f"| --- | --- | --- |\n")
        for col in report['column_names']:
            inferred = "Numerical" if col in report['numerical_cols'] else "Categorical"
            f.write(f"| `{col}` | `{report['col_dtypes'][col]}` | {inferred} |\n")
        f.write("\n")
        
        f.write(f"## 3. Data Integrity & Completeness\n")
        f.write(f"### Missing Values\n")
        missing_exist = False
        f.write(f"| Column Name | Missing Count | Percentage Missing |\n")
        f.write(f"| --- | --- | --- |\n")
        for col in report['column_names']:
            m_cnt = report['missing_counts'][col]
            m_pct = report['missing_pct'][col]
            if m_cnt > 0:
                missing_exist = True
            f.write(f"| `{col}` | {m_cnt:,} | {m_pct:.2f}% |\n")
        f.write("\n")
        if not missing_exist:
            f.write("> **Note**: No missing values were detected in any column.\n\n")
            
        f.write(f"### Suspicious and Invalid Values\n")
        if report['suspicious_warnings']:
            for warning in report['suspicious_warnings']:
                f.write(f"- [WARNING] {warning}\n")
            f.write("\n")
        else:
            f.write("No obvious suspicious or invalid values detected (no negative values or unexpected zero clinical readings).\n\n")
            
        f.write(f"## 4. Target Variables & Class Distributions\n")
        if not report['target_candidates']:
            f.write("No obvious target column candidates were automatically identified.\n\n")
        else:
            f.write(f"Identified target candidate(s): {', '.join([f'`{c}`' for c in report['target_candidates']])}\n\n")
            for target in report['target_candidates']:
                f.write(f"### Class Distribution for Target Column: `{target}`\n")
                f.write(f"| Class / Value | Count | Percentage |\n")
                f.write(f"| --- | --- | --- |\n")
                dist = report['target_distributions'][target]
                for val, cnt in dist['counts'].items():
                    pct = dist['percentages'][val]
                    f.write(f"| {val} | {cnt:,} | {pct:.2f}% |\n")
                f.write("\n")
                
                # Check for class imbalance
                unique_vals = list(dist['counts'].keys())
                if len(unique_vals) >= 2:
                    min_pct = min(dist['percentages'].values())
                    max_pct = max(dist['percentages'].values())
                    ratio = max_pct / min_pct if min_pct > 0 else float('inf')
                    f.write(f"**Imbalance Analysis**: The class ratio between majority and minority is **{ratio:.2f}:1**. ")
                    if ratio > 3:
                        f.write("There is a **significant class imbalance** that may need to be addressed during modeling (e.g., SMOTE, class weighting).\n\n")
                    else:
                        f.write("The classes are relatively balanced.\n\n")
                        
        f.write(f"## 5. Statistical Summaries\n")
        f.write(f"### Numerical Columns (Min/Max/Outliers)\n")
        f.write(f"| Column Name | Min Value | Max Value | Outliers (IQR Method) | Outlier Percentage |\n")
        f.write(f"| --- | --- | --- | --- | --- |\n")
        for col in report['numerical_cols']:
            min_v = report['numerical_min_max'][col]['min']
            max_v = report['numerical_min_max'][col]['max']
            if col in report['outliers']:
                o_cnt = f"{report['outliers'][col]['count']:,}"
                o_pct = f"{report['outliers'][col]['percentage']:.2f}%"
            else:
                o_cnt = "N/A"
                o_pct = "N/A"
            f.write(f"| `{col}` | {min_v} | {max_v} | {o_cnt} | {o_pct} |\n")
        f.write("\n")
        
        f.write(f"### Categorical Columns Unique Values (Top 15 values displayed)\n")
        for col in report['categorical_cols']:
            details = report['categorical_uniques'][col]
            f.write(f"- **`{col}`** ({details['num_unique']} unique values): {details['values']}\n")
        f.write("\n")
        
        f.write(f"### Basic Descriptives Table\n")
        f.write("```\n")
        f.write(report['stats_summary'])
        f.write("\n```\n")
        
    print(f"Saved inspection report: {filepath}")
    return filepath

def print_console_summary(report):
    print("="*60)
    print(f"DATASET SUMMARY: {report['filename']}")
    print("="*60)
    print(f"Dimensions: {report['rows']:,} rows, {report['cols']} columns")
    print(f"Duplicates: {report['num_duplicates']:,} duplicate rows")
    print(f"Variables:  {len(report['numerical_cols'])} numerical, {len(report['categorical_cols'])} categorical")
    print(f"Missing Values: {sum(report['missing_counts'].values())} total missing values")
    
    print("\nTarget Column Candidate(s):")
    for target in report['target_candidates']:
        print(f"  * '{target}':")
        dist = report['target_distributions'][target]
        for val, cnt in dist['counts'].items():
            pct = dist['percentages'][val]
            print(f"    - Class {val}: {cnt:,} rows ({pct:.2f}%)")
            
    print("\nSuspicious Clinical Readings / Data Warnings:")
    if report['suspicious_warnings']:
        for warning in report['suspicious_warnings']:
            print(f"  [WARNING] {warning}")
    else:
        print("  None detected.")
        
    print("\nOutliers Summary (IQR Method):")
    if report['outliers']:
        for col, info in report['outliers'].items():
            print(f"  * '{col}': {info['count']:,} outliers ({info['percentage']:.2f}%) - Bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
    else:
        print("  No continuous numerical columns with outliers.")
    print("="*60 + "\n")

if __name__ == '__main__':
    # Determine base directory
    # Workspace root is likely two levels up from src/analytics/
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "results", "analysis")
    
    # Load files in data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        exit(1)
        
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        print(f"Error: No CSV datasets found in {data_dir}")
        exit(1)
        
    print(f"Found {len(files)} dataset files for analysis in {data_dir}.")
    
    reports = []
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        report = generate_dataset_report(filepath)
        reports.append(report)
        print_console_summary(report)
        write_report_to_file(report, output_dir)
        
    print("Dataset inspection completed successfully!")
