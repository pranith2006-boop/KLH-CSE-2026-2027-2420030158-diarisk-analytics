import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Monkeypatch Pandas 3.0.x / Plotly GroupBy get_group compatibility bug
original_get_group = pd.core.groupby.groupby.GroupBy.get_group

def patched_get_group(self, name):
    try:
        return original_get_group(self, name)
    except KeyError:
        if not isinstance(name, tuple):
            try:
                return original_get_group(self, (name,))
            except KeyError:
                pass
        elif isinstance(name, tuple) and len(name) == 1:
            try:
                return original_get_group(self, name[0])
            except KeyError:
                pass
        raise

pd.core.groupby.groupby.GroupBy.get_group = patched_get_group


def create_target_distribution(df, target_col, title=None):
    """
    Creates an interactive Pie/Donut chart for the target variable distribution.
    """
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found. Skipping chart.")
        return None
        
    counts = df[target_col].value_counts(dropna=False).reset_index()
    counts.columns = ['Class', 'Count']
    
    # Calculate percentage
    total = counts['Count'].sum()
    counts['Percentage'] = (counts['Count'] / total) * 100
    
    # Map class to string for clear labeling
    counts['Class'] = counts['Class'].astype(str)
    
    # Customize titles based on target
    if title is None:
        title = f"Distribution of Target Variable: {target_col}"
        
    fig = px.pie(
        counts, 
        names='Class', 
        values='Count',
        hole=0.4,
        title=title,
        hover_data=['Percentage'],
        labels={'Class': 'Target Class', 'Count': 'Records Count'},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hovertemplate="<b>Target Class: %{label}</b><br>Count: %{value:,}<br>Percentage: %{percent:.2%}<extra></extra>"
    )
    
    fig.update_layout(
        legend_title="Classes",
        font=dict(family="Outfit, Inter, Arial, sans-serif", size=12),
        margin=dict(t=50, b=30, l=30, r=30)
    )
    
    return fig

def create_histogram(df, num_col, target_col=None, title=None):
    """
    Creates an interactive histogram for a numerical feature with mean and median markers.
    """
    if num_col not in df.columns:
        print(f"Warning: Numerical column '{num_col}' not found. Skipping chart.")
        return None
        
    # Calculate stats
    mean_val = df[num_col].mean()
    median_val = df[num_col].median()
    
    if title is None:
        title = f"Distribution of Feature: {num_col}"
        
    # Plot histogram
    if target_col in df.columns:
        # If target exists, group and color-code by target class
        df_plot = df.copy()
        df_plot[target_col] = df_plot[target_col].astype(str)
        fig = px.histogram(
            df_plot,
            x=num_col,
            color=target_col,
            marginal="box", # Adds a box plot on top
            title=title,
            barmode="overlay",
            opacity=0.7,
            labels={num_col: num_col, target_col: 'Target Class'},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
    else:
        fig = px.histogram(
            df,
            x=num_col,
            marginal="box",
            title=title,
            opacity=0.8,
            color_discrete_sequence=['#3b82f6'] # Sleek blue
        )
        
    # Add vertical lines for mean and median
    fig.add_vline(
        x=mean_val, 
        line_dash="dash", 
        line_color="red", 
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top right"
    )
    fig.add_vline(
        x=median_val, 
        line_dash="dot", 
        line_color="green", 
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="top left"
    )
    
    fig.update_layout(
        xaxis_title=num_col,
        yaxis_title="Count",
        font=dict(family="Outfit, Inter, Arial, sans-serif", size=12),
        hovermode="x unified",
        margin=dict(t=50, b=30, l=30, r=30)
    )
    
    return fig

def create_boxplot(df, num_col, target_col=None, title=None):
    """
    Creates an interactive box plot. Compares numerical feature distributions across target classes if target is provided.
    """
    if num_col not in df.columns:
        print(f"Warning: Numerical column '{num_col}' not found. Skipping box plot.")
        return None
        
    if title is None:
        title = f"Box Plot of {num_col}"
        
    if target_col in df.columns:
        # Grouped box plot by target class
        df_plot = df.copy()
        df_plot[target_col] = df_plot[target_col].astype(str)
        
        # Determine logical names/order of classes
        unique_classes = sorted(df_plot[target_col].unique())
        
        fig = px.box(
            df_plot,
            x=target_col,
            y=num_col,
            color=target_col,
            title=title if title else f"{num_col} Distribution by Target Variable: {target_col}",
            category_orders={target_col: unique_classes},
            color_discrete_sequence=px.colors.qualitative.Safe,
            points="outliers" # Show outliers
        )
        fig.update_layout(xaxis_title=f"Target Class ({target_col})")
    else:
        # Simple box plot
        fig = px.box(
            df,
            y=num_col,
            title=title,
            color_discrete_sequence=['#3b82f6']
        )
        
    fig.update_layout(
        yaxis_title=num_col,
        font=dict(family="Outfit, Inter, Arial, sans-serif", size=12),
        margin=dict(t=50, b=30, l=30, r=30)
    )
    
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None, title=None):
    """
    Creates an interactive Scatter Plot supporting axis selections.
    """
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Warning: Columns '{x_col}' or '{y_col}' do not exist. Skipping scatter plot.")
        return None
        
    if title is None:
        title = f"Scatter Plot: {x_col} vs {y_col}"
        
    df_plot = df.copy()
    if color_col in df_plot.columns:
        df_plot[color_col] = df_plot[color_col].astype(str)
        fig = px.scatter(
            df_plot,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            opacity=0.6,
            labels={color_col: 'Class'},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
    else:
        fig = px.scatter(
            df_plot,
            x=x_col,
            y=y_col,
            title=title,
            opacity=0.6,
            color_discrete_sequence=['#3b82f6']
        )
        
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        font=dict(family="Outfit, Inter, Arial, sans-serif", size=12),
        margin=dict(t=50, b=30, l=30, r=30),
        legend_title="Classes" if color_col else None
    )
    
    return fig

def create_correlation_heatmap(df, num_cols, title=None):
    """
    Creates an interactive correlation heatmap for numerical columns.
    """
    # Filter columns that exist
    valid_cols = [c for c in num_cols if c in df.columns]
    if len(valid_cols) < 2:
        print("Warning: Insufficient continuous features to construct a correlation heatmap.")
        return None
        
    corr_matrix = df[valid_cols].corr(method='pearson')
    
    if title is None:
        title = "Correlation Heatmap (Numerical Features)"
        
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        color_continuous_scale="RdBu", # Diverging color palette
        range_color=[-1, 1],
        title=title,
        aspect="auto",
        labels=dict(color="Pearson Correlation")
    )
    
    # Annotate correlation coefficients on the map
    fig.update_traces(
        text=np.round(corr_matrix.values, 2), 
        texttemplate="%{text}",
        hovertemplate="Feature 1: %{x}<br>Feature 2: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    )
    
    # Add non-causal disclaimer subtitle as layout annotation
    fig.add_annotation(
        text="<i>Correlation indicates the strength of association between variables and does not imply causation.</i>",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10, color="gray")
    )
    
    fig.update_layout(
        font=dict(family="Outfit, Inter, Arial, sans-serif", size=11),
        margin=dict(t=60, b=80, l=40, r=40)
    )
    
    return fig

def create_feature_comparison(df, num_col, target_col, title=None):
    """
    Automatically creates a Grouped Violin Plot to compare continuous features across target classes.
    """
    if num_col not in df.columns or target_col not in df.columns:
        print(f"Warning: Feature '{num_col}' or Target '{target_col}' not found. Skipping violin comparison.")
        return None
        
    if title is None:
        title = f"Comparison of {num_col} across Target Classes: {target_col}"
        
    df_plot = df.copy()
    df_plot[target_col] = df_plot[target_col].astype(str)
    unique_classes = sorted(df_plot[target_col].unique())
    
    fig = px.violin(
        df_plot,
        x=target_col,
        y=num_col,
        color=target_col,
        box=True, # Draw box inside violin
        points=False, # Do not draw all points to keep HTML lightweight
        title=title,
        category_orders={target_col: unique_classes},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    fig.update_layout(
        xaxis_title=f"Target Class ({target_col})",
        yaxis_title=num_col,
        font=dict(family="Outfit, Inter, Arial, sans-serif", size=12),
        margin=dict(t=50, b=30, l=30, r=30)
    )
    
    return fig

def create_categorical_chart(df, cat_col, target_col=None, title=None):
    """
    Creates an interactive stacked or grouped bar chart for categorical variables against target.
    """
    if cat_col not in df.columns:
        print(f"Warning: Categorical column '{cat_col}' not found. Skipping chart.")
        return None
        
    if title is None:
        title = f"Categorical Distribution of {cat_col}"
        
    df_plot = df.copy()
    
    if target_col in df_plot.columns:
        # Grouped bar chart comparing counts of cat_col categories by target class
        df_plot[target_col] = df_plot[target_col].astype(str)
        df_plot[cat_col] = df_plot[cat_col].astype(str)
        
        # Aggregate counts
        agg = df_plot.groupby([cat_col, target_col]).size().reset_index(name='Count')
        
        fig = px.bar(
            agg,
            x=cat_col,
            y='Count',
            color=target_col,
            barmode='group',
            title=title if title else f"Distribution of {cat_col} by Target: {target_col}",
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={target_col: 'Target Class'}
        )
    else:
        # Simple count bar chart
        df_plot[cat_col] = df_plot[cat_col].astype(str)
        counts = df_plot[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, 'Count']
        
        fig = px.bar(
            counts,
            x=cat_col,
            y='Count',
            title=title,
            color_discrete_sequence=['#3b82f6']
        )
        
    fig.update_layout(
        xaxis_title=cat_col,
        yaxis_title="Count",
        font=dict(family="Outfit, Inter, Arial, sans-serif", size=12),
        margin=dict(t=50, b=30, l=30, r=30)
    )
    
    return fig

def create_kpi_summary(df, target_col):
    """
    Prepares KPI data for summary cards, dynamically handling column existence.
    """
    kpis = {
        'total_records': len(df),
        'total_features': len(df.columns) - 1 if target_col in df.columns else len(df.columns)
    }
    
    # Target counts (if exists)
    if target_col in df.columns:
        counts = df[target_col].value_counts()
        kpis['classes_counts'] = counts.to_dict()
        
        # If binary 0/1 target
        if set(counts.index).issubset({0, 1, 0.0, 1.0}):
            pos_label = 1 if 1 in counts.index else 1.0
            neg_label = 0 if 0 in counts.index else 0.0
            
            kpis['positive_cases'] = int(counts.get(pos_label, 0))
            kpis['negative_cases'] = int(counts.get(neg_label, 0))
            kpis['positive_percentage'] = float(kpis['positive_cases'] / len(df) * 100)
        # Multi-class target
        else:
            # Report the class sizes
            kpis['class_distributions'] = {str(k): int(v) for k, v in counts.items()}
            
    # Average age if exists (case insensitive check)
    age_cols = [col for col in df.columns if col.lower() == 'age']
    if age_cols:
        kpis['average_age'] = float(df[age_cols[0]].mean())
        
    # Average BMI if exists
    bmi_cols = [col for col in df.columns if col.lower() == 'bmi']
    if bmi_cols:
        kpis['average_bmi'] = float(df[bmi_cols[0]].mean())
        
    # Average Glucose if exists (check columns starting/ending with glucose)
    glucose_cols = [col for col in df.columns if 'glucose' in col.lower()]
    for col in glucose_cols:
        kpis[f'average_{col}'] = float(df[col].mean())
        
    return kpis

if __name__ == '__main__':
    # Determine base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "results", "visualizations")
    
    # CSV file paths
    dataset1_path = os.path.join(data_dir, "diabetes_012_health_indicators_BRFSS2015.csv")
    dataset2_path = os.path.join(data_dir, "Diabetes_and_LifeStyle_Dataset .csv")
    
    print("==================================================")
    print("STARTING PLOTLY DYNAMIC CHART GENERATION PIPELINE")
    print("==================================================\n")
    
    # --------------------------------------------------
    # Process Dataset 1
    # --------------------------------------------------
    if os.path.exists(dataset1_path):
        print("Processing Dataset 1 (BRFSS 2015)...")
        df1 = pd.read_csv(dataset1_path)
        d1_out_dir = os.path.join(output_dir, "dataset1")
        os.makedirs(d1_out_dir, exist_ok=True)
        
        # 1. Target Distribution
        fig_target = create_target_distribution(df1, 'Diabetes_012', "Distribution of Diabetes status (Dataset 1: CDC Survey)")
        if fig_target:
            fig_target.write_html(os.path.join(d1_out_dir, "target_distribution.html"), include_plotlyjs="cdn")
            
        # 2. Correlation Heatmap
        num_cols_1 = ['BMI', 'MentHlth', 'PhysHlth']
        fig_corr = create_correlation_heatmap(df1, num_cols_1, "Correlation Matrix of Numerical Survey Features")
        if fig_corr:
            fig_corr.write_html(os.path.join(d1_out_dir, "correlation_heatmap.html"), include_plotlyjs="cdn")
            
        # 3. Feature Distribution (BMI)
        fig_dist = create_histogram(df1, 'BMI', 'Diabetes_012', "Distribution of BMI grouped by Diabetes status")
        if fig_dist:
            fig_dist.write_html(os.path.join(d1_out_dir, "feature_distributions.html"), include_plotlyjs="cdn")
            
        # 4. Feature Comparison (BMI Violin Plot vs Target)
        fig_comp = create_feature_comparison(df1, 'BMI', 'Diabetes_012', "BMI distributions across Diabetes categories")
        if fig_comp:
            fig_comp.write_html(os.path.join(d1_out_dir, "feature_comparison.html"), include_plotlyjs="cdn")
            
        print("  - Saved HTML files in: results/visualizations/dataset1/")
    else:
        print(f"Error: Dataset 1 not found at {dataset1_path}")
        
    # --------------------------------------------------
    # Process Dataset 2
    # --------------------------------------------------
    if os.path.exists(dataset2_path):
        print("\nProcessing Dataset 2 (Lifestyle & Clinical)...")
        df2 = pd.read_csv(dataset2_path)
        d2_out_dir = os.path.join(output_dir, "dataset2")
        os.makedirs(d2_out_dir, exist_ok=True)
        
        # 1. Target Distribution
        fig_target_2 = create_target_distribution(df2, 'diagnosed_diabetes', "Distribution of Diabetes Diagnosis (Dataset 2: Clinical)")
        if fig_target_2:
            fig_target_2.write_html(os.path.join(d2_out_dir, "target_distribution.html"), include_plotlyjs="cdn")
            
        # 2. Correlation Heatmap
        num_cols_2 = [
            'Age', 'physical_activity_minutes_per_week', 'diet_score', 
            'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 
            'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
            'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 
            'triglycerides', 'glucose_fasting', 'glucose_postprandial', 
            'insulin_level', 'hba1c'
        ]
        fig_corr_2 = create_correlation_heatmap(df2, num_cols_2, "Correlation Matrix of Clinical Parameters")
        if fig_corr_2:
            fig_corr_2.write_html(os.path.join(d2_out_dir, "correlation_heatmap.html"), include_plotlyjs="cdn")
            
        # 3. Feature Distribution (HbA1c)
        fig_dist_2 = create_histogram(df2, 'hba1c', 'diagnosed_diabetes', "Distribution of HbA1c grouped by Diagnosis status")
        if fig_dist_2:
            fig_dist_2.write_html(os.path.join(d2_out_dir, "feature_distributions.html"), include_plotlyjs="cdn")
            
        # 4. Feature Comparison (HbA1c Violin Plot vs Target)
        fig_comp_2 = create_feature_comparison(df2, 'hba1c', 'diagnosed_diabetes', "HbA1c distributions across Diagnosis groups")
        if fig_comp_2:
            fig_comp_2.write_html(os.path.join(d2_out_dir, "feature_comparison.html"), include_plotlyjs="cdn")
            
        print("  - Saved HTML files in: results/visualizations/dataset2/")
    else:
        print(f"Error: Dataset 2 not found at {dataset2_path}")
        
    print("\n==================================================")
    print("VISUALIZATION PIPELINE COMPLETED SUCCESSFULLY")
    print("==================================================")
