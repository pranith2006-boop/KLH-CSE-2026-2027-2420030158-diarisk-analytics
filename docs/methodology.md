# Research Methodology: Data Preprocessing and Exploratory Data Analysis

This document describes the data preprocessing and Exploratory Data Analysis (EDA) methodologies used in the Diarisk Analytics project.

## 1. Data Quality and Integrity Checks
Before performing any analysis or modeling, both datasets undergo automatic validation checks:
- **Missing Value Detection**: Automated checks inspect the dataset for null values. Though both datasets contain zero missing entries in their raw form, the preprocessing pipelines are designed with robust imputers to handle any future missing data.
- **Duplicate Detection**: Duplicate rows are detected by checking for exact matches across all columns.
  - *Dataset 1*: 23,899 duplicate rows (9.42%) were detected and removed before split to prevent identical data points from leaking between training and validation groups.
  - *Dataset 2*: 0 duplicate rows were detected.
- **Invalid Value Detection**: Medical variables are validated using clinical constraints (e.g., checking for negative age or blood pressure, or impossible values such as BMI below 10 or above 80).
  - *Outliers*: Outliers in continuous variables (like glucose levels, insulin, HbA1c, or BMI) are identified using the Interquartile Range (IQR) method (outside $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$). 
  - *Outlier Treatment Decision*: Rather than deleting outliers, we **retain** them. In clinical contexts, extreme physiological readings (e.g., high fasting glucose or extreme BMI) are highly predictive, genuine indicators of diabetes risk rather than measurement errors. Removing them would discard critical diagnostic signals.

## 2. Preprocessing & Pipeline Design
To ensure rigorous validation and avoid data leakage, all feature transformations are implemented using Scikit-Learn `ColumnTransformer` and `Pipeline` objects. The workflow proceeds as follows:

```
Raw Dataset ──> Train/Test Split (80/20) ──> ColumnTransformer (Fitted on Train Only)
                                                      │
                                                      ├──> Numerical: Median Impute ──> StandardScaler
                                                      └──> Categorical: Frequent Impute ──> One-Hot Encoder
```

### Preprocessing Steps:
1. **Train/Test Split**: Both datasets are split into 80% training and 20% testing sets using stratification.
   - For Dataset 1, stratification is based on the multi-class target `Diabetes_012`.
   - For Dataset 2, stratification is based on the binary target `diagnosed_diabetes`.
   - A fixed random seed (`random_state=42`) is used to ensure reproducibility.
2. **Feature Scaling (Numerical)**: Scaled using `StandardScaler` to ensure zero mean and unit variance. Scaling parameters (mean and standard deviation) are fitted **only** on the training split and then applied to transform both train and test splits.
3. **Categorical Encoding**: All categorical columns (nominal, binary, and ordinal variables coded as numeric indices in survey data) are one-hot encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. This prevents linear models from assuming numeric linearity for discrete survey codes (e.g., Age 1–13 categories in Dataset 1).
4. **Data Leakage Mitigation**: Other potential target candidates in Dataset 2 (`diabetes_stage` and `diabetes_risk_score`) are explicitly **dropped** from the features ($X$) prior to the split, preventing target leakage during training.

## 3. Exploratory Data Analysis (EDA) and Statistical Methodology
Each dataset is analyzed independently to reveal underlying structures and guide downstream model selection.

### Descriptive Statistics:
- **Numerical Features**: Evaluated using central tendency (mean, median), dispersion (standard deviation, variance, range, IQR), and percentile distributions (25th, 50th, 75th).
- **Categorical Features**: Analyzed using category counts, frequency tables, and relative percentages.

### Distribution Analysis:
- Feature distributions are evaluated using Fisher-Pearson skewness and kurtosis.
- Features are categorized as:
  - *Approximately Symmetric*: $|skewness| < 0.5$
  - *Right-Skewed (Positive)*: $skewness \ge 0.5$ (e.g., insulin levels, triglycerides)
  - *Left-Skewed (Negative)*: $skewness \le -0.5$
- Log or power transformations are suggested for highly skewed continuous clinical features in downstream linear models.

### Correlation Analysis:
- Linear associations are evaluated using the Pearson correlation coefficient ($r$).
- Monotonic or non-linear relationships are cross-evaluated using Spearman's rank correlation coefficient ($\rho$).
- Strong correlations are flagged ($|r| \ge 0.5$) to check for multicollinearity, which might require regularization in downstream models.

### Statistical Significance Testing:
All statistical tests are executed at a significance level of $\alpha = 0.05$ to check for differences across target classes.
- **Numerical Features vs. Binary Target** (Dataset 2): Evaluated using the non-parametric **Mann-Whitney U test**, as clinical traits (like insulin, glucose, or physical activity) exhibit non-normal, skewed distributions.
- **Numerical Features vs. Multi-class Target** (Dataset 1): Evaluated using the **Kruskal-Wallis test** (a non-parametric alternative to one-way ANOVA) to compare the medians of BMI and health days across the three diabetes classes (0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes).
- **Categorical Features vs. Categorical Target** (Both datasets): Evaluated using the **Chi-Square Test of Independence** to test for significant associations between categorical factors (smoking, hypertension, demographic groups) and diabetes status.
- *Causation Disclaimer*: All reports explicitly adhere to the scientific rule that correlation and statistical significance do not imply a causal relationship.

## 4. Interactive Data Visualization Methodology
To make the analytical insights accessible and ready for web dashboard integration, we built an interactive visualization system using **Plotly (Express and Graph Objects)**.

### Why Plotly is Used:
- **Interactivity**: Unlike static libraries (Matplotlib/Seaborn), Plotly allows users to zoom, pan, toggle legend categories to filter data dynamically, and inspect individual data points via hover tooltips. This is critical for discovering fine-grained patterns in large datasets (e.g. 250k+ records).
- **Web Dashboard Readiness**: Plotly charts are rendered in HTML/JavaScript, making them drop-in components for our downstream Flask web dashboard.
- **Responsive Layout**: Charts adapt dynamically to screen size variations, supporting a premium presentation.

### Chart Selections by Feature Scale:
1. **Target Distribution**: A Donut chart is used to display the count and percentage of target classes. Donut charts represent categorical part-to-whole relationships cleanly and allow class distributions to be scanned immediately.
2. **Numerical Distributions**: Histograms represent the frequency shapes (skewness, modality, tails). We add vertical lines representing mean (red, dashed) and median (green, dotted) to visually check skewness. Grouping by target overlays class distributions.
3. **Feature vs. Target comparison**: Violin plots with internal box plots are used to visualize how continuous values (such as HbA1c or BMI) differ across target classes. This represents the median, interquartile range (IQR), range, and probability density shape in a single visual.
4. **Multi-variable Interactions**: A 2D Scatter Plot compares two numerical features (e.g. BMI vs Glucose) with points color-coded by the target variable. This reveals cluster boundaries.
5. **Correlation Heatmaps**: A diverging color heatmap (`RdBu` scale from -1 to 1) visualizes linear associations, annotated with Pearson coefficients.
6. **Categorical / Demographic / Lifestyle features**: Grouped bar charts compare counts of discrete categories (e.g., smoking status, physical activity) split by target class groups.

### Technical Compatibility Workaround:
To bypass a compatibility mismatch between Pandas 3.0.3 and Plotly Express 5.17.0 (where grouping by a list parameter causes `get_group` KeyErrors due to scalar vs tuple key lookup differences), we implemented a python-level monkeypatch to `pandas.core.groupby.groupby.GroupBy.get_group`. The patched method automatically handles key representation mismatches, ensuring seamless execution across both datasets.

## 5. Machine Learning Workflow

Phase 5 trains six baseline classifiers independently for each dataset: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, XGBoost, and CatBoost. The raw data is deduplicated using the Phase 2 pipeline, split into 80% training and 20% test data with `random_state=42`, and stratified by the target. Each saved model is a complete scikit-learn pipeline, so imputation, scaling, and one-hot encoding are fitted on training data and reused unchanged for test data and later predictions.

The models use median imputation and `StandardScaler` for numerical features, frequent-value imputation and one-hot encoding for categorical features, and retain clinically meaningful outliers. Dataset 1 uses `class_weight='balanced'` for supported scikit-learn classifiers and balanced CatBoost weights because its target is highly imbalanced. Dataset 2 is comparatively balanced, so its baseline models use the default class weighting. No oversampling is applied.

Model evaluation is performed only after fitting. Accuracy, macro precision, macro recall, macro F1, and multiclass one-vs-rest ROC-AUC are reported for Dataset 1; binary versions are reported for Dataset 2. Confusion matrices and classification reports are saved per model, and ROC curves are plotted together for each dataset. Five-fold `StratifiedKFold` cross-validation is run on training data only. Because probability-enabled SVM training is expensive on these datasets, the SVM baseline and its cross-validation use a fixed stratified training subset; the untouched test set remains reserved for final evaluation.

The selection rule is deterministic: maximize test ROC-AUC, then use recall and F1-score as tie-breakers. Based on the generated baseline results, XGBoost is selected for Dataset 1 (ROC-AUC 0.7656), while the Decision Tree is selected for Dataset 2 (ROC-AUC 0.9450). These are baseline findings, not claims of clinical performance.

## 6. Phase 6 Evaluation and Selection

Phase 6 evaluates the saved pipelines on the same held-out test splits without retraining. The comparison tables are rounded to four decimal places for reporting, while ranking uses the unrounded calculated values. Dataset 1 uses macro-averaged precision, recall, and F1 with multiclass one-vs-rest ROC-AUC; Dataset 2 uses binary metrics and ROC-AUC. Confusion matrices use the observed target labels rather than assumed display labels.

Models are ranked lexicographically by ROC-AUC, recall, F1-score, and accuracy. This makes the selection reproducible and keeps discrimination primary while treating positive-case coverage and balanced classification as secondary considerations. Error analysis is performed only for the selected model in each dataset and reports TN, FP, FN, TP, false-positive rate, and false-negative rate using the highest observed class as the positive class for the multiclass Dataset 1 summary. These rates are descriptive evaluation statistics and are not clinical conclusions.

Plotly HTML outputs provide interactive metric, ROC, ranking, and feature-importance comparisons. The selected model JSON files retain the joblib path, embedded preprocessing note, transformed feature names, target name, and selection rule for Phase 7 SHAP/LIME preparation. SHAP and LIME are intentionally not implemented in this phase.

## 7. Explainable AI

Phase 7 explains the selected Phase 6 pipelines without changing or retraining them. SHAP uses `TreeExplainer` on the fitted tree classifiers after applying each pipeline's fitted preprocessing transformer. Global mean absolute SHAP values describe relative model influence, while signed local values show whether each transformed feature moved a prediction toward or away from the selected class. LIME uses `LimeTabularExplainer` on the same transformed feature matrix and the classifier's `predict_proba` function to produce local surrogate explanations.

Representative dataset records include correct positive, correct negative, false-positive, and false-negative cases when available. Explanations describe model behavior only; they do not prove causation or constitute medical diagnoses. One-hot encoded features are retained as readable source-category names (for example, `gender_Male`), but are not automatically recombined into a single categorical feature. SHAP and LIME may disagree because their explanation methodologies differ.

## 8. Prediction Storage and Privacy

Phase 9 stores only the fields needed for prediction history: dataset identifier, model identifier, predicted class, maximum class probability, an analytical risk-level label equal to the predicted class, submitted feature values, UTC timestamp, and XAI availability. Source CSV records are never uploaded, and no names, contact information, credentials, or other unnecessary personal data are collected. MongoDB Atlas is configured through `MONGO_URI` and `DATABASE_NAME` environment variables. A lightweight ping runs at application startup when configured; connection failure does not block analytics or prediction pages. History retrieval is paginated and filtered server-side.

