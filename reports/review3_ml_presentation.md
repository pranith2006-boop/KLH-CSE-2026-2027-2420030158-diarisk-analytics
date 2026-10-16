# Review 3: Machine Learning Presentation & Technical Report

**Course**: Data Analytics and Visualization — 24DSB3101  
**Project Title**: Diabetes Risk Prediction and Model Evaluation (DiaRisk Analytics)  
**Academic Year**: 2026–2027  
**Team Members**:
- Modalla Pranith Reddy (2420030158)
- P.M. Pranitha (2420030277)
- I. Hema Sri (2420030580)  
**Project Supervisor / Guide**: Garimidi Siva Sree

---

## Executive Summary

This report documents the complete Machine Learning (ML) architecture, statistical validation, preprocessing pipelines, model benchmarks, and analytical conclusions for **Review 3**. 

The project evaluates two distinct healthcare datasets:
1. **CDC BRFSS 2015 Health Indicators** (`diabetes_012_health_indicators_BRFSS2015.csv`): Large-scale epidemiological survey with 229,781 clean records, 21 predictors, and a severe 46.15:1 multiclass imbalance (`Diabetes_012`).
2. **Clinical & Lifestyle Profile Dataset** (`Diabetes_and_LifeStyle_Dataset .csv`): Clinical lab and biometric cohort with 97,297 records, 28 predictors, and a balanced binary outcome (`diagnosed_diabetes`).

Six machine learning algorithms spanning diverse inductive biases were trained and evaluated on each dataset: **Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), XGBoost, and CatBoost**.

---

## 1. Data Understanding & Feature Type Identification

A fundamental requirement of healthcare analytics is understanding the semantic and mathematical nature of features to prevent invalid operations (e.g., treating categorical codes as continuous measurements).

### 1.1 Dataset 1: CDC BRFSS 2015 Health Indicators
- **Raw Dimensions**: 253,680 rows × 22 columns
- **Clean Dimensions (Post-Deduplication)**: 229,781 rows × 22 columns
- **Target Variable**: `Diabetes_012` (Multiclass discrete target)
  - `0.0`: No Diabetes (82.71% clean / 190,071 rows)
  - `1.0`: Prediabetes (2.01% clean / 4,631 rows)
  - `2.0`: Diabetes (15.27% clean / 35,079 rows)
  - **Imbalance Ratio**: **46.15 : 1** (Extreme minority representation for prediabetes)

| Feature Classification | Count | Columns | Description & Range |
| :--- | :---: | :--- | :--- |
| **Numerical (Continuous/Discrete Counts)** | 3 | `BMI`, `MentHlth`, `PhysHlth` | • `BMI`: Body Mass Index (12.0 – 98.0)<br>• `MentHlth`: Days of poor mental health in past 30 days (0 – 30)<br>• `PhysHlth`: Days of poor physical health in past 30 days (0 – 30) |
| **Binary Categorical Flags (0 / 1)** | 14 | `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex` | Coded as 0.0 or 1.0 (e.g., `HighBP`: 1=Yes, 0=No; `Sex`: 1=Male, 0=Female). |
| **Ordinal Categorical Scales** | 4 | `GenHlth`, `Age`, `Education`, `Income` | • `GenHlth`: Self-rated health scale (1=Excellent to 5=Poor)<br>• `Age`: 13 discrete age brackets (1=18-24 to 13=80+)<br>• `Education`: 6 educational attainment tiers (1-6)<br>• `Income`: 8 annual household income brackets (1-8) |

### 1.2 Dataset 2: Clinical & Lifestyle Profile Dataset
- **Raw Dimensions**: 97,297 rows × 31 columns
- **Clean Dimensions**: 97,297 rows × 29 columns (after dropping 2 target leakage candidates)
- **Target Variable**: `diagnosed_diabetes` (Binary outcome: Class 0 = 40.0% / 38,915 rows; Class 1 = 60.0% / 58,382 rows; Imbalance Ratio = 1.50 : 1).
- **Target Leakage Mitigation**: Dropped `diabetes_stage` and `diabetes_risk_score` prior to splitting to prevent circular training bias.

| Feature Classification | Count | Features | Clinical Range & Description |
| :--- | :---: | :--- | :--- |
| **Biochemical & Lab Biomarkers** | 8 | `glucose_fasting`, `glucose_postprandial`, `hba1c`, `insulin_level`, `cholesterol_total`, `hdl_cholesterol`, `ldl_cholesterol`, `triglycerides` | • `glucose_fasting`: 60 – 172 mg/dL<br>• `glucose_postprandial`: 70 – 287 mg/dL<br>• `hba1c`: 4.0 – 9.8% (Gold-standard diagnostic test)<br>• `insulin_level`: 2.0 – 32.22 $\mu\text{U/mL}$<br>• Lipid panel: Total, HDL, LDL, Triglycerides |
| **Clinical Vitals & Anthropometrics** | 5 | `bmi`, `waist_to_hip_ratio`, `systolic_bp`, `diastolic_bp`, `heart_rate` | • `bmi`: 15.0 – 39.2 $\text{kg/m}^2$<br>• `waist_to_hip_ratio`: 0.67 – 1.06<br>• Blood Pressure: Systolic (90-179 mmHg), Diastolic (50-110 mmHg)<br>• Resting Heart Rate: 40 – 105 bpm |
| **Lifestyle & Demographics (Numerical)** | 5 | `Age`, `physical_activity_minutes_per_week`, `diet_score`, `sleep_hours_per_day`, `screen_time_hours_per_day` | • `Age`: 19 – 90 years<br>• `physical_activity`: 0 – 833 min/week<br>• `diet_score`: 0.0 – 10.0 scale<br>• Sleep: 3.0 – 10.0 hrs; Screen time: 0.5 – 16.8 hrs |
| **Categorical Demographics & Medical History** | 10 | `gender`, `ethnicity`, `education_level`, `income_level`, `employment_status`, `smoking_status`, `alcohol_consumption_per_week`, `family_history_diabetes`, `hypertension_history`, `cardiovascular_history` | • Demographic strings: 5 ethnicities, 3 genders, 4 education tiers, 5 income tiers, 4 employment states, 3 smoking states<br>• Medical history binary flags (0/1): Family history, hypertension, cardiovascular history |

---

## 2. Data Cleaning & Missing Value Imputation

### 2.1 Missing Value Audit & Production Imputation Strategy
- **Audit Results**: Inspection via `.isnull().sum()` confirmed **0 missing values** across both raw datasets.
- **Biologically Impossible Values**: Checked clinical parameters for impossible zero or negative entries (`glucose == 0`, `bp == 0`, `bmi == 0`, `hba1c == 0`). Zero suspicious entries were found.
- **Production Imputer Design**: To ensure future data inputs or real-time clinical API calls can handle missing data gracefully, automated pipelines embed:
  1. **Numerical Imputation**: `SimpleImputer(strategy='median')` — Chosen because the median is non-parametric and resistant to extreme skewness and outliers in clinical measures.
  2. **Categorical Imputation**: `SimpleImputer(strategy='most_frequent')` — Chosen to retain discrete categorical integrity without introducing fractional floating-point categories.

### 2.2 Deduplication
- **Dataset 1**: **23,899 duplicate records (9.42%)** were detected and removed using `df.drop_duplicates()`. Removing identical survey responses prior to splitting prevents identical samples from leaking between training and test sets.
- **Dataset 2**: 0 duplicate records detected.

### 2.3 Elimination of Target Leakage
- In Dataset 2, `diabetes_stage` and `diabetes_risk_score` were dropped. If retained, models would learn trivial circular rules rather than discovering predictive relationships from raw clinical vitals.

---

## 3. Statistical Checks: Normality Distribution & Outlier Handling

### 3.1 Normality Checks (Skewness, Kurtosis, and Distribution Shapes)
To determine whether parametric (t-test, ANOVA) or non-parametric statistical methods were appropriate, continuous variables were analyzed for normality.

#### Dataset 1 Continuous Features
- `BMI`: Mean = 28.38, Median = 27.0, Skewness = **+1.21** (Moderate right skew), Kurtosis = 2.65.
- `MentHlth`: Mean = 3.18, Median = 0.0, Skewness = **+2.23** (Severe right skew, zero-inflated with secondary spike at 30 days).
- `PhysHlth`: Mean = 4.24, Median = 0.0, Skewness = **+1.98** (Severe right skew, zero-inflated with secondary spike at 30 days).
- **Normality Conclusion**: Severe departures from Gaussian normality ($p < 0.001$). Parametric assumptions are violated.

#### Dataset 2 Continuous Features
- **Approximately Symmetric ($|\text{Skewness}| < 0.5$)**: `Age` (skew: 0.01), `bmi` (skew: 0.12), `systolic_bp` (skew: 0.28), `diastolic_bp` (skew: 0.19), `heart_rate` (skew: 0.08), `hba1c` (skew: 0.34), `glucose_fasting` (skew: 0.41).
- **Positively Skewed ($\text{Skewness} \ge 0.5$)**: `physical_activity_minutes_per_week` (skew: 1.14, long right tail up to 833 minutes), `insulin_level` (skew: 0.62), `triglycerides` (skew: 0.58).

### 3.2 Non-Parametric Hypothesis Testing
Because normality assumptions failed, rigorous non-parametric tests were applied ($\alpha = 0.05$):
- **Dataset 1 (Multiclass Target)**: **Kruskal-Wallis H-Test** was used instead of one-way ANOVA.
  - `BMI`: $H = 14,130.75, p < 10^{-300}$ (Statistically significant difference across target classes).
  - `PhysHlth`: $H = 6,661.88, p < 10^{-300}$ (Statistically significant).
  - `MentHlth`: $H = 528.91, p = 1.41 \times 10^{-115}$ (Statistically significant).
  - Categorical variables (`HighBP`, `HighChol`, `GenHlth`, etc.) tested via **$\chi^2$ Test of Independence**: All 18 showed $p < 10^{-15}$.
- **Dataset 2 (Binary Target)**: **Mann-Whitney U Test** was used instead of independent two-sample t-test.
  - `hba1c`: $U = 1.54 \times 10^8, p < 10^{-300}$ (Extreme separation between diagnosed and undiagnosed).
  - `glucose_postprandial`: $U = 2.41 \times 10^8, p < 10^{-300}$.
  - `glucose_fasting`: $U = 4.36 \times 10^8, p < 10^{-300}$.
  - `bmi`: $U = 1.01 \times 10^9, p = 1.18 \times 10^{-195}$.
  - `systolic_bp`: $U = 1.01 \times 10^9, p = 1.55 \times 10^{-193}$.
  - **$\chi^2$ Independence**: `family_history_diabetes` ($\chi^2 = 3849.95, p < 10^{-300}$), `cardiovascular_history` ($\chi^2 = 86.45, p = 1.43 \times 10^{-20}$). Demographic variables (`gender`, `ethnicity`, `smoking`) showed no significant target association ($p > 0.05$).

### 3.3 Outlier Detection & Handling Strategy
- **Detection Method**: Interquartile Range (IQR) rule: Outlier if $x < Q1 - 1.5 \times IQR$ or $x > Q3 + 1.5 \times IQR$.
  - Dataset 1: `BMI` (9,847 outliers, 3.88%), `MentHlth` (36,208 outliers, 14.27%), `PhysHlth` (40,949 outliers, 16.14%).
  - Dataset 2: `physical_activity` (3,115 outliers, 3.20%), `bmi` (729, 0.75%), `hba1c` (599, 0.62%), `glucose_fasting` (722, 0.74%), `insulin_level` (322, 0.33%).
- **Handling Decision: Complete Outlier Retention (No Truncation / No Deletion)**.
- **Justification**:
  1. *Clinical Diagnostic Validity*: In medicine, extreme values (e.g., HbA1c = 9.8%, Fasting Glucose = 170 mg/dL, BMI = 45) are not measurement noise or errors; they represent the **primary biological pathology of uncontrolled diabetes**. Truncating or clipping them would remove the exact diagnostic signals machine learning models must learn.
  2. *Algorithmic Robustness*: Tree-based ensemble models (Random Forest, XGBoost, CatBoost) perform partition-based split selections on order statistics, making them mathematically invariant to monotonic feature scaling and extreme outliers.

---

## 4. Data Preprocessing: Standardisation vs. Normalisation

### 4.1 Train/Test Stratification
- Stratified 80/20 train/test split with `random_state=42`.
- Preserves the 46.15:1 class ratio in Dataset 1 across train and test sets to guarantee reproducible, unskewed evaluation.

### 4.2 Standardisation vs. Normalisation Selection & Justification
- **Selected Method**: `StandardScaler` (Z-Score Standardization: $z = \frac{x - \mu}{\sigma}$).
- **Justification over Min-Max Normalisation ($[0, 1]$)**:
  1. *Resistance to Outlier Compression*: Min-Max normalization relies on $x_{min}$ and $x_{max}$. A single extreme clinical value (e.g., BMI = 98.0) squashes 99% of normal patient values into a narrow sub-band between 0.1 and 0.25, eliminating feature variance. `StandardScaler` centers data at $\mu = 0$ with $\sigma = 1$ without bound-compression.
  2. *Convergence for Linear & Margin Models*: Gradient descent in Logistic Regression and margin calculation in Support Vector Machines require standardized inputs with spherical loss contours for optimal convergence.
  3. *Zero-Mean Preservation*: Standard scaling maintains negative and positive deviations relative to the cohort mean, aligning with clinical concept of "below-average" or "above-average" physiological indicators.

### 4.3 Categorical Encoding
- **Selected Method**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Justification**:
  - Ordinal features (e.g., `Age` brackets 1–13, `GenHlth` 1–5 in Dataset 1) possess non-linear risk intervals. Label encoding would force linear models to assume that shifting from Age bracket 1 to 2 produces the identical risk change as shifting from bracket 12 to 13. One-hot encoding allows each category to have an independent regression coefficient or tree split.
  - Dimension expansion: Dataset 1 expands to **69 features**; Dataset 2 expands to **65 features**.

### 4.4 Data Leakage Prevention via Scikit-Learn Pipelines
- All preprocessing transformations are encapsulated within a `ColumnTransformer` embedded inside an end-to-end `Pipeline`.
- Scaling parameters ($\mu, \sigma$) and one-hot categories are learned **strictly from `X_train`** and applied unchanged to `X_test`.

---

## 5. Model Building & Algorithm Selection

### 5.1 Evaluated Algorithms & Justifications

| Algorithm | Paradigm & Inductive Bias | Rationale for Selection |
| :--- | :--- | :--- |
| **Logistic Regression** | Parametric Generalized Linear Model | Establishes an interpretable, probabilistic baseline. Reveals linear log-odds relationships. |
| **Decision Tree (CART)** | Non-parametric Hierarchical Rule Splits | Models non-linear interactions; replicates clinical diagnosis flowcharts with interpretable thresholds. |
| **Random Forest** | Bagging Ensemble of De-correlated Trees | Reduces variance, prevents overfitting, and captures complex feature interactions across 100 trees. |
| **Support Vector Machine (SVM)** | Maximum Margin Kernel Classifier | Finds optimal separating hyperplanes in high-dimensional one-hot encoded feature spaces. |
| **XGBoost** | Gradient Tree Boosting via 2nd-Order Taylor | State-of-the-art tabular performance; minimizes pseudo-residuals with explicit $L_1$/$L_2$ tree regularization. |
| **CatBoost** | Symmetric Oblivious Decision Trees | Overcomes prediction shift and target leakage; highly effective on tabular data with categorical indicators. |

### 5.2 Imbalance Mitigation Strategy
- **Dataset 1**: `class_weight='balanced'` applied to Logistic Regression, Decision Tree, Random Forest, and SVM; `auto_class_weights='Balanced'` applied to CatBoost.
- **Dataset 2**: `scale_pos_weight = 0.67` applied to XGBoost to account for the 60:40 target ratio.

---

## 6. Model Analysis, Metrics & Key Findings

### 6.1 Evaluation Strategy & Ranking Rule
- Primary ranking metric: **ROC-AUC** (discriminative capacity independent of class thresholds).
- Secondary decisive metrics: **Recall** (minimizing dangerous false negatives) and **F1-Score**.
- 5-Fold Stratified Cross-Validation performed on training sets to confirm generalization stability.

### 6.2 Dataset 1 (BRFSS 2015) Model Comparison

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC | 5-Fold CV Mean | 5-Fold CV Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Selected Best)** | **0.8363** | **0.4729** | **0.3914** | **0.4015** | **0.7656** | **0.7609** | **0.0028** |
| Logistic Regression | 0.6253 | 0.4466 | 0.5145 | 0.4243 | 0.7611 | 0.7557 | 0.0018 |
| Random Forest | 0.6508 | 0.4389 | 0.5048 | 0.4278 | 0.7557 | 0.7455 | 0.0045 |
| CatBoost | 0.6490 | 0.4388 | 0.5007 | 0.4260 | 0.7398 | 0.7348 | 0.0024 |
| Support Vector Machine (SVM) | 0.4246 | 0.4003 | 0.4404 | 0.3024 | 0.7388 | 0.7405 | 0.0076 |
| Decision Tree | 0.5929 | 0.4300 | 0.4839 | 0.4003 | 0.7216 | 0.7145 | 0.0058 |

#### Dataset 1 Confusion Matrix Analysis (XGBoost, Held-Out Test Set)
```
                  Predicted 0 (No)   Predicted 1 (Pre)   Predicted 2 (Diab)
Actual 0 (No)          37,030                0                   982
Actual 1 (Pre)            836                0                    90
Actual 2 (Diab)         5,615                0                 1,404
```
- **Error Breakdown**: True Negatives (Non-diabetes) = 37,030. False Positive Rate = **2.75%** (1,072 / 38,938).
- **Clinical Insight**: Macro accuracy is high (83.63%) because majority non-diabetics are correctly identified (precision: 85.16%, recall: 97.42%). However, prediabetes (Class 1) has 0 recall due to its extreme 1.83% rarity, demonstrating the classic challenge of extreme multiclass imbalance.
- **Top Predictive Features**: `HighBP_0.0` (23.13%), `GenHlth_1.0` (8.05%), `GenHlth_2.0` (6.75%), `HighChol_0.0` (4.71%), `DiffWalk_0.0` (3.66%), `Age` brackets.

---

### 6.3 Dataset 2 (Clinical & Lifestyle) Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV Mean | 5-Fold CV Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Selected Best)** | **0.9195** | **0.9971** | **0.8685** | **0.9283** | **0.9450** | **0.9434** | **0.0017** |
| CatBoost | 0.9202 | 0.9995 | 0.8674 | 0.9288 | 0.9437 | 0.9430 | 0.0011 |
| Random Forest | 0.9200 | 0.9990 | 0.8676 | 0.9287 | 0.9434 | 0.9438 | 0.0015 |
| XGBoost | 0.9191 | 0.9970 | 0.8677 | 0.9279 | 0.9409 | 0.9414 | 0.0015 |
| Logistic Regression | 0.8599 | 0.8733 | 0.8966 | 0.8848 | 0.9331 | 0.9337 | 0.0011 |
| Support Vector Machine (SVM) | 0.8821 | 0.9320 | 0.8667 | 0.8982 | 0.9313 | 0.9345 | 0.0035 |

#### Dataset 2 Confusion Matrix Analysis (Decision Tree, Held-Out Test Set)
```
                  Predicted 0 (No Diabetes)   Predicted 1 (Diabetes)
Actual 0 (No)               7,753                        30
Actual 1 (Diabetes)         1,536                    10,141
```
- **Error Breakdown**:
  - True Negatives: 7,753 | True Positives: 10,141
  - False Positives: **30** (FPR: **0.38%**)
  - False Negatives: **1,536** (FNR: **13.15%**)
- **Clinical Insight**: Decision Tree achieves an extraordinary precision of **99.71%**, meaning when it flags diabetes, it is virtually certain ($< 0.4\%$ false alarms).
- **Top Predictive Features**:
  1. `hba1c`: **95.01%** importance.
  2. `glucose_fasting`: **3.92%** importance.
  3. `family_history_diabetes`: **0.25%** importance.
  - This perfectly validates clinical endocrinology: American Diabetes Association (ADA) guidelines establish HbA1c $\ge 6.5\%$ and fasting glucose $\ge 126\text{ mg/dL}$ as the definitive diagnostic thresholds.

---

## 7. Conclusions & Final Recommendations for Review 3

1. **Robust Pipeline Architecture**: End-to-end integration using Scikit-Learn `ColumnTransformer` and `Pipeline` prevents any data snooping or parameter leakage between splits.
2. **Clinical Alignment**: Model feature importances independently discovered clinically established diagnostic indicators: `hba1c` (95.01%) and `glucose_fasting` (3.92%) for clinical profiles; `HighBP`, `GenHlth`, and `HighChol` for epidemiological surveys.
3. **Imbalance Insights**: High accuracy can mask minority class omission in imbalanced healthcare datasets (as in Dataset 1's prediabetes class), highlighting the necessity of evaluating ROC-AUC, Precision-Recall, and confusion matrices rather than accuracy alone.
4. **Readiness for Explainability (XAI)**: With XGBoost and Decision Tree established as the optimal baseline models, the pipelines are fully coupled with downstream SHAP (Shapley Additive Explanations) and LIME (Local Interpretable Model-agnostic Explanations) to provide clinicians with transparent, sample-level decision attribution.
