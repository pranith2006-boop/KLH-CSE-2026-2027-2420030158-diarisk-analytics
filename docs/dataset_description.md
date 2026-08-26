# Dataset Description: Diabetes indicators and Clinical Profiles

This document provides a detailed description of the structures, target variables, features, and analytical findings for the two datasets used in the Diarisk Analytics project.

## 1. Dataset 1: BRFSS 2015 Health Indicators
- **Filename**: `diabetes_012_health_indicators_BRFSS2015.csv`
- **Data Source**: Behavioral Risk Factor Surveillance System (BRFSS) 2015, a large-scale survey conducted by the CDC.
- **Records (Raw)**: 253,680
- **Duplicate Rows**: 23,899 (9.42% of the dataset). These were removed in preprocessing, resulting in **229,781** final records.
- **Features**: 21 features, all stored as `float64` in the raw CSV.
- **Target Variable**: `Diabetes_012`
  - *Class 0.0 (No Diabetes)*: 213,703 rows (84.24% of raw, 82.71% of clean)
  - *Class 1.0 (Prediabetes)*: 4,631 rows (1.83% of raw, 2.01% of clean)
  - *Class 2.0 (Diabetes)*: 35,346 rows (13.93% of raw, 15.27% of clean)
  - *Imbalance Ratio*: **46.15:1** (extreme imbalance between majority class 0.0 and minority class 1.0).

### Key Features and Variables
1. **Numerical (3)**:
   - `BMI` (Body Mass Index, range: 12.0–98.0)
   - `MentHlth` (Days of poor mental health in the last 30 days, range: 0–30)
   - `PhysHlth` (Days of poor physical health in the last 30 days, range: 0–30)
2. **Categorical (18)**:
   - *Binary flags (0 or 1)*: `HighBP` (high blood pressure), `HighChol` (high cholesterol), `CholCheck` (cholesterol check in 5 years), `Smoker` (smoked >= 100 cigarettes), `Stroke` (stroke history), `HeartDiseaseorAttack` (coronary heart disease/infarction), `PhysActivity` (physical activity in past 30 days), `Fruits` (consume fruit >= 1/day), `Veggies` (consume vegetables >= 1/day), `HvyAlcoholConsump` (heavy drinkers), `AnyHealthcare` (any health coverage), `NoDocbcCost` (could not afford doctor), `DiffWalk` (difficulty walking/climbing stairs), `Sex` (0 = Female, 1 = Male).
   - *Ordinal scales*: `GenHlth` (general health rating: 1 = excellent, 5 = poor), `Age` (13 age brackets: 1 = 18–24, 13 = 80+), `Education` (1–6 scale), `Income` (1–8 scale).

### Data Quality & EDA Observations
- **Missing Values**: 0 missing values.
- **Outliers**: High outliers exist in `MentHlth` (14.27%) and `PhysHlth` (16.14%) because the distribution is highly bi-modal (most respondents report 0 days, while a secondary peak exists at 30 days).
- **Correlations**: Very weak linear relationships overall. The strongest correlation exists between `PhysHlth` and `MentHlth` ($r = 0.35$).
- **Statistical Association**: All 21 features exhibit highly significant differences or associations across target groups ($p < 0.05$) due to the very large sample size.

---

## 2. Dataset 2: Lifestyle and Clinical Profile
- **Filename**: `Diabetes_and_LifeStyle_Dataset .csv`
- **Data Source**: Demographic and clinical health measurements of individuals.
- **Records (Raw & Clean)**: 97,297 (0 duplicate rows).
- **Features**: 28 features (after dropping other target candidates to prevent leakage).
- **Target Variable**: `diagnosed_diabetes` (Binary classification target)
  - *Class 1 (Diagnosed)*: 58,382 rows (60.00%)
  - *Class 0 (Undiagnosed/No Diabetes)*: 38,915 rows (40.00%)
  - *Imbalance Ratio*: **1.50:1** (relatively balanced).
- **Dropped Target Candidates (Leakage Prevention)**:
  - `diabetes_stage` (Multi-class: Type 2, Pre-Diabetes, No Diabetes, Gestational, Type 1)
  - `diabetes_risk_score` (Continuous risk prediction: 2.7–67.2)

### Key Features and Variables
1. **Numerical (18)**:
   - *Demographics*: `Age` (range: 19–90).
   - *Clinical/Vitals*: `bmi` (range: 15.0–39.2), `waist_to_hip_ratio` (range: 0.67–1.06), `systolic_bp` (range: 90–179), `diastolic_bp` (range: 50–110), `heart_rate` (range: 40–105).
   - *Biochemistry/Labs*: `cholesterol_total` (range: 100–318), `hdl_cholesterol` (range: 20–98), `ldl_cholesterol` (range: 50–263), `triglycerides` (range: 30–344), `glucose_fasting` (range: 60–172), `glucose_postprandial` (range: 70–287), `insulin_level` (range: 2.0–32.22), `hba1c` (range: 4.0–9.8).
   - *Habits/Lifestyle*: `physical_activity_minutes_per_week` (range: 0–833), `diet_score` (range: 0.0–10.0), `sleep_hours_per_day` (range: 3.0–10.0), `screen_time_hours_per_day` (range: 0.5–16.8).
2. **Categorical (10)**:
   - *Demographics (text)*: `gender` (Male, Female, Other), `ethnicity` (Asian, White, Hispanic, Black, Other), `education_level`, `income_level`, `employment_status`, `smoking_status`.
   - *Medical History flags (0/1)*: `family_history_diabetes`, `hypertension_history`, `cardiovascular_history`.
   - *Habits*: `alcohol_consumption_per_week` (range: 0–10).

### Data Quality & EDA Observations
- **Missing Values**: 0 missing values.
- **Skewness & Distributions**: Continuous features like `insulin_level` are moderately right-skewed, while `hba1c` and `glucose_fasting` are relatively symmetric in this population.
- **Outliers**: Outliers are present in continuous medical parameters (e.g., HbA1c, glucose, triglycerides) at low rates (< 1.0%), representing real physiological variations.
- **Correlations**: Several strong linear correlations exist in this clinical dataset:
  - `glucose_postprandial` and `hba1c` ($r = 0.93$)
  - `cholesterol_total` and `ldl_cholesterol` ($r = 0.91$)
  - `bmi` and `waist_to_hip_ratio` ($r = 0.77$)
  - `glucose_fasting` and `hba1c` ($r = 0.70$)
  - `glucose_fasting` and `glucose_postprandial` ($r = 0.59$)
- **Statistical Association**: 22 out of 30 tested features show a statistically significant difference or association with target groups ($p < 0.05$).

---

## 3. Preprocessing and Feature Mapping Decisions

| Decision Area | Dataset 1 (BRFSS 2015) | Dataset 2 (Lifestyle) |
| --- | --- | --- |
| **Duplicates** | Removed 23,899 rows to avoid leakage | None present |
| **Imputation** | Median (numeric) / Mode (categorical) | Median (numeric) / Mode (categorical) |
| **Outliers** | Retained (conceptually valid) | Retained (conceptually valid) |
| **Encoding** | One-Hot Encoding for all 18 categorical columns | One-Hot Encoding for all 10 categorical columns |
| **Scaling** | `StandardScaler` for `BMI`, `MentHlth`, `PhysHlth` | `StandardScaler` for all 18 numerical columns |
| **Split & Stratify** | 80/20 train/test, stratified by `Diabetes_012` | 80/20 train/test, stratified by `diagnosed_diabetes` |
| **Leakage Handling** | N/A | Dropped `diabetes_stage` and `diabetes_risk_score` |

## 4. Key Insights from Graphical Visualizations

Our interactive Plotly charts revealed clear, actionable visual patterns across both datasets:

### Dataset 1: BRFSS 2015
1. **Target Distribution**: The donut chart emphasizes the extreme rarity of prediabetes (Class 1.0, 1.83%) compared to non-diabetic individuals (Class 0.0, 84.24%) and diabetic individuals (Class 2.0, 13.93%).
2. **BMI Distribution and Target Separation**: 
   - The histogram of BMI shows a typical right-skewed distribution, peaking around 26–28.
   - The violin plot of `BMI` vs `Diabetes_012` shows a visible upward shift in the median and interquartile range (IQR) for Class 2.0 (diabetic) respondents (median BMI ~29.0) compared to Class 0.0 (non-diabetic) respondents (median BMI ~27.0). This supports the clinical association between high body mass index and diabetes risk.
3. **Correlation Heatmap**: The heatmap shows very low coefficients across all survey variables. The strongest is `PhysHlth` vs `MentHlth` ($r = 0.35$), suggesting that physical and mental health issues show a mild co-occurrence, but they are not strongly collinear.

### Dataset 2: Lifestyle and Clinical Profile
1. **Target Distribution**: The target chart displays a balanced dataset (60.00% diagnosed vs. 40.00% undiagnosed), which is optimal for binary classification training.
2. **HbA1c and Glucose Group Separation**:
   - The histogram of `hba1c` grouped by `diagnosed_diabetes` reveals a clear biological threshold. Undiagnosed individuals (Class 0) are clustered tightly around the normal range (HbA1c 4.0% to 5.6%). In contrast, diagnosed patients (Class 1) are shifted to the diabetic range (HbA1c 6.5% to 9.8%).
   - The violin plot of `hba1c` vs `diagnosed_diabetes` shows that the median HbA1c for the diagnosed group is ~6.8%, compared to ~5.5% for the undiagnosed group.
3. **Multicollinearity & Clinical Clustering**:
   - The correlation heatmap displays strong collinear blocks: `glucose_postprandial` vs `hba1c` ($r=0.93$), and `cholesterol_total` vs `ldl_cholesterol` ($r=0.91$).
   - The scatter plot of `glucose_postprandial` vs `hba1c` color-coded by target reveals a clear diagonal clustering: diagnosed patients lie exclusively in the high-glucose, high-HbA1c upper-right region, while undiagnosed individuals form a tight group in the bottom-left corner.

## 5. Baseline Model Findings

Six classification baselines were trained independently for each dataset using the preprocessing and stratified 80/20 split described above. Dataset 1's selected baseline is **XGBoost**, with test accuracy 0.8363, macro precision 0.4729, macro recall 0.3914, macro F1 0.4015, ROC-AUC 0.7656, CV mean 0.7609, and CV standard deviation 0.0028. The macro averages reflect the extreme rarity of the prediabetes class and should be interpreted alongside the saved confusion matrices and classification reports.

Dataset 2's selected baseline is **Decision Tree**, with test accuracy 0.9195, precision 0.9971, recall 0.8685, F1 0.9283, ROC-AUC 0.9450, CV mean 0.9434, and CV standard deviation 0.0017. Selection prioritizes ROC-AUC, followed by recall and F1-score rather than accuracy alone. These results describe this dataset and baseline configuration; they do not establish clinical validity.

