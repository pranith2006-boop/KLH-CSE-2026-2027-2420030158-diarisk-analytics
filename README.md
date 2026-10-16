# Diabetes Risk Prediction and Model Evaluation

### Data Analytics and Visualization — 24DSB3101

**Academic Year:** 2026–2027

---

## 📌 Project Overview

**Diabetes Risk Prediction and Model Evaluation** is a machine learning and data analytics-based healthcare project designed to support the early identification of diabetes risk through clinical and demographic data.

The project integrates **Exploratory Data Analysis (EDA), data preprocessing, predictive modelling, Explainable Artificial Intelligence (XAI), model evaluation, and interactive visualization** into a unified analytics framework.

The system compares multiple machine learning algorithms and uses **SHAP (Shapley Additive Explanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** to improve model transparency and explain the factors contributing to individual predictions.

---

## 👥 Team Members

| S. No. | University ID | Name                  |
| ------ | ------------- | --------------------- |
| 1      | 2420030158    | MODALLA PRANITH REDDY |
| 2      | 2420030277    | P.M. PRANITHA         |
| 3      | 2420030580    | I. HEMA SRI           |

### Guide

**Garimidi Siva Sree**

---

## 📝 Abstract

Diabetes is one of the most common chronic diseases worldwide, and its early detection is essential for preventing severe health complications. Machine learning techniques have shown significant potential in predicting diabetes risk using clinical and demographic data. However, many predictive models operate as black-box systems, limiting their interpretability and reducing the confidence of healthcare professionals in adopting AI-based decision support systems. To address this challenge, this project proposes a machine learning-based framework for Diabetes Risk Prediction and Model Evaluation by integrating Exploratory Data Analysis (EDA), predictive modelling, and Explainable Artificial Intelligence (XAI).

The proposed framework utilises diabetes-related datasets for predicting the likelihood of diabetes. Initially, the datasets undergo preprocessing, including missing value handling, outlier detection, feature scaling, and normalisation to improve data quality. A comprehensive EDA is then performed using descriptive statistics, correlation analysis, distribution analysis, and visualisation techniques to identify important patterns and relationships among clinical features.

Multiple machine learning algorithms, including Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), XGBoost, and CatBoost, are trained and evaluated. Their performance is assessed using standard evaluation metrics such as Accuracy, Precision, Recall, F1-Score, ROC-AUC Score, and Confusion Matrix. To improve model transparency, SHAP and LIME are incorporated to provide both global and local explanations of model predictions, highlighting the contribution of important clinical features.

Finally, the results are presented through interactive visualisations, including heatmaps, boxplots, feature importance graphs, ROC curves, and dashboards. A comparative analysis is performed to identify the most accurate and interpretable model for diabetes risk prediction. The proposed framework aims to achieve high predictive performance while providing transparent and explainable insights that support early diagnosis, informed clinical decision-making, and improved healthcare outcomes.

---

## 🎯 Problem Statement

Diabetes requires early detection to reduce the risk of severe health complications. However, analysing large healthcare datasets manually can be time-consuming and may make it difficult to identify important patterns associated with diabetes risk. Furthermore, black-box machine learning models can make predictions without clearly explaining the factors influencing those predictions.

This project addresses these challenges by developing a data-driven framework that combines data preprocessing, exploratory data analysis, machine learning, interactive visualization, and Explainable AI to predict diabetes risk and provide transparent insights into model decisions.

---

## 🎯 Objectives

* Perform data preprocessing and cleaning.
* Handle missing values and detect outliers.
* Apply feature scaling and normalization where required.
* Perform Exploratory Data Analysis (EDA).
* Identify important relationships and patterns among clinical features.
* Develop multiple machine learning classification models.
* Compare machine learning models using standard evaluation metrics.
* Identify the best-performing predictive model.
* Implement SHAP for global and local model explanations.
* Implement LIME for local prediction explanations.
* Create interactive healthcare visualizations and dashboards.
* Provide transparent and interpretable diabetes risk predictions.

---

## 📊 Data Analytics

The project performs the following analytical operations:

* Descriptive statistics
* Missing-value analysis
* Duplicate analysis
* Outlier detection
* Distribution analysis
* Correlation analysis
* Feature analysis
* Feature importance analysis
* Comparative dataset analysis

---

## 🤖 Machine Learning Models

The following algorithms are considered for model training and comparison:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. XGBoost
6. CatBoost

Each model is evaluated using consistent preprocessing and evaluation procedures.

---

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

The final comparison considers both **predictive performance and interpretability**.

---

## 🎓 Review 3: Machine Learning Presentation & Technical Evaluation

For **Review 3**, the project defense covers the six machine learning lifecycle phases implemented in this repository:

### 1. Data Understanding & Feature Semantics
- **Dataset 1 (CDC BRFSS 2015 Health Indicators)**: 229,781 clean records, 21 predictors; 3 continuous numerical (`BMI`, `MentHlth`, `PhysHlth`), 14 binary categorical flags (`HighBP`, `HighChol`, `Smoker`, etc.), and 4 ordinal categorical scales (`GenHlth`, `Age`, `Education`, `Income`). Multiclass target `Diabetes_012` exhibits a severe **46.15:1** imbalance ratio (Class 0: 82.71%, Class 1: 2.01%, Class 2: 15.27%).
- **Dataset 2 (Clinical & Lifestyle Profile)**: 97,297 records, 28 predictors; 18 continuous lab/biometric measurements (`hba1c`, `glucose_fasting`, `glucose_postprandial`, `insulin_level`, `systolic_bp`, etc.) and 10 categorical variables. Binary target `diagnosed_diabetes` has a balanced **1.50:1** ratio (Class 0: 40.0%, Class 1: 60.0%).

### 2. Data Cleaning & Imputation
- Verified **0 raw missing values** across both datasets with physiological integrity checks.
- Embedded production-grade imputers inside pipelines: `SimpleImputer(strategy='median')` for numerical features and `SimpleImputer(strategy='most_frequent')` for categorical features.
- Detected and removed **23,899 duplicate records (9.42%)** in Dataset 1 to prevent data contamination.
- Dropped target proxies (`diabetes_stage`, `diabetes_risk_score`) in Dataset 2 to prevent circular target leakage.

### 3. Statistical Checks & Outlier Handling
- **Normality Testing**: Continuous features severely violate Gaussian distribution assumptions (e.g., `BMI` skewness = +1.21; `MentHlth` skewness = +2.23; $p < 0.001$).
- **Non-Parametric Hypothesis Tests**: Kruskal-Wallis H-test for Dataset 1 ($p < 10^{-100}$) and Mann-Whitney U test for Dataset 2 ($p < 10^{-190}$ for `hba1c` and fasting glucose) established decisive group separation; categorical features confirmed via Chi-Square ($\chi^2$) tests of independence.
- **Outlier Handling**: Detected via the IQR rule ($[Q1 - 1.5\text{IQR}, Q3 + 1.5\text{IQR}]$) and **retained completely**. In clinical healthcare, extreme readings (e.g., HbA1c = 9.8%, fasting glucose = 172 mg/dL) represent genuine biological disease pathology rather than measurement noise. Deleting them would eliminate critical diagnostic signals.

### 4. Data Preprocessing & Scaling Justification
- Stratified 80/20 train/test split preserving class distribution.
- **Scaling Choice**: Selected `StandardScaler` (Z-score) over Min-Max normalisation because Min-Max squashes 99% of normal patient values into a tiny band in the presence of clinical outliers, whereas StandardScaler preserves variance, centers features at $\mu=0, \sigma=1$, and provides spherical loss geometry for gradient/margin optimization.
- **Categorical Encoding**: Applied `OneHotEncoder(handle_unknown='ignore')` to eliminate false linear distance assumptions on ordinal survey codes, expanding feature spaces to 69 (Dataset 1) and 65 (Dataset 2) columns.
- Preprocessing enclosed in Scikit-Learn `ColumnTransformer` fitted strictly on `X_train` to eliminate data leakage.

### 5. Model Building & Imbalance Strategy
- Trained and serialized **6 algorithms** per dataset: **Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), XGBoost, and CatBoost**.
- Evaluated with `class_weight='balanced'` on Dataset 1 to handle the 46.15:1 class imbalance.

### 6. Model Analysis, Metrics & Clinical Findings
- Evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, 5-Fold Stratified Cross-Validation, and Confusion Matrices.
- **Dataset 1 Winner**: **XGBoost** (ROC-AUC: **0.7656**, 5-Fold CV: 0.7609 ± 0.0028, Accuracy: 83.63%).
- **Dataset 2 Winner**: **Decision Tree** (ROC-AUC: **0.9450**, Precision: **99.71%**, Recall: 86.85%, F1: 0.9283, 5-Fold CV: 0.9434 ± 0.0017).
- **Clinical Validation**: Feature importance analysis confirms that **`hba1c` (95.01%)** and **`glucose_fasting` (3.92%)** drive the Decision Tree's decisions, aligning directly with American Diabetes Association (ADA) diagnostic guidelines ($\text{HbA1c} \ge 6.5\%$).

*Full presentation guide & slides script:* [reports/review3_ml_presentation.md](reports/review3_ml_presentation.md)

---

## 🔍 Explainable Artificial Intelligence (XAI)

A major component of the project is Explainable AI.

### SHAP

**SHAP (Shapley Additive Explanations)** is used to understand the contribution of features to model predictions.

It can provide:

* Global feature importance
* Individual prediction explanations
* Positive and negative feature contributions

### LIME

**LIME (Local Interpretable Model-agnostic Explanations)** is used to explain individual predictions by showing which features most influenced a particular prediction.

The XAI component helps reduce the black-box nature of machine learning models and provides more transparent insights.

---

## 📊 Visualizations

The system provides interactive and analytical visualizations such as:

* Correlation Heatmaps
* Boxplots
* Histograms
* Distribution Charts
* Feature Importance Graphs
* ROC Curves
* Confusion Matrices
* Interactive Dashboard Charts
* SHAP Visualizations
* LIME Explanations

---

## 🌐 Web Application

The project is implemented as a web-based analytics platform.

### Main Modules

* **Home**
* **Dashboard**
* **Dataset Analytics**
* **Data Visualization**
* **Diabetes Risk Prediction**
* **Model Evaluation**
* **Explainable AI**
* **Prediction History**
* **About**

---

## 🛠️ Technology Stack

| Category          | Technologies                         |
| ----------------- | ------------------------------------ |
| Programming       | Python                               |
| Data Analytics    | Pandas, NumPy                        |
| Machine Learning  | Scikit-learn, XGBoost, CatBoost      |
| Explainable AI    | SHAP, LIME                           |
| Visualization     | Plotly, Matplotlib, Seaborn          |
| Backend           | Flask                                |
| Frontend          | HTML5, CSS3, JavaScript, Bootstrap 5 |
| Database          | MongoDB Atlas, PyMongo               |
| Model Persistence | Joblib                               |
| Configuration     | Python-dotenv                        |
| Development       | VS Code                              |
| Version Control   | Git, GitHub                          |

---

## 📁 Repository Structure

The repository follows the mandatory project submission structure:

```text
/
├── README.md
│
├── src/
│   ├── app.py
│   ├── config.py
│   ├── routes/
│   ├── preprocessing/
│   ├── analytics/
│   ├── visualization/
│   ├── ml/
│   ├── xai/
│   ├── database/
│   ├── templates/
│   └── static/
│
├── docs/
│   ├── problem_statement.md
│   ├── methodology.md
│   ├── architecture.md
│   └── dataset_description.md
│
├── data/
│   └── README.md
│
├── results/
│   ├── visualizations/
│   ├── model_metrics/
│   └── xai/
│
└── reports/
    ├── review3_ml_presentation.md
    ├── progress/
    └── final/
```

---

## 📂 Dataset

The project uses diabetes-related datasets for data analysis, visualization, and predictive modelling.

The datasets are **not committed to the repository when redistribution is restricted**.

Dataset files should be placed locally in the appropriate `data/` directory before running the project.

The `data/README.md` file contains the dataset source information and instructions.

> **Note:** The submitted project abstract specifically identifies the Pima Indians Diabetes Dataset as the dataset used for prediction. Any change to the approved dataset configuration should be documented and approved according to the project's academic requirements.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Configuration

Create a `.env` file locally.

```env
MONGO_URI=your_mongodb_atlas_connection_string
DATABASE_NAME=diarisk_analytics
```

**Never commit ****`.env`**** or database credentials to GitHub.**

The `.env` file should be included in `.gitignore`.

---

## ▶️ Running the Application

After completing the setup:

```bash
python src/app.py
```

Open the local Flask application in your browser using the URL displayed by Flask.

---

## 🗄️ MongoDB

MongoDB Atlas is used to store prediction history and related application data.

The application uses **PyMongo** to communicate with MongoDB Atlas.

Example collection:

```text
predictions
```

Stored information may include:

* Dataset
* Input features
* Prediction
* Prediction probability
* Risk category
* Model used
* Timestamp

---

## 🔄 Project Workflow

```text
Data Collection
      ↓
Data Preprocessing
      ↓
Missing Value Handling
      ↓
Outlier Detection
      ↓
Feature Scaling & Normalization
      ↓
Exploratory Data Analysis
      ↓
Feature Analysis
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
SHAP & LIME Explainability
      ↓
Interactive Visualization
      ↓
Diabetes Risk Prediction
      ↓
MongoDB Prediction History
```

---

## 📌 Current Phase Status

**Current Phase:** Phase 10 — Final Integration and Testing

**Status:** Completed for local application integration; GitHub governance checks remain team-owned

### Completed

* [x] Project topic finalized
* [x] Project abstract submitted
* [x] Technology stack identified
* [x] GitHub repository structure planned
* [x] Dataset sources identified
* [x] Dataset preprocessing
* [x] Exploratory Data Analysis
* [x] Visualization development
* [x] Machine learning implementation
* [x] Model evaluation
* [x] Model comparison and best-model selection
* [x] Explainable AI implementation
* [x] Flask application integration
* [x] MongoDB Atlas integration and prediction history

### In Progress

* [ ] GitHub governance and submission review

### Planned

* [ ] Final project deployment

### Final Verification Notes

* [x] Dashboard, analytics, visualizations, prediction, evaluation, explainability, and history pages integrated.
* [x] Both datasets tested independently through the Flask prediction workflow.
* [x] SHAP and LIME artifacts are available for the selected models.
* [x] MongoDB Atlas connection, prediction storage, filtering, retrieval, and deletion verified.
* [x] Invalid input, missing dataset, empty history, and unavailable-service handling verified.
* [x] Final testing report and project checklist created under `reports/`.
* [ ] Automated responsive browser testing requires a locally installed Playwright browser.
* [ ] Dataset redistribution permissions must be confirmed before submission; local CSV files are currently tracked in Git.
* [ ] Team contributions, weekly commits, repository permissions, supervisor access, course coordinator access, and milestone tags require GitHub verification.

## Final Model Results

Model selection prioritizes ROC-AUC, followed by Recall, F1-Score, and Accuracy. Values below are from the held-out test evaluation and are not medical performance claims.

| Dataset | Selected Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dataset 1 — BRFSS Health Indicators | XGBoost | 0.8363 | 0.4729 | 0.3914 | 0.4015 | 0.7656 |
| Dataset 2 — Lifestyle and Clinical Profiles | Decision Tree | 0.9195 | 0.9971 | 0.8685 | 0.9283 | 0.9450 |

The complete six-model comparisons, cross-validation results, rankings, confusion matrices, ROC curves, and feature-importance outputs are stored in `results/model_metrics/`.

## MongoDB Configuration

Prediction history uses the `predictions` collection in the `diarisk_analytics` MongoDB database. Configure Atlas locally in `.env`:

```env
MONGO_URI=your_mongodb_atlas_connection_string
DATABASE_NAME=diarisk_analytics
```

The application never stores the original CSV datasets. It stores only the dataset identifier, model, predicted class, probability, submitted feature values, timestamp, and XAI availability status. `.env` is ignored by Git; use `.env.example` as the blank template and never place credentials in README or source files.

## Final Reports

* [Review 3 ML Presentation & Technical Report](reports/review3_ml_presentation.md)
* [Final testing report](reports/progress/final_testing_report.md)
* [Final project checklist](reports/final/final_project_checklist.md)
* [Final technical summary](reports/final/final_technical_summary.md)

---

## 🔒 Security and Repository Guidelines

The repository must not contain:

* MongoDB credentials
* API keys
* Passwords
* `.env` files
* Confidential institutional data
* Restricted or licensed datasets

Use environment variables for sensitive configuration.

---

## 👨‍💻 Development Guidelines

The project follows a modular development approach.

Each major feature should be developed and committed progressively.

Example commit messages:

```text
feat: add dataset preprocessing
feat: implement EDA module
feat: add interactive visualizations
feat: implement random forest model
feat: add SHAP explanations
feat: add LIME explanations
feat: integrate MongoDB
feat: implement prediction dashboard
docs: update project documentation
```

---

## 🏷️ Project Milestones

Major project phases will be tagged using Git tags:

```text
review-1
review-2
review-3
final
```

---

## ⚠️ Disclaimer

This project is developed for **academic and educational purposes**. Diabetes risk predictions generated by the system should not be considered a medical diagnosis or a substitute for professional medical advice.

---

## 🚀 Future Scope

Potential future improvements include:

* Integration with real-time healthcare data
* Deployment on cloud infrastructure
* Integration with electronic health records
* Advanced deep learning models
* Additional explainable AI techniques
* Mobile application support
* Real-time patient monitoring
* Integration with wearable health devices

---

## 📜 License

This project is developed as an academic project for the **Data Analytics and Visualization (24DSB3101)** course.

Dataset licensing and redistribution conditions must be respected separately from the project source code.
