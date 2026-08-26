# DiaRisk Analytics Final Technical Summary

## Project

DiaRisk Analytics — Predict, Analyze, Prevent

## Datasets

- `diabetes_012_health_indicators_BRFSS2015.csv`
- `Diabetes_and_LifeStyle_Dataset .csv`

Original CSV files were not modified or uploaded to MongoDB.

## Models

Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, XGBoost, and CatBoost.

- Dataset 1 selected model: XGBoost, ROC-AUC 0.7656
- Dataset 2 selected model: Decision Tree, ROC-AUC 0.9450

## Integrated Components

- SHAP: implemented and served from generated artifacts.
- LIME: implemented and served from generated artifacts.
- Flask: application and JSON APIs integrated.
- MongoDB Atlas: connected using private environment configuration.
- Prediction history: persisted in the `diarisk_analytics.predictions` collection with pagination, filters, lookup, and deletion.

## Testing Summary

- Nine application pages: PASS
- Dataset 1 and Dataset 2 prediction flows: PASS
- SHAP/LIME artifact integration: PASS
- MongoDB connection, insert, retrieve, filter, and delete: PASS
- Empty history and invalid input handling: PASS
- Credential hygiene and `.env` ignore rules: PASS
- Dataset redistribution compliance: NOT VERIFIED; both CSV files are currently tracked in Git and require licensing/permission review before submission.
- Automated responsive browser test: NOT RUN because the local Playwright browser executable is unavailable.
- GitHub permissions, team contributions, supervisor access, and tags: NOT VERIFIED locally.

## Readiness

The local application integration is ready for team review. Remaining submission tasks are external GitHub governance checks and optional browser automation setup.
