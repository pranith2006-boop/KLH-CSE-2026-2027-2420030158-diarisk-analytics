# DiaRisk Analytics Architecture

## Request Flow

```text
Browser
  -> HTML / CSS / JavaScript
  -> Flask routes and JSON APIs
  -> Shared services
  -> Existing analytics, visualization, ML, and XAI modules
  -> MongoDB service -> MongoDB Atlas (Phase 9)
```

Prediction inputs and model outputs are persisted only in the `predictions` collection. Original CSV datasets are never uploaded. If Atlas is unavailable, analytics and predictions continue and the UI reports that prediction history is temporarily unavailable.

## Application Structure

- `src/app.py`: Flask application factory, blueprint registration, startup message, and user-safe error handlers.
- `src/config.py`: dataset, model, result, and environment configuration.
- `src/routes/services.py`: shared dataset loading, feature metadata, Plotly chart calls, evaluation artifact loading, and prediction orchestration.
- `src/routes/__init__.py`: page routes and JSON API routes.
- `src/ml/`: saved-pipeline prediction and model artifacts from Phases 5-6.
- `src/xai/`: SHAP/LIME explanations and generated artifacts from Phase 7.
- `src/visualization/charts.py`: reusable Plotly chart builders from Phase 4.
- `src/templates/`: responsive Bootstrap pages.
- `src/static/`: shared JavaScript and CSS.

## Routes

Pages: `/`, `/dashboard`, `/analytics`, `/visualizations`, `/prediction`, `/evaluation`, `/explainability`, `/history`, `/about`.

APIs: `/api/datasets`, `/api/dataset/<name>/summary`, `/api/dataset/<name>/features`, `/api/dataset/<name>/charts`, `/api/models/<name>`, `/api/evaluation/<name>`, `/api/predict`, and `/api/explain/<name>/<case_name>`.

Prediction forms are generated from actual dataset columns and invoke the existing saved pipeline through the shared service layer. Successful predictions are handed to the MongoDB service for storage. Flask does not duplicate preprocessing, model selection, or XAI calculations.
