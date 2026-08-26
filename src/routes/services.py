import json
import os
from datetime import datetime, timezone

import pandas as pd

from src.config import DATASETS, MODEL_DIR, RESULTS_DIR, dataset_path
from src.visualization import charts
from src.ml.predict import get_model_pipeline
from src.database.mongodb import MongoUnavailable, get_prediction_history, save_prediction


def valid_dataset(name):
    return name in DATASETS and os.path.exists(dataset_path(name))


def load_dataset(name):
    if not valid_dataset(name):
        raise FileNotFoundError("Dataset is unavailable")
    frame = pd.read_csv(dataset_path(name)).drop_duplicates().reset_index(drop=True)
    config = DATASETS[name]
    return frame.drop(columns=config["drop_columns"], errors="ignore"), config["target"]


def dataset_features(name):
    frame, target = load_dataset(name)
    fields = []
    for column in frame.columns:
        if column == target:
            continue
        series = frame[column]
        field = {"name": column, "type": "number" if pd.api.types.is_numeric_dtype(series) else "select"}
        if field["type"] == "number":
            field.update({"min": float(series.min()), "max": float(series.max()), "step": "any"})
        else:
            field["choices"] = [str(value) for value in series.dropna().unique()]
        fields.append(field)
    return fields, target


def dataset_summary(name):
    frame, target = load_dataset(name)
    counts = frame[target].value_counts().to_dict()
    summary = {
        "dataset": name, "label": DATASETS[name]["label"], "target": target,
        "total_records": len(frame), "total_features": len(frame.columns) - 1,
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_records_removed": int(pd.read_csv(dataset_path(name)).duplicated().sum()),
    }
    for column in frame.columns:
        if column.lower() in {"age", "bmi", "glucose_fasting"}:
            summary[f"average_{column}"] = round(float(frame[column].mean()), 2)
    return summary


def chart_json(name, chart_type="target", x=None, y=None):
    frame, target = load_dataset(name)
    numeric = frame.select_dtypes(include="number").columns.tolist()
    x = x if x in frame.columns else (numeric[0] if numeric else target)
    y = y if y in frame.columns else (numeric[1] if len(numeric) > 1 else target)
    builders = {
        "target": lambda: charts.create_target_distribution(frame, target),
        "histogram": lambda: charts.create_histogram(frame, x, target),
        "boxplot": lambda: charts.create_boxplot(frame, x, target),
        "scatter": lambda: charts.create_scatter_plot(frame, x, y, target),
        "bar": lambda: charts.create_categorical_chart(frame, x, target),
        "correlation": lambda: charts.create_correlation_heatmap(frame, numeric),
    }
    figure = builders.get(chart_type, builders["target"])()
    return figure.to_json() if figure is not None else "{}"


def evaluation(name):
    path = os.path.join(RESULTS_DIR, "model_metrics", f"{name}_model_comparison.csv")
    frame = pd.read_csv(path)
    best_path = os.path.join(RESULTS_DIR, "model_metrics", f"{name}_best_model.json")
    with open(best_path, encoding="utf-8") as file:
        best = json.load(file)
    ranking_path = os.path.join(RESULTS_DIR, "model_metrics", f"{name}_model_ranking.csv")
    ranking = pd.read_csv(ranking_path).to_dict("records") if os.path.exists(ranking_path) else []
    roc_path = f"/artifacts/model-metrics/{name}/roc_curves/model_roc_comparison.html"
    return frame.to_dict("records"), {**best, "ranking": ranking, "roc_chart": roc_path}


def xai_artifacts(name):
    base = os.path.join(RESULTS_DIR, "xai", name)
    return {
        "shap_global": f"/artifacts/xai/{name}/shap/shap_feature_importance.html",
        "shap_summary": f"/artifacts/xai/{name}/shap/shap_summary.html",
        "lime_example": f"/artifacts/xai/{name}/lime/correct_positive.html",
        "available": os.path.exists(os.path.join(base, "shap", "shap_feature_importance.html")),
    }


def predict(name, values, model_name=None):
    _, target = load_dataset(name)
    if model_name is None:
        with open(os.path.join(RESULTS_DIR, "model_metrics", f"{name}_best_model.json"), encoding="utf-8") as file:
            model_name = json.load(file)["best_model"]
    pipeline = get_model_pipeline(name, model_name)
    result = pipeline.predict(pd.DataFrame([values]))[0]
    probabilities = pipeline.predict_proba(pd.DataFrame([values]))[0]
    probabilities = {str(label): round(float(prob), 6) for label, prob in zip(pipeline.classes_, probabilities)}
    xai_available = os.path.exists(os.path.join(RESULTS_DIR, "xai", name, "shap", "local", "correct_positive.json"))
    document = {
        "dataset": name, "model": model_name, "prediction": str(result),
        "probability": max(probabilities.values()), "risk_level": str(result),
        "features": values, "timestamp": datetime.now(timezone.utc),
        "xai_available": xai_available,
    }
    history_id = None
    history_status = "saved"
    try:
        history_id = save_prediction(document)
    except MongoUnavailable:
        history_status = "unavailable"
    except Exception:
        history_status = "unavailable"
    return {"target": target, "model": model_name, "predicted_class": str(result), "probabilities": probabilities, "timestamp": document["timestamp"].isoformat(), "history_id": history_id, "history_status": history_status, "xai_available": xai_available}
