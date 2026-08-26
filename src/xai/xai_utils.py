import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_selected_model(dataset_name: str):
    metadata_path = os.path.join(BASE_DIR, "results", "model_metrics", f"{dataset_name}_best_model.json")
    with open(metadata_path, encoding="utf-8") as file:
        metadata = json.load(file)
    model_path = os.path.join(BASE_DIR, metadata["model_path"])
    return metadata, joblib.load(model_path)


def load_features(dataset_name: str):
    if dataset_name == "dataset1":
        path = os.path.join(BASE_DIR, "data", "diabetes_012_health_indicators_BRFSS2015.csv")
        target = "Diabetes_012"
        frame = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
        return frame.drop(columns=[target]), frame[target].astype(int), target
    path = os.path.join(BASE_DIR, "data", "Diabetes_and_LifeStyle_Dataset .csv")
    target = "diagnosed_diabetes"
    frame = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    return frame.drop(columns=[target, "diabetes_stage", "diabetes_risk_score"]), frame[target].astype(int), target


def transformed_data(pipeline, features: pd.DataFrame):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(features)
    names = list(preprocessor.get_feature_names_out())
    return np.asarray(transformed), names, classifier


def representative_indices(y_true, y_pred, positive_label):
    cases = {
        "correct_positive": np.flatnonzero((y_true == positive_label) & (y_pred == positive_label)),
        "correct_negative": np.flatnonzero((y_true != positive_label) & (y_pred != positive_label)),
        "false_positive": np.flatnonzero((y_true != positive_label) & (y_pred == positive_label)),
        "false_negative": np.flatnonzero((y_true == positive_label) & (y_pred != positive_label)),
    }
    return {name: int(indices[0]) for name, indices in cases.items() if len(indices)}


def json_safe(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def display_feature_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "")
