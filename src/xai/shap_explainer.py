import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import plotly.express as px
from sklearn.model_selection import train_test_split

from .xai_utils import (
    BASE_DIR,
    display_feature_name,
    load_features,
    load_selected_model,
    representative_indices,
    transformed_data,
)


def _values_for_class(values, class_index):
    if isinstance(values, list):
        return np.asarray(values[class_index])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, class_index]
    return values


def get_shap_global_explanation(dataset_name, sample_size=2000):
    metadata, pipeline = load_selected_model(dataset_name)
    features, target, target_name = load_features(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, stratify=target, random_state=42)
    X_sample = X_train.sample(min(sample_size, len(X_train)), random_state=42)
    transformed, names, classifier = transformed_data(pipeline, X_sample)
    explainer = shap.TreeExplainer(classifier)
    explanation = explainer(transformed)
    values = explanation.values
    if values.ndim == 3:
        values = np.mean(np.abs(values), axis=2)
    importance = pd.DataFrame({"feature": [display_feature_name(name) for name in names], "shap_importance": np.mean(np.abs(values), axis=0)})
    importance = importance.sort_values("shap_importance", ascending=False).reset_index(drop=True)
    output_dir = os.path.join(BASE_DIR, "results", "xai", dataset_name, "shap")
    os.makedirs(os.path.join(output_dir, "local"), exist_ok=True)
    importance.to_csv(os.path.join(output_dir, "shap_feature_importance.csv"), index=False)
    px.bar(importance.head(20).sort_values("shap_importance"), x="shap_importance", y="feature", orientation="h", title=f"SHAP Feature Importance - {dataset_name}").write_html(os.path.join(output_dir, "shap_feature_importance.html"))
    plt.figure(figsize=(10, 8))
    shap.summary_plot(values, transformed, feature_names=importance.feature.tolist() if False else [display_feature_name(name) for name in names], show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    px.bar(importance.head(20).sort_values("shap_importance"), x="shap_importance", y="feature", orientation="h", title=f"SHAP Summary - {dataset_name}").write_html(os.path.join(output_dir, "shap_summary.html"))
    return importance, explanation, (X_test, y_test), metadata


def get_shap_local_explanation(dataset_name, case_name="correct_positive", sample_size=2000):
    importance, _, test_data, metadata = get_shap_global_explanation(dataset_name, sample_size)
    X_test, y_test = test_data
    _, pipeline = load_selected_model(dataset_name)
    predictions = pipeline.predict(X_test)
    positive_label = sorted(y_test.unique())[-1]
    cases = representative_indices(y_test.to_numpy(), predictions, positive_label)
    index = cases.get(case_name, 0)
    row = X_test.iloc[[index]]
    transformed, names, classifier = transformed_data(pipeline, row)
    explanation = shap.TreeExplainer(classifier)(transformed)
    predicted_class = int(predictions[index])
    class_index = list(classifier.classes_).index(predicted_class)
    values = _values_for_class(explanation.values, class_index)[0]
    contributions = pd.DataFrame({"feature": [display_feature_name(name) for name in names], "contribution": values, "value": transformed[0]})
    contributions = contributions.sort_values("contribution", key=lambda series: series.abs(), ascending=False)
    output_dir = os.path.join(BASE_DIR, "results", "xai", dataset_name, "shap", "local")
    os.makedirs(output_dir, exist_ok=True)
    contributions.to_csv(os.path.join(output_dir, f"{case_name}.csv"), index=False)
    px.bar(contributions.head(20).sort_values("contribution"), x="contribution", y="feature", orientation="h", title=f"SHAP Local Contributions - {dataset_name} - {case_name}").write_html(os.path.join(output_dir, f"{case_name}.html"))
    payload = {"case": case_name, "actual_target": int(y_test.iloc[index]), "predicted_target": predicted_class, "model": metadata["best_model"], "contributions": contributions.head(20).to_dict("records")}
    with open(os.path.join(output_dir, f"{case_name}.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return payload


def run_shap(dataset_name):
    _, _, test_data, _ = get_shap_global_explanation(dataset_name)
    X_test, y_test = test_data
    _, pipeline = load_selected_model(dataset_name)
    cases = representative_indices(y_test.to_numpy(), pipeline.predict(X_test), sorted(y_test.unique())[-1])
    return {case: get_shap_local_explanation(dataset_name, case) for case in cases}


if __name__ == "__main__":
    for dataset in ("dataset1", "dataset2"):
        print(dataset, run_shap(dataset))
