import os

import json
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split

from .xai_utils import BASE_DIR, display_feature_name, load_features, load_selected_model, representative_indices, transformed_data


def get_lime_explanation(dataset_name, case_name="correct_positive", sample_size=2000, num_features=20):
    metadata, pipeline = load_selected_model(dataset_name)
    features, target, target_name = load_features(dataset_name)
    _, X_test, _, y_test = train_test_split(features, target, test_size=0.20, stratify=target, random_state=42)
    predictions = pipeline.predict(X_test)
    positive_label = sorted(y_test.unique())[-1]
    cases = representative_indices(y_test.to_numpy(), predictions, positive_label)
    index = cases.get(case_name, 0)
    train_features, _, _, _ = train_test_split(features, target, test_size=0.20, stratify=target, random_state=42)
    train_transformed, names, _ = transformed_data(pipeline, train_features.sample(min(sample_size, len(train_features)), random_state=42))
    test_transformed, _, _ = transformed_data(pipeline, X_test.iloc[[index]])

    explainer = LimeTabularExplainer(
        train_transformed,
        feature_names=[display_feature_name(name) for name in names],
        class_names=[str(label) for label in sorted(target.unique())],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    classifier = lambda rows: pipeline.named_steps["classifier"].predict_proba(rows)
    explanation = explainer.explain_instance(test_transformed[0], classifier, num_features=num_features, top_labels=1)
    label = int(predictions[index])
    contributions = [{"feature": feature, "contribution": float(weight)} for feature, weight in explanation.as_list(label=label)]
    output_dir = os.path.join(BASE_DIR, "results", "xai", dataset_name, "lime")
    os.makedirs(output_dir, exist_ok=True)
    explanation.save_to_file(os.path.join(output_dir, f"{case_name}.html"))
    payload = {"case": case_name, "actual_target": int(y_test.iloc[index]), "predicted_target": label, "model": metadata["best_model"], "contributions": contributions}
    with open(os.path.join(output_dir, f"{case_name}.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return payload


def run_lime(dataset_name):
    features, target, _ = load_features(dataset_name)
    _, X_test, _, y_test = train_test_split(features, target, test_size=0.20, stratify=target, random_state=42)
    _, pipeline = load_selected_model(dataset_name)
    cases = representative_indices(y_test.to_numpy(), pipeline.predict(X_test), sorted(y_test.unique())[-1])
    return {case: get_lime_explanation(dataset_name, case) for case in cases}


if __name__ == "__main__":
    for dataset in ("dataset1", "dataset2"):
        print(dataset, run_lime(dataset))
