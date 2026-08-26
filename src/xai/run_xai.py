import os
import pandas as pd

from .xai_utils import BASE_DIR, load_selected_model
from .shap_explainer import run_shap
from .lime_explainer import run_lime


def build_reports():
    root = os.path.join(BASE_DIR, "results", "xai")
    os.makedirs(root, exist_ok=True)
    summaries = []
    for dataset_name in ("dataset1", "dataset2"):
        metadata, _ = load_selected_model(dataset_name)
        shap_cases = run_shap(dataset_name)
        lime_cases = run_lime(dataset_name)
        shap_global = pd.read_csv(os.path.join(root, dataset_name, "shap", "shap_feature_importance.csv"))
        phase6_path = os.path.join(BASE_DIR, "results", "model_metrics", dataset_name, "feature_importance", f"{metadata['best_model']}_feature_importance.csv")
        if os.path.exists(phase6_path):
            phase6 = pd.read_csv(phase6_path).rename(columns={"Feature": "feature", "Importance": "model_importance"})
            comparison = shap_global.merge(phase6, on="feature", how="outer")
            comparison.to_csv(os.path.join(root, dataset_name, "feature_importance_comparison.csv"), index=False)
        top_features = ", ".join(shap_global.head(10)["feature"].tolist())
        summaries.append(f"{dataset_name.upper()}\nBest Model: {metadata['best_model']}\nTop SHAP features: {top_features}\nRepresentative cases: {', '.join(shap_cases)}\nLIME cases: {', '.join(lime_cases)}\nFalse-negative explanation: {'available' if 'false_negative' in shap_cases else 'not available'}\n")

    with open(os.path.join(root, "xai_method_comparison.md"), "w", encoding="utf-8") as file:
        file.write("# SHAP and LIME Method Comparison\n\n")
        file.write("SHAP provides model-aware global importance and additive local feature contributions. TreeExplainer is used here because both Phase 6 selections are tree models. LIME is model-agnostic and fits a local surrogate around one transformed record. Their results can differ because they use different approximation and weighting methodologies; neither is universally superior. Encoded categorical features remain represented as transformed source-category names, which is a limitation when aggregating back to original columns.\n")

    with open(os.path.join(root, "xai_summary.md"), "w", encoding="utf-8") as file:
        file.write("# Explainable AI Summary\n\n" + "\n".join(summaries))


if __name__ == "__main__":
    build_reports()
