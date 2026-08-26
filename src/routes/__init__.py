import os

from flask import Blueprint, jsonify, render_template, request, send_from_directory

from src.config import DATASETS, RESULTS_DIR
from src.routes.services import chart_json, dataset_features, dataset_summary, evaluation, predict, xai_artifacts
from src.database.mongodb import MongoUnavailable, delete_prediction, get_prediction_by_id, get_prediction_history


def create_routes():
    blueprint = Blueprint("web", __name__)

    @blueprint.get("/")
    def home():
        return render_template("index.html")

    @blueprint.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @blueprint.get("/analytics")
    def analytics():
        return render_template("analytics.html")

    @blueprint.get("/visualizations")
    def visualizations():
        return render_template("visualization.html")

    @blueprint.get("/prediction")
    def prediction():
        return render_template("prediction.html")

    @blueprint.get("/evaluation")
    def evaluation_page():
        return render_template("evaluation.html")

    @blueprint.get("/explainability")
    def explainability():
        return render_template("explainability.html")

    @blueprint.get("/history")
    def history():
        return render_template("history.html")

    @blueprint.get("/about")
    def about():
        return render_template("about.html")

    @blueprint.get("/api/datasets")
    def datasets():
        return jsonify({name: config["label"] for name, config in DATASETS.items()})

    @blueprint.get("/api/dataset/<name>/summary")
    def summary(name):
        return jsonify(dataset_summary(name))

    @blueprint.get("/api/dataset/<name>/features")
    def features(name):
        values, target = dataset_features(name)
        return jsonify({"features": values, "target": target})

    @blueprint.get("/api/dataset/<name>/charts")
    def charts_api(name):
        return jsonify({"figure": chart_json(name, request.args.get("type", "target"), request.args.get("x"), request.args.get("y"))})

    @blueprint.get("/api/models/<name>")
    def models(name):
        rows, best = evaluation(name)
        return jsonify({"models": rows, "best": best})

    @blueprint.get("/api/evaluation/<name>")
    def evaluation_api(name):
        rows, best = evaluation(name)
        return jsonify({"models": rows, "best": best})

    @blueprint.get("/api/xai/<name>")
    def xai_api(name):
        return jsonify(xai_artifacts(name))

    @blueprint.get("/api/explain/<name>/<case_name>")
    def explain_api(name, case_name):
        path = os.path.join(RESULTS_DIR, "xai", name, "shap", "local", f"{case_name}.json")
        if not os.path.exists(path):
            return jsonify({"error": "Explanation is unavailable for this case."}), 404
        with open(path, encoding="utf-8") as file:
            return jsonify(__import__("json").load(file))

    @blueprint.get("/artifacts/xai/<dataset>/<path:filename>")
    def xai_artifact(dataset, filename):
        directory = os.path.join(RESULTS_DIR, "xai", dataset)
        return send_from_directory(directory, filename)

    @blueprint.get("/artifacts/model-metrics/<dataset>/<path:filename>")
    def model_artifact(dataset, filename):
        directory = os.path.join(RESULTS_DIR, "model_metrics", dataset)
        return send_from_directory(directory, filename)

    @blueprint.post("/api/predict")
    def predict_api():
        payload = request.get_json(silent=True) or {}
        name = payload.get("dataset")
        values = payload.get("values", {})
        if not name or not values:
            return jsonify({"error": "Dataset and feature values are required."}), 400
        try:
            return jsonify(predict(name, values, payload.get("model")))
        except (KeyError, ValueError, TypeError) as error:
            return jsonify({"error": str(error)}), 400

    @blueprint.get("/api/history")
    def history_api():
        filters = {key: request.args.get(key) for key in ("dataset", "model", "prediction")}
        try:
            return jsonify(get_prediction_history(filters, request.args.get("page", 1), request.args.get("per_page", 10)))
        except MongoUnavailable:
            return jsonify({"available": False, "error": "Prediction history is temporarily unavailable."}), 503

    @blueprint.get("/api/history/<prediction_id>")
    def history_record_api(prediction_id):
        try:
            record = get_prediction_by_id(prediction_id)
        except MongoUnavailable:
            return jsonify({"error": "Prediction history is temporarily unavailable."}), 503
        if record is None:
            return jsonify({"error": "Prediction record not found."}), 404
        return jsonify(record)

    @blueprint.delete("/api/history/<prediction_id>")
    def delete_history_api(prediction_id):
        try:
            deleted = delete_prediction(prediction_id)
        except MongoUnavailable:
            return jsonify({"error": "Prediction history is temporarily unavailable."}), 503
        if not deleted:
            return jsonify({"error": "Prediction record not found."}), 404
        return jsonify({"success": True})

    return blueprint
