import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template, request

from src.config import DATASETS, MONGO_URI
from src.database.mongodb import MongoUnavailable, connect_to_mongodb
from src.routes import create_routes


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False
    app.register_blueprint(create_routes())

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "Resource not found."}), 404
        return render_template("error.html", message="That page does not exist."), 404

    @app.errorhandler(Exception)
    def application_error(error):
        if isinstance(error, FileNotFoundError):
            status_code = 404
            message = "The requested dataset or model is unavailable."
        else:
            status_code = 500
            message = "The request could not be completed."
        if request.path.startswith("/api/"):
            return jsonify({"error": message}), status_code
        return render_template("error.html", message=message), status_code

    return app


app = create_app()

if __name__ == "__main__":
    from src.config import dataset_path

    missing = [name for name in DATASETS if not os.path.exists(dataset_path(name))]
    port = int(os.getenv("PORT", "5000"))
    print(f"DiaRisk Analytics starting on http://127.0.0.1:{port}")
    if MONGO_URI:
        try:
            connect_to_mongodb()
            print("MongoDB Atlas connection: available")
        except MongoUnavailable:
            print("MongoDB Atlas connection: unavailable; prediction history is temporarily unavailable.")
    else:
        print("MongoDB Atlas connection: not configured; prediction history is temporarily unavailable.")
    if missing:
        print(f"Warning: unavailable datasets: {', '.join(missing)}")
    app.run(host="127.0.0.1", port=port, debug=False)
