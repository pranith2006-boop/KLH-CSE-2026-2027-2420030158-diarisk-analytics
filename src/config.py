import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASETS = {
    "dataset1": {
        "label": "BRFSS Health Indicators",
        "filename": "diabetes_012_health_indicators_BRFSS2015.csv",
        "target": "Diabetes_012",
        "drop_columns": [],
    },
    "dataset2": {
        "label": "Lifestyle and Clinical Profiles",
        "filename": "Diabetes_and_LifeStyle_Dataset .csv",
        "target": "diagnosed_diabetes",
        "drop_columns": ["diabetes_stage", "diabetes_risk_score"],
    },
}

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "diarisk_analytics")


def dataset_path(dataset_name):
    return os.path.join(DATA_DIR, DATASETS[dataset_name]["filename"])
