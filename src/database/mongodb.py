import logging
from datetime import datetime, timezone

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient

from src.config import DATABASE_NAME, MONGO_URI

logger = logging.getLogger(__name__)
_client = None


class MongoUnavailable(RuntimeError):
    """Raised when Atlas is not configured or cannot be reached."""


def connect_to_mongodb():
    global _client
    if not MONGO_URI:
        raise MongoUnavailable("MongoDB is not configured")
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000)
    try:
        _client.admin.command("ping")
    except Exception as error:
        logger.warning("MongoDB connection unavailable: %s", type(error).__name__)
        raise MongoUnavailable("MongoDB is temporarily unavailable") from error
    return _client


def get_database():
    return connect_to_mongodb()[DATABASE_NAME]


def get_predictions_collection():
    collection = get_database()["predictions"]
    collection.create_index([("timestamp", DESCENDING)])
    collection.create_index([("dataset", ASCENDING)])
    collection.create_index([("model", ASCENDING)])
    collection.create_index([("prediction", ASCENDING)])
    return collection


def save_prediction(document):
    record = dict(document)
    record.setdefault("timestamp", datetime.now(timezone.utc))
    result = get_predictions_collection().insert_one(record)
    return str(result.inserted_id)


def get_prediction_history(filters=None, page=1, per_page=10):
    filters = filters or {}
    query = {key: value for key, value in filters.items() if value not in (None, "", "all")}
    page = max(int(page), 1)
    per_page = min(max(int(per_page), 1), 50)
    collection = get_predictions_collection()
    cursor = collection.find(query).sort("timestamp", DESCENDING).skip((page - 1) * per_page).limit(per_page)
    records = []
    for record in cursor:
        record["_id"] = str(record["_id"])
        if isinstance(record.get("timestamp"), datetime):
            record["timestamp"] = record["timestamp"].astimezone(timezone.utc).isoformat()
        records.append(record)
    return {"records": records, "page": page, "per_page": per_page, "total": collection.count_documents(query)}


def get_prediction_by_id(prediction_id):
    if not ObjectId.is_valid(prediction_id):
        return None
    record = get_predictions_collection().find_one({"_id": ObjectId(prediction_id)})
    if record:
        record["_id"] = str(record["_id"])
        if isinstance(record.get("timestamp"), datetime):
            record["timestamp"] = record["timestamp"].astimezone(timezone.utc).isoformat()
    return record


def delete_prediction(prediction_id):
    if not ObjectId.is_valid(prediction_id):
        return False
    return get_predictions_collection().delete_one({"_id": ObjectId(prediction_id)}).deleted_count == 1


def clear_prediction_history():
    return get_predictions_collection().delete_many({}).deleted_count
