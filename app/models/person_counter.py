from pymongo import MongoClient
from datetime import datetime
from app.core.config import MONGODB_URL

# MongoDB Setup
client = MongoClient(MONGODB_URL)
db = client["rollcall"]
count_logs = db["people_count_logs"]

def log_people_count_to_db(image_path: str, count: int, boxes: list, metadata: dict = None):
    doc = {
        "image_path": image_path,
        "count": count,
        "boxes": boxes,  # list of [x1, y1, x2, y2]
        "timestamp": datetime.utcnow(),
    }
    if metadata:
        doc["metadata"] = metadata
    count_logs.insert_one(doc)
