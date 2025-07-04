# app/models/api_keys.py

from pymongo import MongoClient
from app.core.config import MONGODB_URL
from datetime import datetime

client = MongoClient(MONGODB_URL)
db = client["rollcall"]
collection = db["api_keys"]

def create_api_key(api_key, consumer_name, allowed_apis):
    doc = {
        "api_key": api_key,
        "consumer_name": consumer_name,
        "allowed_apis": allowed_apis,
        "created_at": datetime.utcnow(),
        "active": True
    }
    collection.insert_one(doc)

def find_api_key(api_key):
    return collection.find_one({"api_key": api_key, "active": True})

def deactivate_api_key(api_key):
    return collection.update_one({"api_key": api_key}, {"$set": {"active": False}})

def update_consumer(api_key, updates: dict):
    return collection.update_one({"api_key": api_key}, {"$set": updates})

def list_all_api_keys():
    return list(collection.find({}, {"_id": 0}))
