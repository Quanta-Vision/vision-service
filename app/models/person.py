# app/models/person.py

from pymongo import MongoClient
from app.core.config import MONGODB_URL

client = MongoClient(MONGODB_URL)
db = client["rollcall"]
collection = db["people"]

def add_person(person_info: dict, image_paths: list, embeddings: list):
    doc = {
        "personInfo": person_info,
        "images": image_paths,
        "embeddings": embeddings
    }
    collection.insert_one(doc)

def get_people_by_consumer(consumer: str):
    return list(collection.find({"personInfo.consumer": consumer}, {"_id": 0}))

def find_person_by_user_id(user_id: str, consumer: str):
    return collection.find_one({"personInfo.user_id": user_id, "personInfo.consumer": consumer}, {"_id": 0})

def update_person(user_id: str, image_paths, embeddings, consumer: str):
    return collection.update_one(
        {"personInfo.user_id": user_id, "personInfo.consumer": consumer},
        {"$set": {"images": image_paths, "embeddings": embeddings}}
    )

def delete_person_by_user_id(user_id: str, consumer: str):
    result = collection.delete_one({"personInfo.user_id": user_id, "personInfo.consumer": consumer})
    return result.deleted_count

def delete_people_by_user_ids(user_ids: list, consumer: str):
    result = collection.delete_many({"personInfo.user_id": {"$in": user_ids}, "personInfo.consumer": consumer})
    return result.deleted_count
