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

def find_person_by_user_id(user_id):
    return collection.find_one({"personInfo.user_id": user_id}, {"_id": 0})

def update_person(user_id, image_paths, embeddings):
    collection.update_one(
        {"personInfo.user_id": user_id},
        {
            "$set": {
                "images": image_paths,
                "embeddings": embeddings
            }
        }
    )

def get_all_people():
    return list(collection.find({}, {"_id": 0}))

def delete_person_by_user_id(user_id):
    result = collection.delete_one({"personInfo.user_id": user_id})
    return result.deleted_count

def delete_people_by_user_ids(user_ids):
    result = collection.delete_many({"personInfo.user_id": {"$in": user_ids}})
    return result.deleted_count
