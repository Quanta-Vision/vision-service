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

def get_all_people():
    return list(collection.find({}, {"_id": 0}))
