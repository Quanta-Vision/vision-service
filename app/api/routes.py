from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from datetime import datetime
import time

from app.core.storage import save_image
from app.models.person import add_person, find_person_by_user_id, get_all_people, update_person
from app.services.face_recognition import extract_face_embedding, find_best_match

router = APIRouter()

@router.post("/add-person")
async def add_person_api(
    name: str = Form(...),
    user_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 10 images per person.")

    image_paths = []
    embeddings = []

    for img in images:
        path = save_image(img, user_id)
        image_paths.append(path)
        emb = extract_face_embedding(path)
        embeddings.append(emb.tolist())

    person_info = {"name": name, "user_id": user_id}
    add_person(person_info, image_paths, embeddings)
    return {"msg": "Person registered successfully."}

@router.put("/update-person")
async def update_person_api(
    user_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 10 images per person.")

    person = find_person_by_user_id(user_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")

    # Save new images
    image_paths = []
    embeddings = []
    for img in images:
        path = save_image(img, user_id)
        image_paths.append(path)
        emb = extract_face_embedding(path)
        embeddings.append(emb.tolist())

    # Update in DB
    update_person(user_id, image_paths, embeddings)

    return {"msg": "Person images updated successfully."}


@router.post("/recognize")
async def recognize_person(image: UploadFile = File(...)):
    path = save_image(image, "temp")
    unknown_emb = extract_face_embedding(path)
    all_people = get_all_people()
    matched = find_best_match(unknown_emb, all_people)

    if matched:
        now = datetime.utcnow()
        timestamp_ms = int(time.time() * 1000)
        return {
            "personInfo": matched["personInfo"],
            "timestamp": timestamp_ms,
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S")
        }

    return {
        "msg": "No match found",
        "timestamp": int(time.time() * 1000),
        "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }