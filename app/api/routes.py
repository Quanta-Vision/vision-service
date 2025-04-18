from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import List, Optional
from datetime import datetime
import time
import json

from app.core.storage import save_image
from app.models.person import add_person, find_person_by_user_id, get_all_people, update_person
from app.services.face_recognition import extract_face_embedding, find_best_match
from app.services.person_counter import count_people_in_image
from app.utils.auth import verify_api_key

router = APIRouter()

@router.get("/", tags=["System Health"], summary="Health Check")
def health_check():
    """Returns status of the e-document AI service."""
    return {"status": "ok", "message": "e-document-ai is running"}

@router.post("/add-person", tags=["Person Regconite"], dependencies=[Depends(verify_api_key)])
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

@router.put("/update-person", tags=["Person Regconite"], dependencies=[Depends(verify_api_key)])
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

@router.post("/recognize", tags=["Person Regconite"], dependencies=[Depends(verify_api_key)])
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

@router.post("/count-person", tags=["Object Count"], summary="Person Counting", dependencies=[Depends(verify_api_key)])
async def count_person_api(
    image: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    try:
        path = save_image(image, "people_count")

        meta_obj = {}
        if metadata:
            try:
                meta_obj = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON.")

        count = count_people_in_image(path, metadata=meta_obj)

        return {
            "peopleCount": count,
            "metadata": meta_obj,
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
