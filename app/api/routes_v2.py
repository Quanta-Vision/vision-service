# app/api/routes_v2.py
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import List, Optional
from datetime import datetime
import time
import json

from app.core.storage import save_image
from app.models.person import add_person, find_person_by_user_id, get_all_people, update_person
from app.services.face_recognition_insight import extract_face_embedding, find_best_match
from app.services.person_counter import count_people_in_image
from app.utils.auth import verify_api_key

router_v2 = APIRouter()

@router_v2.get("/", tags=["System Health"], summary="Health Check (v2)")
def health_check():
    return {"status": "ok", "message": "e-document-ai v2 is running"}

@router_v2.post("/add-person", tags=["Person Register (v2)"], dependencies=[Depends(verify_api_key)])
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
        if emb is None:
            raise HTTPException(status_code=400, detail=f"No face detected in {img.filename}")
        embeddings.append(emb.tolist())

    person_info = {"name": name, "user_id": user_id}
    add_person(person_info, image_paths, embeddings)
    return {"msg": "Person registered successfully (v2)."}

@router_v2.put("/update-person", tags=["Person Register (v2)"], dependencies=[Depends(verify_api_key)])
async def update_person_api(
    user_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 10 images per person.")

    person = find_person_by_user_id(user_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")

    image_paths = []
    embeddings = []
    for img in images:
        path = save_image(img, user_id)
        image_paths.append(path)
        emb = extract_face_embedding(path)
        if emb is None:
            raise HTTPException(status_code=400, detail=f"No face detected in {img.filename}")
        embeddings.append(emb.tolist())

    update_person(user_id, image_paths, embeddings)
    return {"msg": "Person images updated successfully (v2)."}

@router_v2.post("/recognize", tags=["Person Register (v2)"], dependencies=[Depends(verify_api_key)])
async def recognize_person(image: UploadFile = File(...)):
    path = save_image(image, "temp")
    unknown_emb = extract_face_embedding(path)
    if unknown_emb is None:
        return {
            "msg": "No face detected in uploaded image",
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }
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

@router_v2.post("/count-person", tags=["Object Count (v2)"], summary="Person Counting (v2)", dependencies=[Depends(verify_api_key)])
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
