# app/api/routes_v2.py
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import List, Optional
from datetime import datetime
import time
import json

from app.core.storage import save_image
from app.models.person import add_person, delete_people_by_user_ids, find_person_by_user_id, get_all_people, update_person
from app.services.face_recognition_insight import extract_face_embedding, find_best_match
from app.models.person import delete_person_by_user_id
from app.utils.auth import verify_api_key
router_v2 = APIRouter()

import numpy as np
import cv2

async def uploadfile_to_cv2_image(file: UploadFile):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

@router_v2.get("/", tags=["System Health"], summary="Health Check (v2)")
def health_check():
    return {"status": "ok", "message": "vision-service v2 is running"}

@router_v2.get("/persons", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def list_persons_api():
    people = get_all_people()
    return {"users": people, "count": len(people)}

@router_v2.get("/person/{user_id}", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def get_person_api(user_id: str):
    person = find_person_by_user_id(user_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")
    return {"user": person}

@router_v2.post("/add-person", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
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
    return {"msg": "Person Rollcalled successfully (v2)."}

@router_v2.put("/update-person", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
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

@router_v2.post("/recognize", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def recognize_person(image: UploadFile = File(...)):
    img_cv2 = await uploadfile_to_cv2_image(image)
    unknown_emb = extract_face_embedding(img_cv2)
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

@router_v2.delete("/delete-person", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def delete_person_api(user_id: str = Form(...)):
    deleted = delete_person_by_user_id(user_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Person not found or already deleted.")
    return {"msg": f"Person with user_id {user_id} deleted successfully (v2)."}

# Delete persons
from pydantic import BaseModel
from typing import List

class UserIdList(BaseModel):
    user_ids: List[str]

@router_v2.post("/delete-persons", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def delete_persons_api(body: UserIdList):
    deleted_count = delete_people_by_user_ids(body.user_ids)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="No users were deleted (user_ids not found).")
    return {"msg": f"Deleted {deleted_count} users successfully (v2)."}
