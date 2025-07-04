from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request
from typing import List, Optional
from datetime import datetime
import time
import json

from app.core.storage import save_image
from app.models.person import (
    add_person,
    delete_people_by_user_ids,
    find_person_by_user_id,
    get_people_by_consumer,
    update_person,
    delete_person_by_user_id
)
from app.services.face_liveness_antispoof import check_liveness_antispoof_mn3
from app.services.face_recognition_insight import extract_face_embedding, find_best_match_hybrid
from app.utils.auth import verify_api_key
from app.utils.utils_function import uploadfile_to_cv2_image
from app.utils.consumer import get_consumer
from pydantic import BaseModel

router_v2 = APIRouter()

@router_v2.get("/", tags=["System Health"], summary="Health Check (v2)")
def health_check():
    return {"status": "ok", "message": "vision-service v2 is running"}

@router_v2.get("/persons", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def list_persons_api(request: Request):
    consumer = get_consumer(request)
    people = get_people_by_consumer(consumer)
    return {"users": people, "count": len(people)}

@router_v2.get("/person/{user_id}", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def get_person_api(user_id: str, request: Request):
    consumer = get_consumer(request)
    person = find_person_by_user_id(user_id, consumer)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")
    return {"user": person}

@router_v2.post("/add-person", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def add_person_api(
    request: Request,
    name: str = Form(...),
    user_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    consumer = get_consumer(request)
    image_paths = []
    embeddings = []
    import cv2

    for img in images:
        path = save_image(img, user_id)
        image_paths.append(path)
        image_np = cv2.imread(path)
        if image_np is None:
            raise HTTPException(status_code=400, detail=f"Cannot read saved image {img.filename}")
        emb = extract_face_embedding(image_np)
        if emb is None:
            raise HTTPException(status_code=400, detail=f"No face detected in {img.filename}")
        embeddings.append(emb.tolist())

    person_info = {"name": name, "user_id": user_id, "consumer": consumer}
    add_person(person_info, image_paths, embeddings)
    return {"msg": "Person Rollcalled successfully (v2)."}

@router_v2.put("/update-person", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def update_person_api(
    request: Request,
    user_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    consumer = get_consumer(request)
    person = find_person_by_user_id(user_id, consumer)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")

    image_paths = []
    embeddings = []
    import cv2

    for img in images:
        path = save_image(img, user_id)
        image_paths.append(path)
        image_np = cv2.imread(path)
        if image_np is None:
            raise HTTPException(status_code=400, detail=f"Cannot read saved image {img.filename}")
        emb = extract_face_embedding(image_np)
        if emb is None:
            raise HTTPException(status_code=400, detail=f"No face detected in {img.filename}")
        embeddings.append(emb.tolist())

    update_person(user_id, image_paths, embeddings, consumer)
    return {"msg": "Person images updated successfully (v2)."}

@router_v2.post("/recognize", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def recognize_person(request: Request, image: UploadFile = File(...)):
    consumer = get_consumer(request)
    img_cv2 = await uploadfile_to_cv2_image(image)
    unknown_emb = extract_face_embedding(img_cv2)
    if unknown_emb is None:
        return {
            "msg": "No face detected in uploaded image",
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }
    all_people = get_people_by_consumer(consumer)
    matched = find_best_match_hybrid(unknown_emb, all_people)

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
async def delete_person_api(request: Request, user_id: str = Form(...)):
    consumer = get_consumer(request)
    deleted = delete_person_by_user_id(user_id, consumer)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Person not found or already deleted.")
    return {"msg": f"Person with user_id {user_id} deleted successfully (v2)."}

class UserIdList(BaseModel):
    user_ids: List[str]

@router_v2.post("/delete-persons", tags=["Person Rollcall (v2)"], dependencies=[Depends(verify_api_key)])
async def delete_persons_api(request: Request, body: UserIdList):
    consumer = get_consumer(request)
    deleted_count = delete_people_by_user_ids(body.user_ids, consumer)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="No users were deleted (user_ids not found).")
    return {"msg": f"Deleted {deleted_count} users successfully (v2)."}

@router_v2.post("/check-spoofing-mn3", tags=["Spoofing/Liveness"], dependencies=[Depends(verify_api_key)])
async def check_spoofing_mn3(image: UploadFile = File(...)):
    img_cv2 = await uploadfile_to_cv2_image(image)
    score = check_liveness_antispoof_mn3(img_cv2)
    if score == -1.0:
        return {
            "is_live": False,
            "score": score,
            "threshold": 0.5,
            "msg": "No face detected"
        }
    return {
        "is_live": bool(score > 0.5),
        "score": float(score),
        "threshold": 0.5
    }
