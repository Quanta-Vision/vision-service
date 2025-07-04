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
    """Returns status of the vision service."""
    return {"status": "ok", "message": "vision service is running"}

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
