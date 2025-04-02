from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from app.core.storage import save_image
from app.models.person import add_person, get_all_people
from app.services.face_recognition import extract_face_embedding, find_best_match

router = APIRouter()

@router.post("/add-person")
async def add_person_api(
    name: str = Form(...),
    user_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    image_paths = []
    embeddings = []
    for img in images:
        path = save_image(img, user_id)
        image_paths.append(path)
        emb = extract_face_embedding(path)
        embeddings.append(emb.tolist())  # Convert to JSON-serializable

    person_info = {"name": name, "user_id": user_id}
    add_person(person_info, image_paths, embeddings)
    return {"msg": "Person added successfully"}

@router.post("/recognize")
async def recognize_person(image: UploadFile = File(...)):
    path = save_image(image, "temp")
    unknown_emb = extract_face_embedding(path)
    all_people = get_all_people()
    matched = find_best_match(unknown_emb, all_people)

    if matched:
        return matched["personInfo"]
    return {"msg": "No match found"}
