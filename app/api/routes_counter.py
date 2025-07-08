from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO
import torch

router_counter = APIRouter()

device = "cuda" if torch.cuda.is_available() else "cpu"

@router_counter.get("/", tags=["System Health"], summary="Health Check Counter")
def health_check():
    return {"status": "ok", "message": "vision-service counter is running"}

# person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
# traffic light, fire hydrant, stop sign, parking meter, bench, bird,
# cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe,
# backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
# sports ball, kite, baseball bat, skateboard, surfboard, tennis racket,
# bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
# sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake,
# chair, couch, potted plant, bed, dining table, toilet, tv, laptop,
# mouse, remote, keyboard, cell phone, microwave, oven, toaster,
# sink, refrigerator, book, clock, vase, scissors, teddy bear,
# hair drier, toothbrush
@router_counter.post("/object-count", tags=["Counter"], summary="Object counting")
async def counter_task(
    target_object: str = Form(...),
    model_name: str = Form("yolov8m.pt"),
    image: UploadFile = File(...)
):
    # Load model dynamically
    try:
        model = YOLO(model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {model_name}. Error: {str(e)}")

    # Decode image
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run inference
    results = model(img)[0]  # One image -> first result
    names = model.names

    # Count target
    count = 0
    detections = []
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls_id = box
        label = names[int(cls_id)]
        if label == target_object:
            count += 1

        detections.append({
            "label": label,
            "confidence": float(score),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })

    return {
        "target_object": target_object,
        "count": count,
        "detections": detections
    }
 