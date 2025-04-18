from ultralytics import YOLO
import cv2
import os
import time
import math

from app.models.person_counter import log_people_count_to_db

# Load YOLOv8 model once
model = YOLO("yolov8s.pt")  # You can also try yolov8m.pt or yolov8x.pt for better accuracy

# ------------------------------
# Utility Functions
# ------------------------------

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea)

def compute_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def center_distance(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def strict_deduplication(new_box, existing_boxes, iou_thresh=0.6, center_thresh=50, area_thresh=0.25):
    x1, y1, x2, y2 = new_box
    new_center = compute_center(new_box)
    new_area = (x2 - x1) * (y2 - y1)

    for ex in existing_boxes:
        ex_center = compute_center(ex)
        ex_area = (ex[2] - ex[0]) * (ex[3] - ex[1])
        iou = compute_iou(new_box, ex)
        center_dist = center_distance(new_center, ex_center)
        area_diff = abs(new_area - ex_area) / max(new_area, ex_area)

        if iou > iou_thresh or (center_dist < center_thresh and area_diff < area_thresh):
            return True  # This box is likely a duplicate

    return False

# ------------------------------
# Main People Detection Function
# ------------------------------

def count_people_in_image(image_path: str, metadata: dict = None) -> int:
    results = model(image_path, conf=0.25)
    person_boxes = []

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # 'person' class in COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_w, box_h = x2 - x1, y2 - y1

                # Skip small boxes (likely false positives)
                if box_w < 40 or box_h < 40:
                    continue

                new_box = [x1, y1, x2, y2]
                if not strict_deduplication(new_box, person_boxes):
                    person_boxes.append(new_box)
                    
    # Print image result
    # print_image_count_result(image_path, person_boxes)
    
    # Log into database
    # log_people_count_to_db(image_path, len(person_boxes), person_boxes, metadata)
    
    return len(person_boxes)

# ------------------------------
# Drawing / Visualization
# ------------------------------

def print_image_count_result(image_path: str, boxes, output_path: str = None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    if output_path is None:
        base = os.path.basename(image_path)
        name, ext = os.path.splitext(base)
        output_path = f"{name}_detected_{int(time.time())}.jpg"

    for idx, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        label = f"Person {idx}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"[INFO] Detected {len(boxes)} people")
    print(f"[INFO] Saved annotated image to: {output_path}")
