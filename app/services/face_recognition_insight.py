# app/services/face_recognition_insight.py
import numpy as np
import cv2
import insightface

# Initialize InsightFace model
_model = None

def get_model():
    global _model
    if _model is None:
        # Use 'buffalo_l' for best accuracy; ctx_id=0 uses CPU
        _model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        _model.prepare(ctx_id=0, det_size=(640, 640))
    return _model

def extract_face_embedding(image_path: str):
    """
    Extract face embedding from image using InsightFace.
    Returns: np.ndarray (embedding) or None if no face found
    """
    model = get_model()
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = model.get(img)
    if not faces:
        return None
    # Use first detected face
    return faces[0].embedding

def find_best_match(unknown_embedding, people, threshold=0.5):
    """
    Compare unknown embedding with people's embeddings.
    Returns: best matching person (dict) or None
    """
    best_match = None
    best_score = float("inf")
    for person in people:
        for emb in person.get("embeddings", []):
            dist = np.linalg.norm(unknown_embedding - np.array(emb))
            if dist < best_score and dist < threshold:
                best_score = dist
                best_match = person
    return best_match
