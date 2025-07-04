# app/services/face_recognition_insight.py
import numpy as np
import cv2
import insightface

# Initialize InsightFace model
_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
    try:
        # Try GPU first (CUDA)
        _model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        _model.prepare(ctx_id=0, det_size=(640, 640))
        print("[InsightFace] Using GPU (CUDAExecutionProvider)")
        return _model
    except Exception as e:
        print("[InsightFace] Failed to load GPU provider:", e)
    # Fallback: CPU only
    try:
        _model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=['CPUExecutionProvider']
        )
        _model.prepare(ctx_id=-1, det_size=(640, 640))
        print("[InsightFace] Using CPU (CPUExecutionProvider)")
        return _model
    except Exception as e:
        print("[InsightFace] Failed to load CPU provider:", e)
        raise RuntimeError("No suitable provider found for InsightFace model!")
    # return _model

# def extract_face_embedding(image_path: str):
#     """
#     Extract face embedding from image using InsightFace.
#     Returns: np.ndarray (embedding) or None if no face found
#     """
#     model = get_model()
#     img = cv2.imread(image_path)
#     if img is None:
#         return None
#     faces = model.get(img)
#     if not faces:
#         return None
#     # Use first detected face
#     return faces[0].embedding

# app/services/face_recognition_insight.py

def extract_face_embedding(image: np.ndarray):
    model = get_model()
    if image is None:
        return None
    faces = model.get(image)
    if not faces:
        return None
    emb = faces[0].embedding
    # L2 normalize
    emb = emb / np.linalg.norm(emb)
    return emb

def find_best_match_hybrid(unknown_embedding, people, threshold=1.3):
    """
    Hybrid: Accept match if either the average embedding OR any single embedding is below threshold.
    """
    best_match = None
    best_score = float("inf")
    for person in people:
        person['embeddings'] = [
            (np.array(e) / np.linalg.norm(e)).tolist() for e in person.get('embeddings', [])
        ]
        emb_list = person.get("embeddings", [])
        if not emb_list:
            continue

        # Compare to average embedding
        mean_emb = np.mean(np.array(emb_list), axis=0)
        dist_mean = np.linalg.norm(unknown_embedding - mean_emb)

        # Compare to each embedding
        min_dist = min(np.linalg.norm(unknown_embedding - np.array(emb)) for emb in emb_list)

        # Take the *best* score (lowest)
        score = min(dist_mean, min_dist)

        print(
            f"Hybrid compare {person['personInfo']['user_id']}: mean={dist_mean:.3f}, min={min_dist:.3f}, score={score:.3f}"
        )

        if score < best_score and score < threshold:
            best_score = score
            best_match = person

    if best_match:
        print(f"[HYBRID] Best match: {best_match['personInfo']['user_id']}, score={best_score}")
    else:
        print("[HYBRID] No match found.")
    return best_match
