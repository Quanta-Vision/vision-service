import numpy as np
import cv2
import os
import onnxruntime
import insightface

# Model path setup
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "models", "onnx", "anti-spoof-mn3.onnx"
)
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Load ONNX model once
session = onnxruntime.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

# Load InsightFace detector once
_insightface_detector = None

def get_insightface_detector():
    global _insightface_detector
    if _insightface_detector is None:
        _insightface_detector = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        _insightface_detector.prepare(ctx_id=0, det_size=(640, 640))
    return _insightface_detector

def detect_and_crop_face_insightface(image: np.ndarray, size=128):
    """Detect face and crop using InsightFace, return cropped face or None."""
    detector = get_insightface_detector()
    faces = detector.get(image)
    if not faces:
        return None
    # Take the largest face (by area)
    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
    x1, y1, x2, y2 = map(int, face.bbox)
    face_img = image[y1:y2, x1:x2]
    face_img = cv2.resize(face_img, (size, size))
    return face_img

def check_liveness_antispoof_mn3(image: np.ndarray) -> float:
    """
    Takes a BGR image, crops face using InsightFace, and returns liveness score.
    """
    face_img = detect_and_crop_face_insightface(image)
    if face_img is None:
        return -1.0
    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_norm = img_norm.transpose(2, 0, 1)[None]  # (1, 3, 128, 128)
    output = session.run(None, {'actual_input_1': img_norm})[0]
    score = float(output[0][0])
    return score  # 0 (spoof) .. 1 (live)
