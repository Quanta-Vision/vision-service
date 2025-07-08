import numpy as np
import cv2
import os
import onnxruntime
import insightface

# First, let's check what models are available
def check_available_models():
    """Check what models are available in your InsightFace installation"""
    try:
        print("InsightFace model zoo models:")
        models = insightface.model_zoo.model_names
        print(models)
        return models
    except Exception as e:
        print(f"Error checking models: {e}")
        return []

# Method 1: Try to find the correct antispoof model
_antifraud_model = None

def get_antispoof_model():
    global _antifraud_model
    if _antifraud_model is None:
        try:
            # First check available models
            available_models = check_available_models()
            
            # Try different possible names for antispoof models
            possible_names = [
                "antispoof",
                "antifraud", 
                "spoof",
                "liveness",
                "anti_spoof",
                "anti_fraud",
                "face_liveness",
                "spoof_detection"
            ]
            
            model_found = False
            for name in possible_names:
                if name in available_models:
                    try:
                        print(f"Trying to load model: {name}")
                        _antifraud_model = insightface.model_zoo.get_model(name)
                        if _antifraud_model is not None:
                            _antifraud_model.prepare(ctx_id=0, input_size=(128, 128))
                            print(f"Successfully loaded antispoof model: {name}")
                            model_found = True
                            break
                    except Exception as e:
                        print(f"Failed to load {name}: {e}")
                        continue
            
            if not model_found:
                print("No antispoof model found in InsightFace model zoo.")
                print("Available models:", available_models)
                print("You may need to:")
                print("1. Install a specific InsightFace version that includes antispoof models")
                print("2. Download the model manually")
                print("3. Use the ONNX approach with your own model file")
                return None
                
        except Exception as e:
            print(f"Error loading antispoof model: {e}")
            return None
    
    return _antifraud_model

# Method 2: Use ONNX model directly (recommended if you have the model file)
def get_onnx_session(model_path):
    """Load ONNX model for antispoof detection"""
    try:
        if not os.path.exists(model_path):
            print(f"ONNX model file not found: {model_path}")
            return None
        
        session = onnxruntime.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        print(f"Successfully loaded ONNX model: {model_path}")
        return session
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None

# Load InsightFace detector
_insightface_detector = None

def get_insightface_detector():
    global _insightface_detector
    if _insightface_detector is None:
        try:
            _insightface_detector = insightface.app.FaceAnalysis(
                name="buffalo_l", 
                providers=['CPUExecutionProvider']
            )
            _insightface_detector.prepare(ctx_id=0, det_size=(640, 640))
            print("Successfully loaded InsightFace detector")
        except Exception as e:
            print(f"Error loading InsightFace detector: {e}")
            raise
    return _insightface_detector

def detect_and_crop_face_insightface(image: np.ndarray, size=128):
    """Detect face and crop using InsightFace"""
    try:
        detector = get_insightface_detector()
        faces = detector.get(image)
        if not faces:
            print("No face detected")
            return None
        
        # Take the largest face
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
        x1, y1, x2, y2 = map(int, face.bbox)
        
        # Add some padding and ensure we don't go out of bounds
        h, w = image.shape[:2]
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        face_img = image[y1:y2, x1:x2]
        if face_img.size == 0:
            return None
            
        face_img = cv2.resize(face_img, (size, size))
        print(f"Face detected and cropped: {face_img.shape}")
        return face_img
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None

# Method 1: Using InsightFace antispoof model
def check_liveness_insightface(image: np.ndarray) -> float:
    """Check liveness using InsightFace antispoof model"""
    try:
        face_img = detect_and_crop_face_insightface(image)
        if face_img is None:
            return -1.0
        
        model = get_antispoof_model()
        if model is None:
            print("Antispoof model not available")
            return -1.0
        
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        score = model.detect_liveness(face_rgb)
        print(f"InsightFace liveness score: {score}")
        return float(score)
    except Exception as e:
        print(f"Error in InsightFace liveness detection: {e}")
        return -1.0

# Method 2: Using ONNX model (recommended)
def check_liveness_onnx(image: np.ndarray, model_path: str) -> float:
    """Check liveness using ONNX model"""
    try:
        session = get_onnx_session(model_path)
        if session is None:
            return -1.0
        
        face_img = detect_and_crop_face_insightface(image)
        if face_img is None:
            return -1.0
        
        # Preprocess image for ONNX model
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_norm = img_norm.transpose(2, 0, 1)[None]  # (1, 3, 128, 128)
        
        print(f"Input shape: {img_norm.shape}")
        print(f"Input range: {img_norm.min():.3f} - {img_norm.max():.3f}")
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: img_norm})[0]
        score = float(output[0][0])
        
        print(f"ONNX liveness score: {score}")
        return score
        
    except Exception as e:
        print(f"Error in ONNX liveness detection: {e}")
        return -1.0

# Method 3: Improved rule-based antispoof (basic fallback)
def check_liveness_simple(image: np.ndarray) -> float:
    """Improved rule-based antispoof detection (basic fallback)"""
    try:
        face_img = detect_and_crop_face_insightface(image)
        if face_img is None:
            return -1.0
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Texture analysis using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Color distribution analysis
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        h_std = np.std(hsv[:,:,0])
        s_std = np.std(hsv[:,:,1])
        v_std = np.std(hsv[:,:,2])
        
        # 3. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # 4. Screen/display detection features
        # Check for screen patterns, uniform lighting, etc.
        
        # 4a. Check for screen-like uniform illumination
        # Screens tend to have more uniform lighting
        illumination_std = np.std(cv2.GaussianBlur(gray, (15, 15), 0))
        
        # 4b. Check for RGB channel correlation (screens often have high correlation)
        b, g, r = cv2.split(face_img)
        bg_corr = np.corrcoef(b.flatten(), g.flatten())[0, 1]
        br_corr = np.corrcoef(b.flatten(), r.flatten())[0, 1]
        gr_corr = np.corrcoef(g.flatten(), r.flatten())[0, 1]
        avg_corr = (bg_corr + br_corr + gr_corr) / 3
        
        # 4c. Check for high frequency noise (real faces have more natural noise)
        # Apply high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(gray, -1, kernel)
        high_freq_energy = np.std(high_freq)
        
        # 4d. Check for screen reflection/glare patterns
        # Look for bright spots that might indicate screen reflection
        bright_pixels = np.sum(gray > 200) / (gray.shape[0] * gray.shape[1])
        
        print(f"Laplacian variance: {laplacian_var:.2f}")
        print(f"Color std (H,S,V): {h_std:.2f}, {s_std:.2f}, {v_std:.2f}")
        print(f"Edge density: {edge_density:.4f}")
        print(f"Illumination std: {illumination_std:.2f}")
        print(f"RGB correlation: {avg_corr:.3f}")
        print(f"High freq energy: {high_freq_energy:.2f}")
        print(f"Bright pixels ratio: {bright_pixels:.4f}")
        
        # Improved scoring with spoof detection
        score = 0.5  # Start with neutral score
        
        # Positive indicators (real face)
        if laplacian_var > 150:  # Good texture variation
            score += 0.15
        if h_std > 8:  # Good color variation in hue
            score += 0.1
        if s_std > 15:  # Good saturation variation
            score += 0.1
        if v_std > 20:  # Good value variation
            score += 0.1
        if high_freq_energy > 5:  # Natural high frequency content
            score += 0.1
        if illumination_std > 20:  # Natural lighting variation
            score += 0.1
        
        # Negative indicators (likely spoof/screen)
        if avg_corr > 0.85:  # High RGB correlation suggests screen
            score -= 0.3
        if bright_pixels > 0.1:  # Too many bright pixels (screen glare)
            score -= 0.2
        if illumination_std < 10:  # Too uniform lighting
            score -= 0.2
        if edge_density < 0.05:  # Too few edges
            score -= 0.1
        if laplacian_var < 50:  # Too smooth/blurry
            score -= 0.3
        
        # Additional check for screen-like characteristics
        # If image is too "perfect" (high quality but artificial), reduce score
        if (laplacian_var > 100 and avg_corr > 0.8 and 
            illumination_std < 15 and bright_pixels > 0.05):
            score -= 0.4
            print("Screen-like characteristics detected")
        
        final_score = max(0.0, min(1.0, score))
        print(f"Simple rule-based score: {final_score}")
        return final_score
        
    except Exception as e:
        print(f"Error in simple liveness detection: {e}")
        return -1.0

# Main function that tries different methods
def check_liveness_antispoof_mn3(image: np.ndarray, model_path: str = None) -> float:
    """
    Main liveness detection function that tries different methods
    """
    print("Starting liveness detection...")
    
    # Method 1: Try ONNX model if path provided
    if model_path and os.path.exists(model_path):
        print("Trying ONNX model...")
        score = check_liveness_onnx(image, model_path)
        if score != -1.0:
            return score
    
    # Method 2: Try InsightFace antispoof model
    print("Trying InsightFace antispoof model...")
    score = check_liveness_insightface(image)
    if score != -1.0:
        return score
    
    # Method 3: Fallback to simple rule-based detection
    print("Falling back to simple rule-based detection...")
    score = check_liveness_simple(image)
    
    return score
