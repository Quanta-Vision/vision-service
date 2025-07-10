import numpy as np
import cv2
import os
import onnxruntime
import insightface
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LivenessMethod(Enum):
    ONNX_MODEL = "onnx"
    INSIGHTFACE = "insightface"
    MULTI_FEATURE = "multi_feature"
    ENSEMBLE = "ensemble"

@dataclass
class LivenessResult:
    """Enhanced result structure for liveness detection"""
    is_live: bool
    confidence: float
    method_used: str
    detection_time: float
    face_quality: float
    spoof_indicators: Dict[str, float]
    recommendations: List[str]

class EnhancedLivenessDetector:
    """Enhanced liveness detection with multiple methods and improved accuracy"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._insightface_detector = None
        self._onnx_session = None
        self._antispoof_model = None
        self._face_detector_lock = threading.Lock()
        self._model_lock = threading.Lock()
        
        # Thresholds and parameters
        self.liveness_threshold = 0.5
        self.face_quality_threshold = 0.4
        self.min_face_size = 60
        self.max_face_size = 1200
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available models"""
        try:
            # Initialize InsightFace detector
            with self._face_detector_lock:
                if self._insightface_detector is None:
                    self._insightface_detector = insightface.app.FaceAnalysis(
                        name="buffalo_l", 
                        providers=['CPUExecutionProvider']
                    )
                    self._insightface_detector.prepare(ctx_id=0, det_size=(640, 640))
                    logger.info("InsightFace detector initialized successfully")
            
            # Initialize ONNX model if available
            if self.model_path and os.path.exists(self.model_path):
                self._onnx_session = onnxruntime.InferenceSession(
                    self.model_path, 
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"ONNX model loaded: {self.model_path}")
            
            # Try to initialize InsightFace antispoof model
            self._try_load_antispoof_model()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _try_load_antispoof_model(self):
        """Try to load InsightFace antispoof model"""
        try:
            possible_names = ["antispoof", "antifraud", "spoof", "liveness"]
            
            for name in possible_names:
                try:
                    model = insightface.model_zoo.get_model(name)
                    if model is not None:
                        model.prepare(ctx_id=0, input_size=(128, 128))
                        self._antispoof_model = model
                        logger.info(f"Antispoof model loaded: {name}")
                        break
                except Exception:
                    continue
                    
        except Exception as e:
            logger.warning(f"Could not load InsightFace antispoof model: {e}")
    
    def detect_and_analyze_face(self, image: np.ndarray) -> Optional[Dict]:
        """Enhanced face detection with quality analysis"""
        try:
            with self._face_detector_lock:
                faces = self._insightface_detector.get(image)
                logger.info(f"Detected {len(faces)} faces")
            
            if not faces:
                return None
            
            # Select the best face based on size and quality
            best_face = None
            best_score = 0
            
            for face in faces:
                bbox = face.bbox
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                face_area = face_width * face_height
                
                # Check face size constraints
                if face_width < self.min_face_size or face_height < self.min_face_size:
                    continue  # too small (likely background or artifact)
                if face_width > self.max_face_size or face_height > self.max_face_size:
                    logger.info(f"Skipping very large face: {face_width}x{face_height}")
                    continue  # very close-up face might be distorted
                
                # Calculate face quality score
                quality_score = self._calculate_face_quality(image, face)
                
                # Combined score considering area and quality
                combined_score = (face_area / 10000) * quality_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_face = face
            
            if best_face is None:
                return None
            
            # Extract face region with padding
            x1, y1, x2, y2 = map(int, best_face.bbox)
            h, w = image.shape[:2]
            
            # Dynamic padding based on face size
            padding = max(10, int(min(x2-x1, y2-y1) * 0.2))
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            face_img = image[y1:y2, x1:x2]
            
            return {
                'face_img': face_img,
                'bbox': (x1, y1, x2, y2),
                'quality': best_score,
                'landmarks': getattr(best_face, 'landmark', None),
                'age': getattr(best_face, 'age', None),
                'gender': getattr(best_face, 'gender', None)
            }
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return None
    
    def _calculate_face_quality(self, image: np.ndarray, face) -> float:
        """Calculate face quality score"""
        try:
            x1, y1, x2, y2 = map(int, face.bbox)
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500.0)
            
            # 2. Brightness adequacy
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            
            # 3. Contrast
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 50.0)
            
            # 4. Face size score
            face_area = (x2 - x1) * (y2 - y1)
            size_score = min(1.0, face_area / 40000.0)  # Optimal around 200x200
            
            # Weighted combination
            quality = (sharpness_score * 0.3 + brightness_score * 0.25 + 
                      contrast_score * 0.25 + size_score * 0.2)
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.error(f"Error calculating face quality: {e}")
            return 0.0
    
    def _advanced_spoof_detection(self, face_img: np.ndarray) -> Dict[str, float]:
        """Advanced spoof detection with multiple features"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            indicators = {}
            
            # 1. Texture Analysis
            # Local Binary Pattern variance
            def lbp_variance(image, radius=1, n_points=8):
                from skimage.feature import local_binary_pattern
                lbp = local_binary_pattern(image, n_points, radius, method='uniform')
                return np.var(lbp)
            
            try:
                lbp_var = lbp_variance(gray)
                indicators['texture_richness'] = min(1.0, lbp_var / 50.0)
            except ImportError:
                # Fallback to simpler texture measure
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                indicators['texture_richness'] = min(1.0, laplacian_var / 300.0)
            
            # 2. Color Analysis
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            
            # Color diversity
            h_entropy = self._calculate_entropy(hsv[:,:,0])
            s_entropy = self._calculate_entropy(hsv[:,:,1])
            v_entropy = self._calculate_entropy(hsv[:,:,2])
            indicators['color_diversity'] = (h_entropy + s_entropy + v_entropy) / 3.0
            
            # Skin color realism
            skin_score = self._evaluate_skin_color(face_img)
            indicators['skin_realism'] = skin_score
            
            # 3. Frequency Domain Analysis
            # High frequency content
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log(np.abs(f_shift) + 1)
            
            # Energy in high frequencies
            h, w = magnitude.shape
            center_y, center_x = h//2, w//2
            high_freq_mask = np.zeros((h, w))
            high_freq_mask[center_y-h//4:center_y+h//4, center_x-w//4:center_x+w//4] = 1
            high_freq_energy = np.sum(magnitude * (1 - high_freq_mask))
            indicators['high_freq_energy'] = min(1.0, high_freq_energy / 10000.0)
            
            # 4. Screen Detection Features
            # Moiré pattern detection
            moire_score = self._detect_moire_patterns(gray)
            indicators['moire_absence'] = 1.0 - moire_score
            
            # RGB correlation (screens have high correlation)
            b, g, r = cv2.split(face_img)
            correlations = [
                np.corrcoef(b.flatten(), g.flatten())[0, 1],
                np.corrcoef(b.flatten(), r.flatten())[0, 1],
                np.corrcoef(g.flatten(), r.flatten())[0, 1]
            ]
            avg_corr = np.mean([abs(c) for c in correlations if not np.isnan(c)])
            indicators['rgb_independence'] = 1.0 - min(1.0, avg_corr)
            
            # 5. Lighting Analysis
            # Natural lighting variation
            lighting_var = self._analyze_lighting_patterns(gray)
            indicators['natural_lighting'] = lighting_var
            
            # 6. Edge Analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            indicators['edge_naturalness'] = min(1.0, edge_density * 10)
            
            # 7. Reflection Detection
            reflection_score = self._detect_screen_reflections(face_img)
            indicators['reflection_absence'] = 1.0 - reflection_score
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error in advanced spoof detection: {e}")
            return {'error': 1.0}
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy"""
        try:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist[hist > 0]  # Remove zero entries
            hist = hist / hist.sum()  # Normalize
            entropy = -np.sum(hist * np.log2(hist))
            return entropy / 8.0  # Normalize to [0, 1]
        except Exception:
            return 0.0
    
    def _evaluate_skin_color(self, face_img: np.ndarray) -> float:
        """Evaluate skin color realism"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            
            # Skin color ranges in HSV
            # Typical skin hue range: 0-30 and 150-180
            h = hsv[:,:,0]
            s = hsv[:,:,1]
            v = hsv[:,:,2]
            
            # Create skin mask
            skin_mask1 = cv2.inRange(hsv, (0, 20, 20), (30, 255, 255))
            skin_mask2 = cv2.inRange(hsv, (150, 20, 20), (180, 255, 255))
            skin_mask = skin_mask1 | skin_mask2
            
            skin_ratio = np.sum(skin_mask > 0) / (face_img.shape[0] * face_img.shape[1])
            
            # Additional checks in LAB space
            l_mean = np.mean(lab[:,:,0])
            a_mean = np.mean(lab[:,:,1])
            b_mean = np.mean(lab[:,:,2])
            
            # Typical skin values in LAB
            l_score = 1.0 - abs(l_mean - 120) / 120.0
            a_score = 1.0 - abs(a_mean - 140) / 40.0
            b_score = 1.0 - abs(b_mean - 135) / 35.0
            
            combined_score = (skin_ratio * 0.4 + l_score * 0.2 + 
                             a_score * 0.2 + b_score * 0.2)
            
            return max(0.0, min(1.0, combined_score))
            
        except Exception as e:
            logger.error(f"Error evaluating skin color: {e}")
            return 0.5
    
    def _detect_moire_patterns(self, gray: np.ndarray) -> float:
        """Detect moiré patterns that indicate screen capture"""
        try:
            # Apply band-pass filter to detect periodic patterns
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(gray, -1, kernel)
            
            # Look for periodic patterns using FFT
            f_transform = np.fft.fft2(filtered)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Check for strong periodic components
            h, w = magnitude.shape
            center_y, center_x = h//2, w//2
            
            # Sample points around center in circular pattern
            angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
            radii = [min(h, w) // 8, min(h, w) // 4]
            
            max_magnitude = 0
            for radius in radii:
                for angle in angles:
                    y = int(center_y + radius * np.sin(angle))
                    x = int(center_x + radius * np.cos(angle))
                    if 0 <= y < h and 0 <= x < w:
                        max_magnitude = max(max_magnitude, magnitude[y, x])
            
            # Normalize and invert (higher moire = lower score)
            moire_score = min(1.0, max_magnitude / 1000.0)
            return moire_score
            
        except Exception as e:
            logger.error(f"Error detecting moire patterns: {e}")
            return 0.0
    
    def _analyze_lighting_patterns(self, gray: np.ndarray) -> float:
        """Analyze lighting patterns for naturalness"""
        try:
            # Apply Gaussian blur to get lighting map
            lighting_map = cv2.GaussianBlur(gray, (31, 31), 0)
            
            # Calculate lighting variation
            lighting_std = np.std(lighting_map)
            
            # Natural faces have gradual lighting changes
            # Screens tend to have more uniform lighting
            gradient_x = cv2.Sobel(lighting_map, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(lighting_map, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Natural lighting should have moderate gradients
            avg_gradient = np.mean(gradient_magnitude)
            
            # Score based on lighting variation and gradient
            lighting_score = min(1.0, lighting_std / 30.0)
            gradient_score = min(1.0, avg_gradient / 20.0)
            
            return (lighting_score + gradient_score) / 2.0
            
        except Exception as e:
            logger.error(f"Error analyzing lighting patterns: {e}")
            return 0.5
    
    def _detect_screen_reflections(self, face_img: np.ndarray) -> float:
        """Detect screen reflections and glare"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Look for bright spots that might be reflections
            bright_threshold = np.percentile(gray, 95)
            bright_mask = gray > bright_threshold
            
            # Find connected components of bright regions
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                bright_mask.astype(np.uint8), connectivity=8
            )
            
            # Analyze bright regions
            total_area = gray.shape[0] * gray.shape[1]
            bright_area_ratio = np.sum(bright_mask) / total_area
            
            # Large bright regions might indicate screen glare
            large_bright_regions = 0
            for i in range(1, num_labels):  # Skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if area > total_area * 0.02:  # > 2% of face area
                    large_bright_regions += 1
            
            # Screen reflection score
            reflection_score = (bright_area_ratio * 5.0 + 
                               large_bright_regions * 0.3)
            
            return min(1.0, reflection_score)
            
        except Exception as e:
            logger.error(f"Error detecting screen reflections: {e}")
            return 0.0
    
    def _ensemble_scoring(self, indicators: Dict[str, float]) -> float:
        """Improved ensemble scoring from multiple indicators"""
        try:
            # Rebalanced weights with more focus on screen artifact detection
            weights = {
                'texture_richness': 0.15,
                'color_diversity': 0.10,
                'skin_realism': 0.10,
                'high_freq_energy': 0.10,
                'moire_absence': 0.20,             # ↑ Stronger weight
                'rgb_independence': 0.15,          # ↑ Stronger weight
                'natural_lighting': 0.10,
                'edge_naturalness': 0.05,
                'reflection_absence': 0.15         # ↑ Stronger weight
            }

            total_score = 0.0
            total_weight = 0.0

            for indicator, value in indicators.items():
                if indicator in weights and not np.isnan(value):
                    total_score += value * weights[indicator]
                    total_weight += weights[indicator]

            final_score = total_score / total_weight if total_weight > 0 else 0.5
            logger.info("--- Spoof Indicators ---")
            for k, v in indicators.items():
                logger.info(f"{k:25}: {v:.3f}")

            return max(0.0, min(1.0, final_score))

        except Exception as e:
            logger.error(f"Error in ensemble scoring: {e}")
            return 0.5

    def check_liveness(self, image: np.ndarray, method: LivenessMethod = LivenessMethod.ENSEMBLE) -> LivenessResult:
        """Main liveness detection function with enhanced features"""
        start_time = time.time()
        
        try:
            # Detect and analyze face
            face_data = self.detect_and_analyze_face(image)
            if face_data is None:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    method_used="face_detection_failed",
                    detection_time=time.time() - start_time,
                    face_quality=0.0,
                    spoof_indicators={},
                    recommendations=["No face detected", "Ensure face is clearly visible", "Improve lighting"]
                )
            
            face_img = face_data['face_img']
            face_quality = face_data['quality']
            
            # Check face quality
            if face_quality < self.face_quality_threshold:
                return LivenessResult(
                    is_live=False,
                    confidence=0.0,
                    method_used="low_quality",
                    detection_time=time.time() - start_time,
                    face_quality=face_quality,
                    spoof_indicators={},
                    recommendations=["Face quality too low", "Move closer to camera", "Improve lighting"]
                )
            
            # Resize face for processing
            face_resized = cv2.resize(face_img, (128, 128))
            
            # Initialize results
            confidence = 0.0
            method_used = method.value
            spoof_indicators = {}
            recommendations = []
            
            # Method selection and execution
            if method == LivenessMethod.ONNX_MODEL and self._onnx_session:
                confidence = self._onnx_liveness_check(face_resized)
                method_used = "onnx_model"
                
            elif method == LivenessMethod.INSIGHTFACE and self._antispoof_model:
                confidence = self._insightface_liveness_check(face_resized)
                method_used = "insightface_model"
                
            elif method == LivenessMethod.MULTI_FEATURE or method == LivenessMethod.ENSEMBLE:
                # Advanced multi-feature analysis
                spoof_indicators = self._advanced_spoof_detection(face_resized)
                confidence = self._ensemble_scoring(spoof_indicators)
                method_used = "multi_feature_ensemble"
                
            else:
                # Fallback to basic analysis
                spoof_indicators = self._advanced_spoof_detection(face_resized)
                confidence = self._ensemble_scoring(spoof_indicators)
                method_used = "fallback_analysis"
            
            # Generate recommendations
            recommendations = self._generate_recommendations(confidence, spoof_indicators, face_quality)
            
            # Final decision
            is_live = confidence > self.liveness_threshold
            detection_time = time.time() - start_time
            
            return LivenessResult(
                is_live=is_live,
                confidence=confidence,
                method_used=method_used,
                detection_time=detection_time,
                face_quality=face_quality,
                spoof_indicators=spoof_indicators,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in liveness detection: {e}")
            return LivenessResult(
                is_live=False,
                confidence=0.0,
                method_used="error",
                detection_time=time.time() - start_time,
                face_quality=0.0,
                spoof_indicators={},
                recommendations=["Detection failed", "Please try again"]
            )
    
    def _onnx_liveness_check(self, face_img: np.ndarray) -> float:
        """ONNX model liveness check"""
        try:
            # Preprocess image
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_norm = img_norm.transpose(2, 0, 1)[None]
            
            # Run inference
            input_name = self._onnx_session.get_inputs()[0].name
            output = self._onnx_session.run(None, {input_name: img_norm})[0]
            score = float(output[0][0])
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error in ONNX liveness check: {e}")
            return 0.0
    
    def _insightface_liveness_check(self, face_img: np.ndarray) -> float:
        """InsightFace model liveness check"""
        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            score = self._antispoof_model.detect_liveness(face_rgb)
            return max(0.0, min(1.0, float(score)))
            
        except Exception as e:
            logger.error(f"Error in InsightFace liveness check: {e}")
            return 0.0
    
    def _generate_recommendations(self, confidence: float, indicators: Dict[str, float], face_quality: float) -> List[str]:
        """Generate user recommendations based on detection results"""
        recommendations = []
        
        if confidence < 0.3:
            recommendations.append("Very low liveness confidence - likely spoofing attempt")
        elif confidence < 0.6:
            recommendations.append("Low liveness confidence - please retry")
        
        if face_quality < 0.6:
            recommendations.append("Improve image quality - move closer or improve lighting")
        
        # Specific recommendations based on indicators
        if indicators.get('texture_richness', 1.0) < 0.4:
            recommendations.append("Image appears too smooth - ensure natural facial texture")
        
        if indicators.get('skin_realism', 1.0) < 0.4:
            recommendations.append("Skin color appears unnatural - check lighting conditions")
        
        if indicators.get('rgb_independence', 1.0) < 0.3:
            recommendations.append("Possible screen detection - avoid displaying photos on screen")
        
        if indicators.get('reflection_absence', 1.0) < 0.3:
            recommendations.append("Screen glare detected - avoid reflective surfaces")
        
        if indicators.get('moire_absence', 1.0) < 0.3:
            recommendations.append("Moiré patterns detected - avoid photographing screens")
        
        if not recommendations:
            if confidence > 0.8:
                recommendations.append("High confidence live detection")
            else:
                recommendations.append("Moderate confidence - consider retrying for better results")
        
        return recommendations

# Global instance
_detector_instance = None

def get_detector_instance(model_path: Optional[str] = None) -> EnhancedLivenessDetector:
    """Get singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EnhancedLivenessDetector(model_path)
    return _detector_instance

def check_liveness_antispoof_mn3(image: np.ndarray, model_path: Optional[str] = None) -> float:
    """
    Enhanced liveness detection function (backward compatible)
    Returns simple float score for compatibility
    """
    detector = get_detector_instance(model_path)
    result = detector.check_liveness(image, LivenessMethod.ENSEMBLE)
    
    # Log detailed results
    logger.info(f"Liveness Detection Results:")
    logger.info(f"  - Is Live: {result.is_live}")
    logger.info(f"  - Confidence: {result.confidence:.3f}")
    logger.info(f"  - Method: {result.method_used}")
    logger.info(f"  - Detection Time: {result.detection_time:.3f}s")
    logger.info(f"  - Face Quality: {result.face_quality:.3f}")
    logger.info(f"  - Recommendations: {result.recommendations}")
    
    return result.confidence if result.confidence != -1 else -1.0

def check_liveness_enhanced(image: np.ndarray, model_path: Optional[str] = None) -> LivenessResult:
    """
    Enhanced liveness detection with detailed results
    """
    detector = get_detector_instance(model_path)
    return detector.check_liveness(image, LivenessMethod.ENSEMBLE)