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
    """Optimized liveness detection with configurable sensitivity"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._insightface_detector = None
        self._onnx_session = None
        self._antispoof_model = None
        self._face_detector_lock = threading.Lock()
        self._model_lock = threading.Lock()
        
        # Configurable thresholds - can be adjusted based on your needs
        self.liveness_threshold = 0.45
        self._screen_penalty_multiplier = 0.6
        self._natural_boost_multiplier = 1.15
        self.face_quality_threshold = 0.25
        self.min_face_size = 60
        self.max_face_size = 1200
        
        # Sensitivity settings - can be modified
        self.sensitivity_mode = "lenient"  # "strict", "balanced", "lenient"
        
        # Initialize models
        self._initialize_models()
    
    def set_sensitivity(self, mode: str):
        """Set detection sensitivity mode"""
        if mode == "strict":
            self.liveness_threshold = 0.70
            self._screen_penalty_multiplier = 1.0
            self._natural_boost_multiplier = 1.0
        elif mode == "balanced":
            self.liveness_threshold = 0.58
            self._screen_penalty_multiplier = 0.8
            self._natural_boost_multiplier = 1.05
        elif mode == "lenient":
            self.liveness_threshold = 0.45
            self._screen_penalty_multiplier = 0.6
            self._natural_boost_multiplier = 1.15
        
        self.sensitivity_mode = mode
        logger.info(f"Sensitivity mode set to: {mode}, threshold: {self.liveness_threshold}")
    
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
            
            # Set default sensitivity
            self.set_sensitivity("balanced")
            
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
                    continue
                if face_width > self.max_face_size or face_height > self.max_face_size:
                    logger.info(f"Skipping very large face: {face_width}x{face_height}")
                    continue
                
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
            size_score = min(1.0, face_area / 40000.0)
            
            # Weighted combination
            quality = (sharpness_score * 0.3 + brightness_score * 0.25 + 
                      contrast_score * 0.25 + size_score * 0.2)
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.error(f"Error calculating face quality: {e}")
            return 0.0
    
    def _advanced_spoof_detection(self, face_img: np.ndarray) -> Dict[str, float]:
        """Advanced spoof detection with configurable sensitivity"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            indicators = {}
            
            # 1. Texture Analysis - adjusted for sensitivity
            texture_scores = []
            for scale in [3, 5, 7]:
                kernel = np.ones((scale, scale), np.float32) / (scale * scale)
                smooth = cv2.filter2D(gray, -1, kernel)
                texture = cv2.absdiff(gray, smooth)
                texture_scores.append(np.std(texture))
            
            avg_texture = np.mean(texture_scores)
            # Adjust threshold based on sensitivity
            texture_threshold = 22.0 if self.sensitivity_mode == "strict" else (20.0 if self.sensitivity_mode == "balanced" else 18.0)
            indicators['texture_richness'] = min(1.0, avg_texture / texture_threshold)
            
            # 2. Screen Pattern Detection - configurable
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log(np.abs(f_shift) + 1)
            
            h, w = magnitude.shape
            center_y, center_x = h//2, w//2
            
            screen_pattern_score = 0
            for radius in range(8, min(h, w)//4, 4):
                ring_mask = np.zeros((h, w))
                y, x = np.ogrid[:h, :w]
                mask = ((x - center_x)**2 + (y - center_y)**2) >= radius**2
                mask &= ((x - center_x)**2 + (y - center_y)**2) < (radius + 3)**2
                ring_mask[mask] = 1
                
                ring_energy = np.sum(magnitude * ring_mask)
                if ring_energy > screen_pattern_score:
                    screen_pattern_score = ring_energy
            
            # Adjust threshold based on sensitivity
            pattern_threshold = 5200.0 if self.sensitivity_mode == "strict" else (5500.0 if self.sensitivity_mode == "balanced" else 6000.0)
            indicators['screen_pattern_absence'] = 1.0 - min(1.0, screen_pattern_score / pattern_threshold)
            
            # 3. RGB Channel Analysis - configurable
            b, g, r = cv2.split(face_img)
            
            corr_rg = np.corrcoef(r.flatten(), g.flatten())[0, 1]
            corr_rb = np.corrcoef(r.flatten(), b.flatten())[0, 1]
            corr_gb = np.corrcoef(g.flatten(), b.flatten())[0, 1]
            
            correlations = [abs(c) for c in [corr_rg, corr_rb, corr_gb] if not np.isnan(c)]
            avg_correlation = np.mean(correlations) if correlations else 0.8
            
            # Adjust sensitivity for RGB correlation
            rgb_multiplier = 0.95 if self.sensitivity_mode == "strict" else (0.9 if self.sensitivity_mode == "balanced" else 0.85)
            indicators['rgb_independence'] = 1.0 - min(1.0, avg_correlation * rgb_multiplier)
            
            # 4. Enhanced Color Analysis
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            
            skin_score = self._evaluate_skin_color_enhanced(face_img, hsv, lab)
            indicators['skin_realism'] = skin_score
            
            h_entropy = self._calculate_entropy(hsv[:,:,0])
            s_entropy = self._calculate_entropy(hsv[:,:,1])
            v_entropy = self._calculate_entropy(hsv[:,:,2])
            
            # Adjust color diversity threshold
            color_threshold = 13.5 if self.sensitivity_mode == "strict" else (13.0 if self.sensitivity_mode == "balanced" else 12.0)
            indicators['color_diversity'] = min(1.0, (h_entropy + s_entropy + v_entropy) / color_threshold)
            
            # 5. Lighting Analysis
            lighting_score = self._analyze_lighting_patterns_enhanced(gray)
            indicators['natural_lighting'] = lighting_score
            
            # 6. Edge Analysis
            edges_canny = cv2.Canny(gray, 30, 100)
            edges_sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3)
            
            edge_density_canny = np.sum(edges_canny > 0) / (gray.shape[0] * gray.shape[1])
            edge_density_sobel = np.sum(edges_sobel > 50) / (gray.shape[0] * gray.shape[1])
            
            edge_score = (edge_density_canny * 0.6 + edge_density_sobel * 0.4) * 12
            indicators['edge_naturalness'] = min(1.0, edge_score)
            
            # 7. Moiré Detection - adjustable sensitivity
            moire_score = self._detect_moire_enhanced(gray)
            moire_multiplier = 0.9 if self.sensitivity_mode == "strict" else (0.8 if self.sensitivity_mode == "balanced" else 0.7)
            indicators['moire_absence'] = 1.0 - (moire_score * moire_multiplier)
            
            # 8. Reflection Detection
            reflection_score = self._detect_screen_reflections_enhanced(face_img)
            indicators['reflection_absence'] = 1.0 - reflection_score
            
            # 9. Compression Artifacts
            compression_score = self._detect_compression_artifacts(gray)
            compression_multiplier = 0.8 if self.sensitivity_mode == "strict" else (0.7 if self.sensitivity_mode == "balanced" else 0.6)
            indicators['compression_naturalness'] = 1.0 - (compression_score * compression_multiplier)
            
            # 10. Pixel Uniformity
            uniformity_score = self._analyze_pixel_uniformity(gray)
            indicators['pixel_naturalness'] = uniformity_score
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error in advanced spoof detection: {e}")
            return {'error': 1.0}
    
    def _evaluate_skin_color_enhanced(self, face_img: np.ndarray, hsv: np.ndarray, lab: np.ndarray) -> float:
        """Enhanced skin color evaluation with sensitivity adjustment"""
        try:
            h, s, v = cv2.split(hsv)
            l, a, b = cv2.split(lab)
            
            # Adjust HSV ranges based on sensitivity
            if self.sensitivity_mode == "lenient":
                # More relaxed skin detection
                skin_mask1 = cv2.inRange(hsv, (0, 20, 40), (30, 255, 255))
                skin_mask2 = cv2.inRange(hsv, (150, 20, 40), (180, 255, 255))
                lab_skin_mask = cv2.inRange(lab, (40, 110, 110), (220, 160, 160))
            else:
                # Standard skin detection
                skin_mask1 = cv2.inRange(hsv, (0, 25, 50), (25, 255, 255))
                skin_mask2 = cv2.inRange(hsv, (155, 25, 50), (180, 255, 255))
                lab_skin_mask = cv2.inRange(lab, (45, 115, 115), (210, 155, 155))
            
            hsv_skin_mask = skin_mask1 | skin_mask2
            hsv_skin_ratio = np.sum(hsv_skin_mask > 0) / (face_img.shape[0] * face_img.shape[1])
            lab_skin_ratio = np.sum(lab_skin_mask > 0) / (face_img.shape[0] * face_img.shape[1])
            
            # RGB ratios
            b_ch, g_ch, r_ch = cv2.split(face_img)
            r_mean, g_mean, b_mean = np.mean(r_ch), np.mean(g_ch), np.mean(b_ch)
            
            # More lenient RGB scoring for lenient mode
            if self.sensitivity_mode == "lenient":
                rgb_order_score = 0.6  # Higher default
                if r_mean > g_mean > b_mean:
                    rgb_order_score = 1.0
                elif r_mean > g_mean or g_mean > b_mean:
                    rgb_order_score = 0.8
            else:
                rgb_order_score = 0.4  # Standard default
                if r_mean > g_mean > b_mean:
                    rgb_order_score = 1.0
                elif r_mean > g_mean or g_mean > b_mean:
                    rgb_order_score = 0.7
            
            # Combine models
            combined_score = (hsv_skin_ratio * 0.35 + lab_skin_ratio * 0.35 + rgb_order_score * 0.3)
            
            # Adjust minimum score based on sensitivity
            min_score = 0.1 if self.sensitivity_mode == "strict" else (0.15 if self.sensitivity_mode == "balanced" else 0.25)
            return max(min_score, min(1.0, combined_score))
            
        except Exception as e:
            logger.error(f"Error in enhanced skin color evaluation: {e}")
            return 0.5
    
    def _analyze_lighting_patterns_enhanced(self, gray: np.ndarray) -> float:
        """Enhanced lighting pattern analysis"""
        try:
            lighting_scores = []
            
            for blur_size in [15, 31, 63]:
                if blur_size < min(gray.shape):
                    lighting_map = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
                    
                    grad_x = cv2.Sobel(lighting_map, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(lighting_map, cv2.CV_64F, 0, 1, ksize=3)
                    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                    
                    grad_std = np.std(grad_magnitude)
                    grad_mean = np.mean(grad_magnitude)
                    
                    if grad_mean > 0:
                        multiplier = 1.5 if self.sensitivity_mode == "strict" else (1.3 if self.sensitivity_mode == "balanced" else 1.0)
                        consistency = 1.0 - min(1.0, grad_std / (grad_mean * multiplier))
                        min_consistency = 0.15 if self.sensitivity_mode == "strict" else (0.2 if self.sensitivity_mode == "balanced" else 0.3)
                        lighting_scores.append(max(min_consistency, consistency))
            
            default_score = 0.25 if self.sensitivity_mode == "strict" else (0.3 if self.sensitivity_mode == "balanced" else 0.4)
            return max(default_score, np.mean(lighting_scores)) if lighting_scores else default_score
            
        except Exception as e:
            logger.error(f"Error in enhanced lighting analysis: {e}")
            return 0.4
    
    def _detect_moire_enhanced(self, gray: np.ndarray) -> float:
        """Enhanced moiré pattern detection"""
        try:
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(gray, -1, kernel)
            
            f_transform = np.fft.fft2(filtered)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            h, w = magnitude.shape
            center_y, center_x = h//2, w//2
            
            max_periodic_energy = 0
            
            for radius in range(5, min(h, w)//3, 3):
                angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
                energies = []
                
                for angle in angles:
                    y = int(center_y + radius * np.sin(angle))
                    x = int(center_x + radius * np.cos(angle))
                    if 0 <= y < h and 0 <= x < w:
                        energies.append(magnitude[y, x])
                
                if energies:
                    energy_variance = np.var(energies)
                    max_periodic_energy = max(max_periodic_energy, energy_variance)
            
            # Adjust threshold based on sensitivity
            moire_threshold = 2300.0 if self.sensitivity_mode == "strict" else (2500.0 if self.sensitivity_mode == "balanced" else 2800.0)
            moire_score = min(1.0, max_periodic_energy / moire_threshold)
            return moire_score
            
        except Exception as e:
            logger.error(f"Error in enhanced moiré detection: {e}")
            return 0.0
    
    def _detect_screen_reflections_enhanced(self, face_img: np.ndarray) -> float:
        """Enhanced screen reflection detection"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            reflection_scores = []
            
            # Adjust percentiles based on sensitivity
            percentiles = [90, 94, 97] if self.sensitivity_mode == "strict" else ([91, 95, 98] if self.sensitivity_mode == "balanced" else [92, 96, 99])
            
            for percentile in percentiles:
                threshold = np.percentile(gray, percentile)
                bright_mask = gray > threshold
                
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    bright_mask.astype(np.uint8), connectivity=8
                )
                
                total_area = gray.shape[0] * gray.shape[1]
                large_bright_area = 0
                
                area_threshold = 0.01 if self.sensitivity_mode == "strict" else (0.015 if self.sensitivity_mode == "balanced" else 0.02)
                
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > total_area * area_threshold:
                        large_bright_area += area
                
                reflection_score = large_bright_area / total_area
                reflection_scores.append(reflection_score)
            
            multiplier = 10.0 if self.sensitivity_mode == "strict" else (9.0 if self.sensitivity_mode == "balanced" else 8.0)
            return min(1.0, np.mean(reflection_scores) * multiplier)
            
        except Exception as e:
            logger.error(f"Error in enhanced reflection detection: {e}")
            return 0.0
    
    def _detect_compression_artifacts(self, gray: np.ndarray) -> float:
        """Detect compression artifacts"""
        try:
            block_size = 8
            h, w = gray.shape
            
            artifact_scores = []
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size].astype(np.float32)
                    
                    dct = cv2.dct(block)
                    high_freq = dct[4:, 4:]
                    high_freq_energy = np.sum(high_freq**2)
                    artifact_scores.append(high_freq_energy)
            
            if artifact_scores:
                avg_energy = np.mean(artifact_scores)
                # Adjust threshold based on sensitivity
                energy_threshold = 1100.0 if self.sensitivity_mode == "strict" else (1200.0 if self.sensitivity_mode == "balanced" else 1400.0)
                compression_score = 1.0 - min(1.0, avg_energy / energy_threshold)
                return compression_score
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error detecting compression artifacts: {e}")
            return 0.0
    
    def _analyze_pixel_uniformity(self, gray: np.ndarray) -> float:
        """Analyze pixel uniformity"""
        try:
            kernel = np.ones((3, 3), np.float32) / 9
            smooth = cv2.filter2D(gray, -1, kernel)
            variation = cv2.absdiff(gray, smooth)
            
            variation_std = np.std(variation)
            
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            grad_entropy = self._calculate_entropy(grad_magnitude.astype(np.uint8))
            
            # Adjust thresholds based on sensitivity
            var_divisor = 8.5 if self.sensitivity_mode == "strict" else (9.0 if self.sensitivity_mode == "balanced" else 9.5)
            entropy_divisor = 6.5 if self.sensitivity_mode == "strict" else (7.0 if self.sensitivity_mode == "balanced" else 7.5)
            
            uniformity_score = min(1.0, (variation_std / var_divisor + grad_entropy / entropy_divisor) / 2.0)
            
            min_score = 0.15 if self.sensitivity_mode == "strict" else (0.2 if self.sensitivity_mode == "balanced" else 0.3)
            return max(min_score, uniformity_score)
            
        except Exception as e:
            logger.error(f"Error analyzing pixel uniformity: {e}")
            return 0.4
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy"""
        try:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist[hist > 0]
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist))
            return entropy
        except Exception:
            return 0.0
    
    def _optimized_ensemble_scoring(self, indicators: Dict[str, float]) -> float:
        """Optimized ensemble scoring with configurable sensitivity"""
        try:
            # Check for strong screen indicators
            moire_score = indicators.get('moire_absence', 1.0)
            rgb_score = indicators.get('rgb_independence', 1.0)
            screen_pattern_score = indicators.get('screen_pattern_absence', 1.0)
            
            # Adjust thresholds based on sensitivity mode
            if self.sensitivity_mode == "strict":
                moire_threshold, rgb_threshold, pattern_threshold = 0.5, 0.5, 0.7
            elif self.sensitivity_mode == "balanced":
                moire_threshold, rgb_threshold, pattern_threshold = 0.4, 0.4, 0.6
            else:  # lenient
                moire_threshold, rgb_threshold, pattern_threshold = 0.3, 0.3, 0.5
            
            # Count strong negative indicators
            strong_screen_indicators = 0
            if moire_score < moire_threshold:
                strong_screen_indicators += 1
            if rgb_score < rgb_threshold:
                strong_screen_indicators += 1
            if screen_pattern_score < pattern_threshold:
                strong_screen_indicators += 1
            
            # Balanced weights
            weights = {
                'texture_richness': 0.13,
                'screen_pattern_absence': 0.14,
                'rgb_independence': 0.12,
                'skin_realism': 0.11,
                'color_diversity': 0.09,
                'natural_lighting': 0.11,
                'edge_naturalness': 0.09,
                'moire_absence': 0.12,
                'reflection_absence': 0.08,
                'compression_naturalness': 0.06,
                'pixel_naturalness': 0.10
            }

            total_score = 0.0
            total_weight = 0.0

            for indicator, value in indicators.items():
                if indicator in weights and not np.isnan(value):
                    total_score += value * weights[indicator]
                    total_weight += weights[indicator]

            base_score = total_score / total_weight if total_weight > 0 else 0.5
            
            # Apply penalties based on sensitivity and screen indicators
            final_score = base_score
            
            # Make penalty less harsh (tune as needed)
            if strong_screen_indicators >= 2:
                penalty = 0.85 if self.sensitivity_mode == "strict" else (0.88 if self.sensitivity_mode == "balanced" else 0.9)
                final_score *= penalty
                logger.info(f"Multiple screen indicators detected: {strong_screen_indicators}")
            elif strong_screen_indicators == 1:
                penalty = 0.92 if self.sensitivity_mode == "strict" else (0.95 if self.sensitivity_mode == "balanced" else 0.97)
                final_score *= penalty
                logger.info(f"One screen indicator detected: {strong_screen_indicators}")
            
            # Boost score if face_quality or skin_realism is high, to help real images
            skin_realism = indicators.get('skin_realism', 0)
            if skin_realism > 0.8:
                final_score = min(1.0, final_score + 0.15)
                
            # Bonus: if texture_richness and edge_naturalness are both high, add a bit more
            if indicators.get('texture_richness', 0) > 0.6 and indicators.get('edge_naturalness', 0) > 0.8:
                final_score = min(1.0, final_score + 0.05)

            # Apply boost for natural characteristics based on sensitivity
            skin_realism = indicators.get('skin_realism', 0)
            natural_lighting = indicators.get('natural_lighting', 0)
            
            if skin_realism > 0.7 and natural_lighting > 0.5 and strong_screen_indicators == 0:
                boost = getattr(self, '_natural_boost_multiplier', 1.05)
                final_score *= boost
            
            logger.info("--- Optimized Spoof Indicators ---")
            for k, v in indicators.items():
                logger.info(f"{k:25}: {v:.3f}")
            logger.info(f"Sensitivity: {self.sensitivity_mode}, Base: {base_score:.3f}, Screen indicators: {strong_screen_indicators}, Final: {final_score:.3f}")

            return max(0.0, min(1.0, final_score))

        except Exception as e:
            logger.error(f"Error in optimized ensemble scoring: {e}")
            return 0.5

    def check_liveness(self, image: np.ndarray, method: LivenessMethod = LivenessMethod.ENSEMBLE) -> LivenessResult:
        """Main liveness detection function with optimized scoring"""
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
                # Optimized multi-feature analysis
                spoof_indicators = self._advanced_spoof_detection(face_resized)
                confidence = self._optimized_ensemble_scoring(spoof_indicators)
                method_used = f"optimized_ensemble_v5_{self.sensitivity_mode}"
                
            else:
                # Fallback to optimized analysis
                spoof_indicators = self._advanced_spoof_detection(face_resized)
                confidence = self._optimized_ensemble_scoring(spoof_indicators)
                method_used = f"optimized_fallback_{self.sensitivity_mode}"
            
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
            img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_norm = img_norm.transpose(2, 0, 1)[None]
            
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
        """Generate user recommendations based on detection results and sensitivity"""
        recommendations = []
        
        # Adjust confidence interpretation based on sensitivity
        if self.sensitivity_mode == "strict":
            if confidence < 0.4:
                recommendations.append("Very low liveness confidence - likely spoofing attempt")
            elif confidence < 0.6:
                recommendations.append("Low liveness confidence - possible spoofing")
            elif confidence < 0.7:
                recommendations.append("Moderate confidence - borderline detection")
            elif confidence < 0.8:
                recommendations.append("Good confidence - likely live")
            else:
                recommendations.append("High confidence live detection")
        elif self.sensitivity_mode == "balanced":
            if confidence < 0.3:
                recommendations.append("Very low liveness confidence - likely spoofing attempt")
            elif confidence < 0.5:
                recommendations.append("Low liveness confidence - possible spoofing")
            elif confidence < 0.58:
                recommendations.append("Moderate confidence - borderline detection")
            elif confidence < 0.75:
                recommendations.append("Good confidence - likely live")
            else:
                recommendations.append("High confidence live detection")
        else:  # lenient
            if confidence < 0.25:
                recommendations.append("Very low liveness confidence - likely spoofing attempt")
            elif confidence < 0.4:
                recommendations.append("Low liveness confidence - possible spoofing")
            elif confidence < 0.45:
                recommendations.append("Moderate confidence - borderline detection")
            elif confidence < 0.7:
                recommendations.append("Good confidence - likely live")
            else:
                recommendations.append("High confidence live detection")
        
        if face_quality < 0.3:
            recommendations.append("Improve image quality - move closer or improve lighting")
        
        # Specific recommendations based on indicators and sensitivity
        texture_threshold = 0.3 if self.sensitivity_mode == "strict" else (0.25 if self.sensitivity_mode == "balanced" else 0.2)
        if indicators.get('texture_richness', 1.0) < texture_threshold:
            recommendations.append("Image appears very smooth - check for filters or processing")
        
        skin_threshold = 0.5 if self.sensitivity_mode == "strict" else (0.4 if self.sensitivity_mode == "balanced" else 0.3)
        if indicators.get('skin_realism', 1.0) < skin_threshold:
            recommendations.append("Skin color appears unnatural - check lighting conditions")
        
        rgb_threshold = 0.4 if self.sensitivity_mode == "strict" else (0.3 if self.sensitivity_mode == "balanced" else 0.25)
        if indicators.get('rgb_independence', 1.0) < rgb_threshold:
            recommendations.append("High RGB correlation detected - avoid screen photos")
        
        if indicators.get('reflection_absence', 1.0) < 0.4:
            recommendations.append("Reflections detected - avoid glossy surfaces")
        
        moire_threshold = 0.5 if self.sensitivity_mode == "strict" else (0.4 if self.sensitivity_mode == "balanced" else 0.3)
        if indicators.get('moire_absence', 1.0) < moire_threshold:
            recommendations.append("Moiré patterns detected - avoid screen interference")
        
        pattern_threshold = 0.7 if self.sensitivity_mode == "strict" else (0.6 if self.sensitivity_mode == "balanced" else 0.5)
        if indicators.get('screen_pattern_absence', 1.0) < pattern_threshold:
            recommendations.append("Screen patterns detected - use direct camera capture")
        
        return recommendations

# Global instance
_detector_instance = None

def get_detector_instance(model_path: Optional[str] = None) -> EnhancedLivenessDetector:
    """Get singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EnhancedLivenessDetector(model_path)
    return _detector_instance

def set_detection_sensitivity(mode: str):
    """Set global detection sensitivity mode"""
    detector = get_detector_instance()
    detector.set_sensitivity(mode)

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