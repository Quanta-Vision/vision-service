import cv2
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class LivenessDetector:
    """
    Enhanced liveness detection system that combines multiple anti-spoofing techniques
    to determine if an image is captured from a real camera or through screen/photo spoofing
    """
    
    def __init__(self):
        self.version = "1.0.0"
        self.default_threshold = 0.5
        self.sensitivity = "medium"
        self.detection_methods = [
            "texture_analysis",
            "frequency_analysis", 
            "color_analysis",
            "reflection_analysis",
            "edge_analysis",
            "noise_analysis",
            "digital_artifacts_analysis",
            "compression_analysis",
            "lighting_analysis"
        ]
        
        # Initialize face detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            logger.warning("Face cascade not loaded, using full image analysis")
            self.face_cascade = None
    
    def detect_liveness(self, image: np.ndarray, threshold: float = None) -> Dict[str, Any]:
        """
        Main liveness detection function
        
        Args:
            image: Input image as numpy array
            threshold: Detection threshold (0.0 to 1.0)
            
        Returns:
            Dictionary with detection results
        """
        if threshold is None:
            threshold = self.default_threshold
            
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Detect face region (optional)
            face_region = self._detect_face_region(image)
            if face_region is not None:
                analysis_region = face_region
            else:
                analysis_region = gray
            
            # Perform multiple analyses
            results = {}
            
            # 1. Texture Analysis (LBP - Local Binary Patterns)
            results['texture'] = self._analyze_texture(analysis_region)
            
            # 2. Frequency Analysis (FFT)
            results['frequency'] = self._analyze_frequency(analysis_region)
            
            # 3. Color Analysis
            if len(image.shape) == 3:
                results['color'] = self._analyze_color(image, face_region)
            else:
                results['color'] = {'score': 0.5, 'confidence': 0.0}
            
            # 4. Reflection Analysis
            results['reflection'] = self._analyze_reflections(analysis_region)
            
            # 5. Edge Analysis
            results['edge'] = self._analyze_edges(analysis_region)
            
            # 6. Noise Analysis
            results['noise'] = self._analyze_noise(analysis_region)
            
            # 7. NEW: Digital Artifact Analysis
            results['digital_artifacts'] = self._analyze_digital_artifacts(image)
            
            # 8. NEW: Compression Analysis
            results['compression'] = self._analyze_compression_patterns(analysis_region)
            
            # 9. NEW: Lighting Analysis
            results['lighting'] = self._analyze_lighting_patterns(analysis_region)
            
            # Combine all scores with updated weights
            final_score, confidence = self._combine_scores(results)
            
            # Apply stricter decision logic
            is_live = self._make_final_decision(results, final_score, threshold)
            
            # Determine spoof type if detected
            spoof_type = None
            if not is_live:
                spoof_type = self._determine_spoof_type(results)
            
            return {
                'is_live': is_live,
                'confidence': confidence,
                'liveness_score': final_score,
                'decision_margin': abs(final_score - threshold),  # How far from threshold
                'spoof_type': spoof_type,
                'details': {
                    'individual_scores': results,
                    'final_score': final_score,
                    'threshold_used': threshold,
                    'decision_explanation': f"Final score ({final_score:.3f}) {'≥' if final_score >= threshold else '<'} threshold ({threshold:.3f}), Decision: {'LIVE' if is_live else 'SPOOF'}",
                    'face_detected': face_region is not None,
                    'image_quality': self._assess_image_quality(gray),
                    'spoof_indicators': self._get_spoof_indicators(results)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in liveness detection: {str(e)}")
            return {
                'is_live': False,
                'confidence': 0.0,
                'liveness_score': 0.0,
                'decision_margin': 0.0,
                'spoof_type': 'analysis_error',
                'details': {'error': str(e)}
            }
    
    def _detect_face_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face region in image"""
        if self.face_cascade is None:
            return None
            
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                return gray[y:y+h, x:x+w]
                
        except Exception as e:
            logger.warning(f"Face detection failed: {str(e)}")
            
        return None
    
    def _analyze_texture(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze texture using Local Binary Patterns"""
        try:
            # Calculate LBP
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(image, n_points, radius, method='uniform')
            
            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            # Calculate texture measures
            uniformity = np.sum(hist ** 2)
            entropy = shannon_entropy(hist)
            
            # More discriminative texture scoring
            # Real images: entropy typically 3.5-5.0, uniformity 0.05-0.15
            # Screen images: entropy typically 3.0-4.5, uniformity 0.1-0.4
            
            # Combine entropy and uniformity for better discrimination
            if entropy > 4.5 and uniformity < 0.15:
                texture_score = 0.9  # High entropy, low uniformity = very textured (real)
            elif entropy > 4.0 and uniformity < 0.2:
                texture_score = 0.7  # Good texture
            elif entropy > 3.5 and uniformity < 0.25:
                texture_score = 0.5  # Moderate texture
            elif entropy > 3.0 and uniformity < 0.35:
                texture_score = 0.3  # Low texture (possibly screen)
            else:
                texture_score = 0.1  # Very low texture (likely screen)
            
            return {
                'score': float(texture_score),
                'uniformity': float(uniformity),
                'entropy': float(entropy),
                'confidence': float(min(abs(texture_score - 0.5) * 2, 1.0))
            }
            
        except Exception as e:
            logger.warning(f"Texture analysis failed: {str(e)}")
            return {'score': 0.5, 'confidence': 0.0}
    
    def _analyze_frequency(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze frequency domain characteristics"""
        try:
            # Apply FFT
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Calculate frequency distribution
            h, w = image.shape
            center_h, center_w = h // 2, w // 2
            
            # Create frequency bands
            y, x = np.ogrid[:h, :w]
            mask_low = (x - center_w)**2 + (y - center_h)**2 < (min(h, w) // 8)**2
            mask_high = (x - center_w)**2 + (y - center_h)**2 > (min(h, w) // 4)**2
            
            low_freq_energy = np.sum(magnitude_spectrum[mask_low])
            high_freq_energy = np.sum(magnitude_spectrum[mask_high])
            
            # Calculate ratio
            freq_ratio = high_freq_energy / (low_freq_energy + 1e-7)
            
            # More conservative scoring for frequency analysis
            # Screen images typically have freq_ratio between 5-15
            # Real images typically have freq_ratio > 8
            if freq_ratio < 6:
                freq_score = 0.2  # Very likely screen
            elif freq_ratio < 8:
                freq_score = 0.4  # Possibly screen
            elif freq_ratio < 12:
                freq_score = 0.6  # Borderline
            elif freq_ratio < 16:
                freq_score = 0.8  # Likely real
            else:
                freq_score = 0.9  # Very likely real
            
            return {
                'score': float(freq_score),
                'freq_ratio': float(freq_ratio),
                'low_freq_energy': float(low_freq_energy),
                'high_freq_energy': float(high_freq_energy),
                'confidence': float(min(abs(freq_score - 0.5) * 2, 1.0))
            }
            
        except Exception as e:
            logger.warning(f"Frequency analysis failed: {str(e)}")
            return {'score': 0.5, 'confidence': 0.0}
    
    def _analyze_color(self, image: np.ndarray, face_region: Optional[np.ndarray]) -> Dict[str, float]:
        """Analyze color characteristics"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Calculate color statistics
            color_variance = np.var(image, axis=(0, 1))
            color_mean = np.mean(image, axis=(0, 1))
            
            # Calculate color diversity
            unique_colors = len(np.unique(image.reshape(-1, image.shape[-1]), axis=0))
            total_pixels = image.shape[0] * image.shape[1]
            color_diversity = unique_colors / total_pixels
            
            # Calculate saturation statistics
            saturation = hsv[:, :, 1]
            sat_mean = np.mean(saturation)
            sat_std = np.std(saturation)
            
            # Live images typically have more natural color variation
            # Screen images may have color shifts or reduced gamut
            color_score = min(color_diversity * 10, 1.0)
            
            return {
                'score': float(color_score),
                'color_diversity': float(color_diversity),
                'color_variance': [float(x) for x in color_variance],
                'saturation_mean': float(sat_mean),
                'saturation_std': float(sat_std),
                'confidence': float(min(abs(color_score - 0.5) * 2, 1.0))
            }
            
        except Exception as e:
            logger.warning(f"Color analysis failed: {str(e)}")
            return {'score': 0.5, 'confidence': 0.0}
    
    def _analyze_reflections(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze reflection patterns"""
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Find bright spots (potential reflections)
            bright_threshold = np.percentile(blurred, 95)
            bright_spots = blurred > bright_threshold
            
            # Calculate reflection characteristics
            reflection_ratio = np.sum(bright_spots) / bright_spots.size
            
            # Group bright spots to find reflection patterns
            labeled_spots, num_spots = ndimage.label(bright_spots)
            
            # Calculate spot sizes
            spot_sizes = []
            for i in range(1, num_spots + 1):
                spot_size = np.sum(labeled_spots == i)
                spot_sizes.append(spot_size)
            
            # Screen reflections often have different patterns than natural reflections
            if len(spot_sizes) > 0:
                avg_spot_size = np.mean(spot_sizes)
                reflection_score = 1.0 - min(reflection_ratio * 5, 1.0)  # Fewer artificial reflections = more likely live
            else:
                reflection_score = 0.7  # No strong reflections detected
            
            return {
                'score': float(reflection_score),
                'reflection_ratio': float(reflection_ratio),
                'num_spots': int(num_spots),
                'avg_spot_size': float(avg_spot_size if spot_sizes else 0),
                'confidence': float(min(abs(reflection_score - 0.5) * 2, 1.0))
            }
            
        except Exception as e:
            logger.warning(f"Reflection analysis failed: {str(e)}")
            return {'score': 0.5, 'confidence': 0.0}
    
    def _analyze_edges(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze edge characteristics"""
        try:
            # Apply Canny edge detection
            edges = cv2.Canny(image, 50, 150)
            
            # Calculate edge statistics
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate edge strength
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)
            avg_edge_strength = np.mean(edge_strength)
            
            # Live images typically have sharper, more varied edges
            # Screen images may have softer edges due to display characteristics
            edge_score = min(avg_edge_strength / 100, 1.0)
            
            return {
                'score': float(edge_score),
                'edge_density': float(edge_density),
                'avg_edge_strength': float(avg_edge_strength),
                'confidence': float(min(abs(edge_score - 0.5) * 2, 1.0))
            }
            
        except Exception as e:
            logger.warning(f"Edge analysis failed: {str(e)}")
            return {'score': 0.5, 'confidence': 0.0}
    
    def _analyze_noise(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze noise characteristics"""
        try:
            # Apply median filter to estimate noise
            filtered = cv2.medianBlur(image, 5)
            noise = image.astype(float) - filtered.astype(float)
            
            # Calculate noise statistics
            noise_std = np.std(noise)
            noise_mean = np.mean(np.abs(noise))
            
            # Calculate noise distribution
            noise_hist, _ = np.histogram(noise.ravel(), bins=50, range=(-50, 50))
            noise_hist = noise_hist.astype(float)
            noise_hist /= (noise_hist.sum() + 1e-7)
            
            # Live images typically have more natural noise
            # Screen images may have artificial noise patterns or reduced noise
            noise_score = min(noise_std / 10, 1.0)
            
            return {
                'score': float(noise_score),
                'noise_std': float(noise_std),
                'noise_mean': float(noise_mean),
                'confidence': float(min(abs(noise_score - 0.5) * 2, 1.0))
            }
            
        except Exception as e:
            logger.warning(f"Noise analysis failed: {str(e)}")
            return {'score': 0.5, 'confidence': 0.0}
    
    def _combine_scores(self, results: Dict[str, Dict[str, float]]) -> tuple:
        """Combine individual analysis scores into final score"""
        try:
            # Updated weights with new detection methods
            # More emphasis on digital artifact detection
            weights = {
                'texture': 0.15,           # Reduced weight
                'frequency': 0.15,         # Reduced weight  
                'color': 0.12,
                'reflection': 0.10,
                'edge': 0.10,
                'noise': 0.08,
                'digital_artifacts': 0.15,  # NEW: High weight for digital artifacts
                'compression': 0.10,        # NEW: Compression analysis
                'lighting': 0.05           # NEW: Lighting analysis
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            weighted_confidence = 0.0
            method_count = 0
            
            for method, result in results.items():
                if method in weights and 'score' in result:
                    weight = weights[method]
                    score = result['score']
                    method_confidence = result.get('confidence', 0.5)
                    
                    weighted_score += weight * score
                    weighted_confidence += weight * method_confidence
                    total_weight += weight
                    method_count += 1
            
            # Normalize
            if total_weight > 0:
                final_score = weighted_score / total_weight
                base_confidence = weighted_confidence / total_weight
                decision_confidence = abs(final_score - 0.5) * 2
                overall_confidence = (base_confidence + decision_confidence) / 2
                overall_confidence = min(overall_confidence, 1.0)
            else:
                final_score = 0.5
                overall_confidence = 0.0
            
            return final_score, overall_confidence
            
        except Exception as e:
            logger.warning(f"Score combination failed: {str(e)}")
            return 0.5, 0.0
    
    def _determine_spoof_type(self, results: Dict[str, Dict[str, float]]) -> str:
        """Determine the type of spoofing detected"""
        try:
            spoof_scores = {}
            
            # Analyze patterns to determine spoof type
            for method, result in results.items():
                score = result.get('score', 0.5)
                spoof_scores[method] = score
            
            # Screen display indicators
            screen_indicators = 0
            if spoof_scores.get('frequency', 1.0) < 0.4:  # Low high-freq content
                screen_indicators += 2
            if spoof_scores.get('lighting', 1.0) < 0.3:  # Uniform backlighting
                screen_indicators += 2
            if spoof_scores.get('digital_artifacts', 1.0) < 0.4:  # Digital processing
                screen_indicators += 1
            if results.get('lighting', {}).get('backlight_intensity', 0) > 15:
                screen_indicators += 2
            
            # Printed photo indicators
            print_indicators = 0
            if spoof_scores.get('texture', 1.0) < 0.3:  # Paper texture interference
                print_indicators += 2
            if spoof_scores.get('color', 1.0) < 0.3:  # Reduced color gamut
                print_indicators += 2
            if spoof_scores.get('reflection', 1.0) < 0.4:  # Paper reflections
                print_indicators += 1
            
            # Digital manipulation indicators
            digital_indicators = 0
            if spoof_scores.get('digital_artifacts', 1.0) < 0.2:  # Heavy processing
                digital_indicators += 3
            if spoof_scores.get('compression', 1.0) < 0.3:  # Compression artifacts
                digital_indicators += 2
            if spoof_scores.get('edge', 1.0) < 0.3:  # Artificial edges
                digital_indicators += 1
            
            # Determine most likely spoof type
            max_indicators = max(screen_indicators, print_indicators, digital_indicators)
            
            if max_indicators >= 4:
                if screen_indicators == max_indicators:
                    return 'screen_display'
                elif print_indicators == max_indicators:
                    return 'printed_photo'
                elif digital_indicators == max_indicators:
                    return 'digital_manipulation'
            
            # Fallback logic for edge cases
            if spoof_scores.get('frequency', 1.0) < 0.3:
                return 'screen_display'
            elif spoof_scores.get('color', 1.0) < 0.3 and spoof_scores.get('texture', 1.0) < 0.4:
                return 'printed_photo'
            elif spoof_scores.get('digital_artifacts', 1.0) < 0.3:
                return 'digital_manipulation'
            else:
                return 'unknown_spoof'
                
        except Exception as e:
            logger.warning(f"Spoof type determination failed: {str(e)}")
            return 'unknown_spoof'
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess overall image quality"""
        try:
            # Calculate sharpness (variance of Laplacian)
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            sharpness = np.var(laplacian)
            
            # Calculate brightness
            brightness = np.mean(image)
            
            # Calculate contrast
            contrast = np.std(image)
            
            return {
                'sharpness': float(sharpness),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'overall_quality': float(min((sharpness / 1000 + contrast / 100) / 2, 1.0))
            }
        
        except Exception as e:
            logger.warning(f"Image quality assessment failed: {str(e)}")
            return {'overall_quality': 0.5}
            
    def _analyze_digital_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze for digital processing artifacts"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 1. Check for JPEG compression artifacts
            # Look for 8x8 block patterns common in JPEG
            h, w = gray.shape
            block_differences = []
            
            for y in range(0, h-8, 8):
                for x in range(0, w-8, 8):
                    block = gray[y:y+8, x:x+8]
                    # Check uniformity within block vs across block boundaries
                    if y+16 < h and x+16 < w:
                        next_block = gray[y+8:y+16, x+8:x+16]
                        boundary_diff = np.mean(np.abs(block.astype(float) - next_block.astype(float)))
                        block_differences.append(boundary_diff)
            
            # 2. Check for digital sharpening artifacts
            # Oversharpened images have characteristic edge halos
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            sharpening_artifacts = np.mean(np.abs(sharpened.astype(float) - gray.astype(float)))
            
            # 3. Check for unnatural uniformity (digital smoothing)
            # Calculate local variance across the image
            kernel_size = 5
            local_variances = []
            for y in range(0, h-kernel_size, kernel_size):
                for x in range(0, w-kernel_size, kernel_size):
                    patch = gray[y:y+kernel_size, x:x+kernel_size]
                    local_variances.append(np.var(patch))
            
            variance_uniformity = np.std(local_variances) if local_variances else 0
            
            # Combine indicators
            if block_differences:
                avg_block_diff = np.mean(block_differences)
                # High block differences suggest compression artifacts
                compression_score = min(avg_block_diff / 20, 1.0)
            else:
                compression_score = 0.5
            
            # Lower sharpening artifacts = more natural
            sharpening_score = max(0, 1.0 - min(sharpening_artifacts / 50, 1.0))
            
            # Higher variance uniformity = less digital processing
            uniformity_score = min(variance_uniformity / 100, 1.0)
            
            # Combine scores (lower = more digital artifacts = more likely spoof)
            digital_score = (sharpening_score + uniformity_score) / 2
            digital_score = max(0, min(digital_score, 1.0))
            
            return {
                'score': float(digital_score),
                'compression_artifacts': float(compression_score),
                'sharpening_artifacts': float(sharpening_artifacts),
                'variance_uniformity': float(variance_uniformity),
                'confidence': float(abs(digital_score - 0.5) * 2)
            }
            
        except Exception as e:
            logger.warning(f"Digital artifact analysis failed: {str(e)}")
            return {'score': 0.5, 'confidence': 0.0}
    
    def _analyze_compression_patterns(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze compression patterns that indicate digital processing"""
        try:
            # Apply DCT analysis to detect compression patterns
            # Real camera images have different compression characteristics than processed images
            
            # Convert to float
            img_float = image.astype(np.float32)
            
            # Apply 2D DCT on 8x8 blocks
            h, w = img_float.shape
            dct_coeffs = []
            
            for y in range(0, h-8, 8):
                for x in range(0, w-8, 8):
                    block = img_float[y:y+8, x:x+8]
                    dct_block = cv2.dct(block)
                    # Focus on high-frequency coefficients
                    high_freq = dct_block[4:, 4:]
                    dct_coeffs.extend(high_freq.flatten())
            
            if dct_coeffs:
                # Calculate statistics of DCT coefficients
                dct_mean = np.mean(np.abs(dct_coeffs))
                dct_std = np.std(dct_coeffs)
                
                # Real images typically have more varied DCT coefficients
                # Heavily compressed images have more uniform/quantized coefficients
                compression_score = min(dct_std / 10, 1.0)
            else:
                compression_score = 0.5
            
            return {
                'score': float(compression_score),
                'dct_mean': float(dct_mean if dct_coeffs else 0),
                'dct_std': float(dct_std if dct_coeffs else 0),
                'confidence': float(abs(compression_score - 0.5) * 2)
            }
            
        except Exception as e:
            logger.warning(f"Compression pattern analysis failed: {str(e)}")
            return {'score': 0.5, 'confidence': 0.0}
    
    def _analyze_lighting_patterns(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze lighting patterns for naturalness"""
        try:
            # Real photos have natural lighting variations
            # Screen photos often have artificial, uniform lighting
            
            # Calculate gradient magnitude to detect lighting transitions
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Analyze lighting uniformity
            # Divide image into regions and check lighting consistency
            h, w = image.shape
            region_size = min(h, w) // 4
            region_means = []
            
            for y in range(0, h-region_size, region_size):
                for x in range(0, w-region_size, region_size):
                    region = image[y:y+region_size, x:x+region_size]
                    region_means.append(np.mean(region))
            
            if region_means:
                lighting_variance = np.var(region_means)
                lighting_std = np.std(region_means)
                
                # Natural images have more lighting variation
                # Screen images often have uniform backlighting
                lighting_score = min(lighting_variance / 500, 1.0)
            else:
                lighting_score = 0.5
            
            # Check for screen backlight patterns
            # Screens often have subtle grid patterns from backlighting
            blur_kernel = 15
            blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
            backlight_pattern = np.abs(image.astype(float) - blurred.astype(float))
            backlight_intensity = np.mean(backlight_pattern)
            
            # Lower backlight intensity = more natural
            backlight_score = max(0, 1.0 - min(backlight_intensity / 20, 1.0))
            
            # Combine scores
            final_lighting_score = (lighting_score + backlight_score) / 2
            
            return {
                'score': float(final_lighting_score),
                'lighting_variance': float(lighting_variance if region_means else 0),
                'backlight_intensity': float(backlight_intensity),
                'gradient_mean': float(np.mean(gradient_mag)),
                'confidence': float(abs(final_lighting_score - 0.5) * 2)
            }
            
        except Exception as e:
            logger.warning(f"Lighting pattern analysis failed: {str(e)}")
            return {'score': 0.5, 'confidence': 0.0}
    
    def _make_final_decision(self, results: Dict, final_score: float, threshold: float) -> bool:
        """Make final decision with additional checks"""
        try:
            # Start with score-based decision
            basic_decision = final_score >= threshold
            
            # Count red flags instead of immediate rejection
            red_flags = 0
            red_flag_details = []
            
            # Rule 1: Low compression score (but not immediate rejection)
            compression_score = results.get('compression', {}).get('score', 0.5)
            if compression_score < 0.08:  # Very low = red flag
                red_flags += 1
                red_flag_details.append(f"Very low compression: {compression_score:.3f}")
            
            # Rule 2: Count low scores
            if results.get('color', {}).get('score', 0.5) < 0.25:  # More strict threshold
                red_flags += 1
                red_flag_details.append(f"Low color: {results.get('color', {}).get('score', 0.5):.3f}")
            if results.get('edge', {}).get('score', 0.5) < 0.20:   # More strict threshold  
                red_flags += 1
                red_flag_details.append(f"Low edge: {results.get('edge', {}).get('score', 0.5):.3f}")
            if results.get('noise', {}).get('score', 0.5) < 0.25:  # More strict threshold
                red_flags += 1
                red_flag_details.append(f"Low noise: {results.get('noise', {}).get('score', 0.5):.3f}")
                
            # Rule 3: Perfect texture and frequency scores are suspicious
            texture_score = results.get('texture', {}).get('score', 0.5)
            frequency_score = results.get('frequency', {}).get('score', 0.5)
            
            if texture_score >= 0.99 and frequency_score >= 0.99:
                red_flags += 1
                red_flag_details.append(f"Perfect scores: texture={texture_score:.3f}, freq={frequency_score:.3f}")
            
            # Rule 4: Check digital artifacts
            digital_score = results.get('digital_artifacts', {}).get('score', 0.5)
            if digital_score < 0.3:
                red_flags += 1
                red_flag_details.append(f"Low digital artifacts: {digital_score:.3f}")
            
            # Rule 5: Check lighting
            lighting_score = results.get('lighting', {}).get('score', 0.5)
            if lighting_score < 0.3:
                red_flags += 1
                red_flag_details.append(f"Low lighting: {lighting_score:.3f}")
            
            # Final decision based on red flags and basic score
            if red_flags >= 4:  # Many red flags = definitely spoof
                return False
            elif red_flags >= 3 and final_score < 0.6:  # Some red flags + low score = spoof
                return False
            elif red_flags >= 2 and final_score < 0.4:  # Few red flags but very low score = spoof
                return False
            else:
                return basic_decision  # Trust the weighted score
            
        except Exception as e:
            logger.warning(f"Final decision logic failed: {str(e)}")
            return final_score >= threshold
        
    def _get_spoof_indicators(self, results: Dict) -> List[str]:
        """Get list of detected spoof indicators"""
        indicators = []
        
        try:
            for method, result in results.items():
                score = result.get('score', 0.5)
                if score < 0.3:
                    method_name = method.replace('_', ' ').title()
                    indicators.append(f"Low {method_name} Score ({score:.2f})")
            
            # Add specific indicators
            if results.get('digital_artifacts', {}).get('compression_artifacts', 0) > 0.7:
                indicators.append("High compression artifacts detected")
            
            if results.get('lighting', {}).get('backlight_intensity', 0) > 15:
                indicators.append("Screen backlight pattern detected")
            
            if results.get('frequency', {}).get('freq_ratio', 0) < 0.5:
                indicators.append("Reduced high-frequency content (screen characteristic)")
            
        except Exception as e:
            logger.warning(f"Spoof indicator detection failed: {str(e)}")
        
        return indicators
    
    def get_version(self) -> str:
        """Get detector version"""
        return self.version
    
    def get_detection_methods(self) -> List[str]:
        """Get list of detection methods"""
        return self.detection_methods.copy()
    
    def set_threshold(self, threshold: float):
        """Set default detection threshold"""
        if 0.0 <= threshold <= 1.0:
            self.default_threshold = threshold
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def set_sensitivity(self, sensitivity: str):
        """Set detection sensitivity"""
        if sensitivity in ['low', 'medium', 'high']:
            self.sensitivity = sensitivity
            # Adjust detection parameters based on sensitivity
            if sensitivity == 'low':
                self.default_threshold = 0.3
            elif sensitivity == 'medium':
                self.default_threshold = 0.5
            elif sensitivity == 'high':
                self.default_threshold = 0.7
        else:
            raise ValueError("Sensitivity must be 'low', 'medium', or 'high'")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current detector configuration"""
        return {
            'version': self.version,
            'default_threshold': self.default_threshold,
            'sensitivity': self.sensitivity,
            'detection_methods': self.detection_methods,
            'face_detection_enabled': self.face_cascade is not None
        }
    
    def detect_multiple_faces(self, image: np.ndarray, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Detect liveness for multiple faces in an image
        
        Args:
            image: Input image as numpy array
            threshold: Detection threshold (0.0 to 1.0)
            
        Returns:
            List of detection results for each face
        """
        if threshold is None:
            threshold = self.default_threshold
            
        try:
            results = []
            
            if self.face_cascade is None:
                # No face detection available, analyze whole image
                result = self.detect_liveness(image, threshold)
                result['face_id'] = 0
                result['bbox'] = None
                return [result]
            
            # Detect all faces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                # No faces detected, analyze whole image
                result = self.detect_liveness(image, threshold)
                result['face_id'] = 0
                result['bbox'] = None
                return [result]
            
            # Analyze each face
            for i, (x, y, w, h) in enumerate(faces):
                face_region = image[y:y+h, x:x+w]
                result = self.detect_liveness(face_region, threshold)
                result['face_id'] = i
                result['bbox'] = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Multiple face detection failed: {str(e)}")
            return [{
                'is_live': False,
                'confidence': 0.0,
                'liveness_score': 0.0,
                'spoof_type': 'analysis_error',
                'face_id': 0,
                'bbox': None,
                'details': {'error': str(e)}
            }]
    
    def analyze_video_frame(self, frame: np.ndarray, frame_number: int, threshold: float = None) -> Dict[str, Any]:
        """
        Analyze a single video frame for liveness
        
        Args:
            frame: Video frame as numpy array
            frame_number: Frame number in sequence
            threshold: Detection threshold (0.0 to 1.0)
            
        Returns:
            Dictionary with detection results including temporal information
        """
        if threshold is None:
            threshold = self.default_threshold
            
        try:
            # Standard liveness detection
            result = self.detect_liveness(frame, threshold)
            
            # Add frame-specific information
            result['frame_number'] = frame_number
            result['frame_quality'] = self._assess_frame_quality(frame)
            
            # Additional video-specific analysis
            result['motion_blur'] = self._detect_motion_blur(frame)
            result['compression_artifacts'] = self._detect_compression_artifacts(frame)
            
            return result
            
        except Exception as e:
            logger.error(f"Video frame analysis failed: {str(e)}")
            return {
                'is_live': False,
                'confidence': 0.0,
                'liveness_score': 0.0,
                'spoof_type': 'analysis_error',
                'frame_number': frame_number,
                'details': {'error': str(e)}
            }
    
    def _assess_frame_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """Assess video frame quality"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            # Calculate frame quality metrics
            quality_metrics = self._assess_image_quality(gray)
            
            # Additional video-specific metrics
            # Check for interlacing artifacts
            interlace_score = self._detect_interlacing(gray)
            quality_metrics['interlacing'] = interlace_score
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Frame quality assessment failed: {str(e)}")
            return {'overall_quality': 0.5}
    
    def _detect_motion_blur(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect motion blur in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            # Calculate variance of Laplacian (blur detection)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = np.var(laplacian)
            
            # Lower values indicate more blur
            is_blurred = blur_score < 100
            
            return {
                'blur_score': blur_score,
                'is_blurred': is_blurred,
                'blur_confidence': min(abs(blur_score - 100) / 100, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Motion blur detection failed: {str(e)}")
            return {'blur_score': 0.0, 'is_blurred': False, 'blur_confidence': 0.0}
    
    def _detect_compression_artifacts(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect compression artifacts in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            
            # Look for block artifacts (8x8 DCT blocks)
            h, w = gray.shape
            block_size = 8
            
            # Calculate block variance
            block_variances = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    block_variances.append(np.var(block))
            
            if block_variances:
                avg_block_variance = np.mean(block_variances)
                compression_score = min(avg_block_variance / 1000, 1.0)
            else:
                compression_score = 0.5
            
            return {
                'compression_score': compression_score,
                'has_artifacts': compression_score < 0.3,
                'avg_block_variance': avg_block_variance if block_variances else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Compression artifact detection failed: {str(e)}")
            return {'compression_score': 0.5, 'has_artifacts': False}
    
    def _detect_interlacing(self, gray: np.ndarray) -> float:
        """Detect interlacing artifacts"""
        try:
            # Check for interlacing by comparing odd and even rows
            odd_rows = gray[1::2, :]
            even_rows = gray[::2, :]
            
            # Resize to same dimensions
            min_rows = min(odd_rows.shape[0], even_rows.shape[0])
            odd_rows = odd_rows[:min_rows, :]
            even_rows = even_rows[:min_rows, :]
            
            # Calculate difference
            diff = np.abs(odd_rows.astype(float) - even_rows.astype(float))
            interlace_score = np.mean(diff)
            
            return min(interlace_score / 50, 1.0)
            
        except Exception as e:
            logger.warning(f"Interlacing detection failed: {str(e)}")
            return 0.0
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported image formats"""
        return ['JPG', 'JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']
    
    def validate_image_format(self, image: np.ndarray) -> bool:
        """Validate if image format is supported"""
        try:
            if image is None or image.size == 0:
                return False
            
            # Check dimensions
            if len(image.shape) not in [2, 3]:
                return False
            
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                return False
            
            # Check data type
            if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Image format validation failed: {str(e)}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for optimal analysis"""
        try:
            # Validate input
            if not self.validate_image_format(image):
                raise ValueError("Invalid image format")
            
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.dtype in [np.float32, np.float64]:
                    image = (image * 255).astype(np.uint8)
                elif image.dtype == np.uint16:
                    image = (image / 256).astype(np.uint8)
            
            # Resize if too large (for performance)
            h, w = image.shape[:2]
            max_size = 1024
            
            if max(h, w) > max_size:
                if h > w:
                    new_h, new_w = max_size, int(w * max_size / h)
                else:
                    new_h, new_w = int(h * max_size / w), max_size
                
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Ensure minimum size
            min_size = 64
            if min(h, w) < min_size:
                scale = min_size / min(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'detector_version': self.version,
            'methods_count': len(self.detection_methods),
            'face_detection_available': self.face_cascade is not None,
            'current_threshold': self.default_threshold,
            'current_sensitivity': self.sensitivity,
            'supported_formats': self.get_supported_formats(),
            'max_recommended_size': '1024x1024',
            'min_recommended_size': '64x64'
        }