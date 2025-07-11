import cv2
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field, validator
from enum import Enum

# Import your utilities and auth
from app.utils.auth import verify_api_key
from app.utils.utils_function import uploadfile_to_cv2_image
from app.utils.consumer import get_consumer

# Import the enhanced liveness detection
from app.services.enhance_liveness_detection import (
    check_liveness_enhanced, 
    check_liveness_antispoof_mn3,
    LivenessResult,
    LivenessMethod,
    get_detector_instance
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router_liveness = APIRouter()

# Thread pool for async processing
thread_pool = ThreadPoolExecutor(max_workers=4)

class LivenessMethodEnum(str, Enum):
    """API enum for liveness detection methods"""
    ONNX_MODEL = "onnx"
    INSIGHTFACE = "insightface"
    MULTI_FEATURE = "multi_feature"
    ENSEMBLE = "ensemble"

class LivenessRequest(BaseModel):
    """Request model for liveness detection with validation"""
    method: Optional[LivenessMethodEnum] = LivenessMethodEnum.ENSEMBLE
    threshold: Optional[float] = Field(default=0.55, ge=0.0, le=1.0, description="Liveness confidence threshold")
    include_details: Optional[bool] = Field(default=True, description="Include detailed analysis results")
    include_face_info: Optional[bool] = Field(default=False, description="Include face detection info")
    
    @validator('threshold')
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Threshold must be between 0.0 and 1.0')
        return v

class BatchLivenessRequest(BaseModel):
    """Request model for batch liveness detection"""
    method: Optional[LivenessMethodEnum] = LivenessMethodEnum.ENSEMBLE
    threshold: Optional[float] = Field(default=0.55, ge=0.0, le=1.0)
    include_details: Optional[bool] = Field(default=True)
    max_concurrent: Optional[int] = Field(default=3, ge=1, le=10, description="Maximum concurrent processing")

class LivenessResponse(BaseModel):
    """Enhanced response model for liveness detection"""
    is_live: bool
    confidence: float
    method_used: str
    detection_time: float
    face_quality: Optional[float] = None
    spoof_indicators: Optional[Dict[str, float]] = None
    recommendations: Optional[List[str]] = None
    timestamp: int
    datetime: str
    threshold_used: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_live": True,
                "confidence": 0.72,
                "method_used": "adaptive_ensemble_v3",
                "detection_time": 0.234,
                "face_quality": 0.78,
                "spoof_indicators": {
                    "texture_richness": 0.65,
                    "skin_realism": 0.75,
                    "rgb_independence": 0.45,
                    "screen_pattern_absence": 0.85
                },
                "recommendations": ["Good confidence live detection"],
                "timestamp": 1625097600000,
                "datetime": "2021-07-01 00:00:00",
                "threshold_used": 0.55
            }
        }

class BatchLivenessResponse(BaseModel):
    """Response model for batch liveness detection"""
    total_images: int
    processed_images: int
    failed_images: int
    processing_time: float
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

class SystemHealthResponse(BaseModel):
    """System health check response"""
    status: str
    message: str
    version: str
    available_methods: List[str]
    system_info: Dict[str, Any]

@router_liveness.get("/", 
                    tags=["Liveness Health"], 
                    summary="Liveness Service Health Check",
                    response_model=SystemHealthResponse)
async def health_check():
    """
    Check the health status of the liveness detection service.
    Returns information about available methods and system status.
    """
    try:
        # Get detector instance to check model availability
        detector = get_detector_instance()
        
        available_methods = []
        if detector._onnx_session:
            available_methods.append("onnx_model")
        if detector._antispoof_model:
            available_methods.append("insightface_model")
        available_methods.extend(["multi_feature", "ensemble"])
        
        system_info = {
            "onnx_model_available": detector._onnx_session is not None,
            "insightface_model_available": detector._antispoof_model is not None,
            "face_detector_available": detector._insightface_detector is not None,
            "default_threshold": detector.liveness_threshold,
            "min_face_size": detector.min_face_size,
            "max_face_size": detector.max_face_size,
            "adaptive_scoring": True
        }
        
        return SystemHealthResponse(
            status="healthy",
            message="Enhanced liveness detection service with adaptive scoring",
            version="3.0.0",
            available_methods=available_methods,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemHealthResponse(
            status="error",
            message=f"Service health check failed: {str(e)}",
            version="3.0.0",
            available_methods=[],
            system_info={}
        )

@router_liveness.post("/check-liveness", 
                     tags=["Liveness Detection"], 
                     summary="Adaptive Liveness Detection",
                     response_model=LivenessResponse,
                     dependencies=[Depends(verify_api_key)])
async def check_liveness_enhanced_api(
    request: Request,
    image: UploadFile = File(..., description="Face image for liveness detection"),
    method: LivenessMethodEnum = Form(LivenessMethodEnum.ENSEMBLE),
    threshold: float = Form(0.55, ge=0.0, le=1.0),
    include_details: bool = Form(True),
    include_face_info: bool = Form(False)
):
    """
    Adaptive liveness detection with balanced scoring for real vs screen photos.
    
    **Features:**
    - Adaptive scoring system that's more balanced for real images
    - Automatic detection of obvious screen captures vs ambiguous cases
    - Multiple detection methods (ONNX, InsightFace, Multi-feature, Ensemble)
    - Advanced spoof detection indicators with balanced weights
    - Face quality assessment
    - Detailed recommendations
    - Configurable confidence threshold (default: 0.55 - more balanced)
    
    **Parameters:**
    - **image**: Face image file (JPEG, PNG, etc.)
    - **method**: Detection method to use
    - **threshold**: Confidence threshold for liveness decision (0.55 recommended)
    - **include_details**: Include detailed analysis results
    - **include_face_info**: Include face detection metadata
    """
    start_time = time.time()
    
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Convert uploaded file to cv2 image
        img_cv2 = await uploadfile_to_cv2_image(image)
        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Could not process image")
        
        # Map enum to detection method
        method_map = {
            LivenessMethodEnum.ONNX_MODEL: LivenessMethod.ONNX_MODEL,
            LivenessMethodEnum.INSIGHTFACE: LivenessMethod.INSIGHTFACE,
            LivenessMethodEnum.MULTI_FEATURE: LivenessMethod.MULTI_FEATURE,
            LivenessMethodEnum.ENSEMBLE: LivenessMethod.ENSEMBLE
        }
        
        # Run liveness detection in thread pool
        loop = asyncio.get_event_loop()
        detection_method = method_map.get(method, LivenessMethod.ENSEMBLE)
        
        result: LivenessResult = await loop.run_in_executor(
            thread_pool,
            lambda: check_liveness_enhanced(img_cv2)
        )
        
        # Override threshold if provided
        if threshold != 0.55:
            result.is_live = result.confidence > threshold
        
        # Prepare response
        now = datetime.utcnow()
        timestamp_ms = int(time.time() * 1000)
        
        response_data = {
            "is_live": result.is_live,
            "confidence": round(result.confidence, 3),
            "method_used": result.method_used,
            "detection_time": round(result.detection_time, 3),
            "timestamp": timestamp_ms,
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold_used": threshold
        }
        
        # Add optional details
        if include_details:
            response_data["face_quality"] = round(result.face_quality, 3) if result.face_quality else None
            response_data["spoof_indicators"] = {
                k: round(v, 3) for k, v in result.spoof_indicators.items()
            } if result.spoof_indicators else None
            response_data["recommendations"] = result.recommendations
        
        # Log detection result
        consumer = get_consumer(request)
        logger.info(f"Adaptive liveness detection - Consumer: {consumer}, "
                   f"Method: {method}, Confidence: {result.confidence:.3f}, "
                   f"Is Live: {result.is_live}, Time: {result.detection_time:.3f}s")
        
        return LivenessResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in liveness detection: {e}")
        raise HTTPException(status_code=500, detail=f"Liveness detection failed: {str(e)}")

@router_liveness.post("/check-liveness-batch", 
                     tags=["Liveness Detection"], 
                     summary="Batch Adaptive Liveness Detection",
                     response_model=BatchLivenessResponse,
                     dependencies=[Depends(verify_api_key)])
async def check_liveness_batch_api(
    request: Request,
    images: List[UploadFile] = File(..., description="Multiple face images for batch processing"),
    method: LivenessMethodEnum = Form(LivenessMethodEnum.ENSEMBLE),
    threshold: float = Form(0.55, ge=0.0, le=1.0),
    include_details: bool = Form(True),
    max_concurrent: int = Form(3, ge=1, le=10)
):
    """
    Batch adaptive liveness detection for multiple images with concurrent processing.
    
    **Features:**
    - Process multiple images simultaneously with adaptive scoring
    - Configurable concurrency level
    - Detailed batch processing statistics
    - Individual results for each image
    - Balanced detection for real vs screen photos
    
    **Limitations:**
    - Maximum 20 images per batch
    - Maximum 10 concurrent processes
    """
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(images) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 images per batch")
        
        # Validate all images
        for i, image in enumerate(images):
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {i+1} must be an image")
        
        # Process images concurrently
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_image(img_file: UploadFile, index: int):
            async with semaphore:
                try:
                    # Convert to cv2 image
                    img_cv2 = await uploadfile_to_cv2_image(img_file)
                    if img_cv2 is None:
                        return {
                            "index": index,
                            "filename": img_file.filename,
                            "success": False,
                            "error": "Could not process image"
                        }
                    
                    # Run detection
                    loop = asyncio.get_event_loop()
                    result: LivenessResult = await loop.run_in_executor(
                        thread_pool,
                        lambda: check_liveness_enhanced(img_cv2)
                    )
                    
                    # Override threshold
                    if threshold != 0.55:
                        result.is_live = result.confidence > threshold
                    
                    response_data = {
                        "index": index,
                        "filename": img_file.filename,
                        "success": True,
                        "is_live": result.is_live,
                        "confidence": round(result.confidence, 3),
                        "method_used": result.method_used,
                        "detection_time": round(result.detection_time, 3),
                        "face_quality": round(result.face_quality, 3) if result.face_quality else None
                    }
                    
                    if include_details:
                        response_data["spoof_indicators"] = {
                            k: round(v, 3) for k, v in result.spoof_indicators.items()
                        } if result.spoof_indicators else None
                        response_data["recommendations"] = result.recommendations
                    
                    return response_data
                    
                except Exception as e:
                    logger.error(f"Error processing image {index}: {e}")
                    return {
                        "index": index,
                        "filename": img_file.filename,
                        "success": False,
                        "error": str(e)
                    }
        
        # Process all images
        tasks = [process_single_image(img, i) for i, img in enumerate(images)]
        results = await asyncio.gather(*tasks)
        
        # Calculate statistics
        total_images = len(images)
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        processed_images = len(successful_results)
        failed_images = len(failed_results)
        processing_time = time.time() - start_time
        
        # Calculate summary statistics
        if successful_results:
            live_count = sum(1 for r in successful_results if r.get("is_live", False))
            avg_confidence = sum(r.get("confidence", 0) for r in successful_results) / len(successful_results)
            avg_detection_time = sum(r.get("detection_time", 0) for r in successful_results) / len(successful_results)
        else:
            live_count = 0
            avg_confidence = 0.0
            avg_detection_time = 0.0
        
        summary = {
            "live_faces": live_count,
            "spoof_faces": processed_images - live_count,
            "live_percentage": (live_count / processed_images * 100) if processed_images > 0 else 0,
            "average_confidence": round(avg_confidence, 3),
            "average_detection_time": round(avg_detection_time, 3),
            "success_rate": (processed_images / total_images * 100) if total_images > 0 else 0
        }
        
        # Log batch result
        consumer = get_consumer(request)
        logger.info(f"Batch adaptive liveness detection - Consumer: {consumer}, "
                   f"Total: {total_images}, Processed: {processed_images}, "
                   f"Failed: {failed_images}, Time: {processing_time:.3f}s")
        
        return BatchLivenessResponse(
            total_images=total_images,
            processed_images=processed_images,
            failed_images=failed_images,
            processing_time=round(processing_time, 3),
            results=results,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch liveness detection: {e}")
        raise HTTPException(status_code=500, detail=f"Batch liveness detection failed: {str(e)}")

@router_liveness.post("/check-liveness-legacy", 
                     tags=["Liveness Detection"], 
                     summary="Legacy Liveness Detection (Backward Compatible)",
                     dependencies=[Depends(verify_api_key)])
async def check_liveness_legacy_api(
    request: Request,
    image: UploadFile = File(...),
    model_path: Optional[str] = Form(None)
):
    """
    Legacy liveness detection endpoint for backward compatibility.
    Returns the same format as the original check-spoofing-mn3 endpoint.
    Now uses adaptive scoring for better real image detection.
    """
    try:
        # Convert uploaded file to cv2 image
        img_cv2 = await uploadfile_to_cv2_image(image)
        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Could not process image")
        
        # Run legacy detection with adaptive scoring
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(
            thread_pool,
            lambda: check_liveness_antispoof_mn3(img_cv2, model_path)
        )
        
        # Legacy response format with adaptive threshold
        if score == -1.0:
            return {
                "is_live": False,
                "score": score,
                "threshold": 0.55,
                "msg": "No face detected",
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        is_live = score > 0.55  # Updated threshold
        logger.info(f"Legacy adaptive liveness score: {score:.3f} - {'LIVE' if is_live else 'SPOOF'}")
        
        return {
            "is_live": is_live,
            "score": float(score),
            "threshold": 0.55,
            "msg": "Live face detected" if is_live else "Spoofing detected",
            "timestamp": int(time.time() * 1000),
            "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in legacy liveness detection: {e}")
        raise HTTPException(status_code=500, detail=f"Legacy liveness detection failed: {str(e)}")

@router_liveness.get("/methods", 
                    tags=["Liveness Configuration"], 
                    summary="Available Detection Methods")
async def get_available_methods():
    """
    Get information about available liveness detection methods.
    """
    try:
        detector = get_detector_instance()
        
        methods = {
            "onnx_model": {
                "available": detector._onnx_session is not None,
                "description": "ONNX neural network model for liveness detection",
                "accuracy": "High (if model available)",
                "speed": "Fast"
            },
            "insightface": {
                "available": detector._antispoof_model is not None,
                "description": "InsightFace built-in antispoof model",
                "accuracy": "High (if model available)",
                "speed": "Fast"
            },
            "multi_feature": {
                "available": True,
                "description": "Advanced multi-feature analysis without neural networks",
                "accuracy": "High",
                "speed": "Medium"
            },
            "ensemble": {
                "available": True,
                "description": "Adaptive ensemble method with balanced real vs fake detection",
                "accuracy": "Highest",
                "speed": "Medium",
                "recommended": True,
                "features": [
                    "Adaptive scoring system",
                    "Balanced real image detection",
                    "Screen pattern detection",
                    "Enhanced RGB correlation analysis", 
                    "Compression artifact detection",
                    "Pixel uniformity analysis",
                    "Multi-scale texture analysis"
                ]
            }
        }
        
        return {
            "available_methods": methods,
            "default_method": "ensemble",
            "default_threshold": 0.55,
            "recommendations": [
                "Use 'ensemble' method for balanced real vs fake detection",
                "Use 'multi_feature' for consistent performance without external models",
                "Threshold 0.55 recommended for balanced security vs usability",
                "Adaptive scoring system in v3.0.0 provides better real image detection"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting available methods: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve methods: {str(e)}")

@router_liveness.post("/configure", 
                     tags=["Liveness Configuration"], 
                     summary="Configure Detection Parameters",
                     dependencies=[Depends(verify_api_key)])
async def configure_detection_parameters(
    request: Request,
    liveness_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
    face_quality_threshold: Optional[float] = Form(None, ge=0.0, le=1.0),
    min_face_size: Optional[int] = Form(None, ge=20, le=200),
    max_face_size: Optional[int] = Form(None, ge=200, le=1000)
):
    """
    Configure detection parameters for the current session.
    Note: Configuration is temporary and will reset when service restarts.
    """
    try:
        detector = get_detector_instance()
        
        # Update parameters if provided
        updated_params = {}
        if liveness_threshold is not None:
            detector.liveness_threshold = liveness_threshold
            updated_params["liveness_threshold"] = liveness_threshold
        
        if face_quality_threshold is not None:
            detector.face_quality_threshold = face_quality_threshold
            updated_params["face_quality_threshold"] = face_quality_threshold
        
        if min_face_size is not None:
            detector.min_face_size = min_face_size
            updated_params["min_face_size"] = min_face_size
        
        if max_face_size is not None:
            detector.max_face_size = max_face_size
            updated_params["max_face_size"] = max_face_size
        
        current_config = {
            "liveness_threshold": detector.liveness_threshold,
            "face_quality_threshold": detector.face_quality_threshold,
            "min_face_size": detector.min_face_size,
            "max_face_size": detector.max_face_size
        }
        
        consumer = get_consumer(request)
        logger.info(f"Configuration updated - Consumer: {consumer}, "
                   f"Updated: {updated_params}")
        
        return {
            "message": "Configuration updated successfully",
            "updated_parameters": updated_params,
            "current_configuration": current_config,
            "note": "Configuration is temporary and will reset on service restart"
        }
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

@router_liveness.get("/stats", 
                    tags=["Liveness Statistics"], 
                    summary="Detection Statistics",
                    dependencies=[Depends(verify_api_key)])
async def get_detection_statistics(request: Request):
    """
    Get basic statistics about the liveness detection service.
    Note: This is a simple implementation. For production, consider using proper monitoring.
    """
    try:
        detector = get_detector_instance()
        
        # Basic system information
        stats = {
            "service_status": "running",
            "version": "3.0.0 - Adaptive Scoring System",
            "available_methods": {
                "onnx_model": detector._onnx_session is not None,
                "insightface_model": detector._antispoof_model is not None,
                "multi_feature": True,
                "ensemble": True
            },
            "current_configuration": {
                "liveness_threshold": detector.liveness_threshold,
                "face_quality_threshold": detector.face_quality_threshold,
                "min_face_size": detector.min_face_size,
                "max_face_size": detector.max_face_size
            },
            "adaptive_features": [
                "Adaptive scoring system",
                "Balanced real vs fake detection",
                "Automatic screen vs ambiguous case detection",
                "Enhanced real image tolerance",
                "Screen pattern detection",
                "Enhanced RGB correlation analysis",
                "Compression artifact detection", 
                "Pixel uniformity analysis",
                "Multi-scale texture analysis",
                "Enhanced moiré detection"
            ],
            "scoring_methods": {
                "aggressive_screen_scoring": "For obvious screen captures",
                "balanced_scoring": "For ambiguous cases and real images",
                "adaptive_selection": "Automatic method selection based on indicators"
            },
            "thread_pool_info": {
                "max_workers": thread_pool._max_workers,
                "active_threads": len(thread_pool._threads) if hasattr(thread_pool, '_threads') else 0
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve statistics: {str(e)}")

@router_liveness.post("/test-adaptive", 
                     tags=["Liveness Testing"], 
                     summary="Test Adaptive Scoring",
                     dependencies=[Depends(verify_api_key)])
async def test_adaptive_scoring(
    request: Request,
    image: UploadFile = File(..., description="Face image for adaptive scoring test"),
    force_method: Optional[str] = Form(None, description="Force specific scoring method: 'aggressive' or 'balanced'")
):
    """
    Test the adaptive scoring system with detailed breakdown.
    Shows how the system would score using both aggressive and balanced methods.
    """
    try:
        # Convert uploaded file to cv2 image
        img_cv2 = await uploadfile_to_cv2_image(image)
        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Could not process image")
        
        # Get detector instance
        detector = get_detector_instance()
        
        # Analyze face
        face_data = detector.detect_and_analyze_face(img_cv2)
        if face_data is None:
            raise HTTPException(status_code=400, detail="No face detected")
        
        face_resized = cv2.resize(face_data['face_img'], (128, 128))
        indicators = detector._advanced_spoof_detection(face_resized)
        
        # Test both scoring methods
        aggressive_score = detector._aggressive_screen_scoring(indicators)
        balanced_score = detector._balanced_scoring(indicators)
        
        # Determine which method would be used
        moire_score = indicators.get('moire_absence', 1.0)
        rgb_score = indicators.get('rgb_independence', 1.0)
        screen_pattern_score = indicators.get('screen_pattern_absence', 1.0)
        
        is_definite_screen = (
            moire_score < 0.1 and 
            rgb_score < 0.1 and 
            screen_pattern_score < 0.2
        )
        
        # Apply forced method if specified
        if force_method == "aggressive":
            final_score = aggressive_score
            method_used = "aggressive (forced)"
        elif force_method == "balanced":
            final_score = balanced_score
            method_used = "balanced (forced)"
        else:
            final_score = aggressive_score if is_definite_screen else balanced_score
            method_used = "aggressive (auto)" if is_definite_screen else "balanced (auto)"
        
        # Prepare response
        response = {
            "image_analysis": {
                "face_detected": True,
                "face_quality": round(face_data['quality'], 3),
                "is_definite_screen": is_definite_screen
            },
            "spoof_indicators": {k: round(v, 3) for k, v in indicators.items()},
            "scoring_comparison": {
                "aggressive_score": round(aggressive_score, 3),
                "balanced_score": round(balanced_score, 3),
                "final_score": round(final_score, 3),
                "method_used": method_used,
                "score_difference": round(balanced_score - aggressive_score, 3)
            },
            "decision_criteria": {
                "moire_absence": round(moire_score, 3),
                "rgb_independence": round(rgb_score, 3),
                "screen_pattern_absence": round(screen_pattern_score, 3),
                "triggers_aggressive": is_definite_screen
            },
            "thresholds": {
                "current_threshold": detector.liveness_threshold,
                "aggressive_result": aggressive_score > detector.liveness_threshold,
                "balanced_result": balanced_score > detector.liveness_threshold,
                "final_result": final_score > detector.liveness_threshold
            },
            "recommendations": detector._generate_recommendations(final_score, indicators, face_data['quality'])
        }
        
        # Log test result
        consumer = get_consumer(request)
        logger.info(f"Adaptive scoring test - Consumer: {consumer}, "
                   f"Aggressive: {aggressive_score:.3f}, Balanced: {balanced_score:.3f}, "
                   f"Method: {method_used}, Final: {final_score:.3f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in adaptive scoring test: {e}")
        raise HTTPException(status_code=500, detail=f"Adaptive scoring test failed: {str(e)}")

# Health check endpoint for monitoring
@router_liveness.get("/health", 
                    tags=["Liveness Health"], 
                    summary="Service Health Check")
async def service_health():
    """Simple health check endpoint for monitoring systems."""
    return {
        "status": "healthy", 
        "timestamp": int(time.time()),
        "version": "3.0.0",
        "adaptive_scoring": True
    }