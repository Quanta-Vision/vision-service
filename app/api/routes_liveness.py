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
    threshold: Optional[float] = Field(default=0.6, ge=0.0, le=1.0, description="Liveness confidence threshold")
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
    threshold: Optional[float] = Field(default=0.6, ge=0.0, le=1.0)
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
                "confidence": 0.85,
                "method_used": "multi_feature_ensemble",
                "detection_time": 0.234,
                "face_quality": 0.78,
                "spoof_indicators": {
                    "texture_richness": 0.82,
                    "skin_realism": 0.75,
                    "rgb_independence": 0.68
                },
                "recommendations": ["High confidence live detection"],
                "timestamp": 1625097600000,
                "datetime": "2021-07-01 00:00:00",
                "threshold_used": 0.6
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
            "max_face_size": detector.max_face_size
        }
        
        return SystemHealthResponse(
            status="healthy",
            message="Enhanced liveness detection service is running",
            version="2.0.0",
            available_methods=available_methods,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemHealthResponse(
            status="error",
            message=f"Service health check failed: {str(e)}",
            version="2.0.0",
            available_methods=[],
            system_info={}
        )

@router_liveness.post("/check-liveness", 
                     tags=["Liveness Detection"], 
                     summary="Enhanced Liveness Detection",
                     response_model=LivenessResponse,
                     dependencies=[Depends(verify_api_key)])
async def check_liveness_enhanced_api(
    request: Request,
    image: UploadFile = File(..., description="Face image for liveness detection"),
    method: LivenessMethodEnum = Form(LivenessMethodEnum.ENSEMBLE),
    threshold: float = Form(0.6, ge=0.0, le=1.0),
    include_details: bool = Form(True),
    include_face_info: bool = Form(False)
):
    """
    Enhanced liveness detection with multiple algorithms and detailed analysis.
    
    **Features:**
    - Multiple detection methods (ONNX, InsightFace, Multi-feature, Ensemble)
    - Advanced spoof detection indicators
    - Face quality assessment
    - Detailed recommendations
    - Configurable confidence threshold
    
    **Parameters:**
    - **image**: Face image file (JPEG, PNG, etc.)
    - **method**: Detection method to use
    - **threshold**: Confidence threshold for liveness decision
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
        if threshold != 0.6:
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
        logger.info(f"Liveness detection - Consumer: {consumer}, "
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
                     summary="Batch Liveness Detection",
                     response_model=BatchLivenessResponse,
                     dependencies=[Depends(verify_api_key)])
async def check_liveness_batch_api(
    request: Request,
    images: List[UploadFile] = File(..., description="Multiple face images for batch processing"),
    method: LivenessMethodEnum = Form(LivenessMethodEnum.ENSEMBLE),
    threshold: float = Form(0.6, ge=0.0, le=1.0),
    include_details: bool = Form(True),
    max_concurrent: int = Form(3, ge=1, le=10)
):
    """
    Batch liveness detection for multiple images with concurrent processing.
    
    **Features:**
    - Process multiple images simultaneously
    - Configurable concurrency level
    - Detailed batch processing statistics
    - Individual results for each image
    
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
                    if threshold != 0.6:
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
        logger.info(f"Batch liveness detection - Consumer: {consumer}, "
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
    """
    try:
        # Convert uploaded file to cv2 image
        img_cv2 = await uploadfile_to_cv2_image(image)
        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Could not process image")
        
        # Run legacy detection
        loop = asyncio.get_event_loop()
        score = await loop.run_in_executor(
            thread_pool,
            lambda: check_liveness_antispoof_mn3(img_cv2, model_path)
        )
        
        # Legacy response format
        if score == -1.0:
            return {
                "is_live": False,
                "score": score,
                "threshold": 0.5,
                "msg": "No face detected",
                "timestamp": int(time.time() * 1000),
                "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        is_live = score > 0.5
        logger.info(f"Legacy liveness score: {score:.3f} - {'LIVE' if is_live else 'SPOOF'}")
        
        return {
            "is_live": is_live,
            "score": float(score),
            "threshold": 0.5,
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
                "accuracy": "Medium-High",
                "speed": "Medium"
            },
            "ensemble": {
                "available": True,
                "description": "Combination of all available methods for best accuracy",
                "accuracy": "Highest",
                "speed": "Medium",
                "recommended": True
            }
        }
        
        return {
            "available_methods": methods,
            "default_method": "ensemble",
            "default_threshold": 0.6,
            "recommendations": [
                "Use 'ensemble' method for best accuracy",
                "Use 'multi_feature' for consistent performance without external models",
                "Adjust threshold based on your security requirements"
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
            "thread_pool_info": {
                "max_workers": thread_pool._max_workers,
                "active_threads": len(thread_pool._threads) if hasattr(thread_pool, '_threads') else 0
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve statistics: {str(e)}")

# Health check endpoint for monitoring
@router_liveness.get("/health", 
                    tags=["Liveness Health"], 
                    summary="Service Health Check")
async def service_health():
    """Simple health check endpoint for monitoring systems."""
    return {"status": "healthy", "timestamp": int(time.time())}
