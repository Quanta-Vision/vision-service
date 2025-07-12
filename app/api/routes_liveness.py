# routes_liveness.py
import time
import io
import base64
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import cv2
import numpy as np
from PIL import Image
import logging
from app.services.enhance_liveness_detection import LivenessDetector
from app.utils.auth import verify_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router_liveness = APIRouter(tags=["Liveness Detection"])

# Initialize the liveness detector
liveness_detector = LivenessDetector()

# Pydantic models for request/response
class LivenessResponse(BaseModel):
    is_live: bool = Field(..., description="True if image is from real camera, False if spoofed")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    spoof_type: Optional[str] = Field(None, description="Type of spoofing detected if any")
    analysis_details: Dict[str, Any] = Field(..., description="Detailed analysis results")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    timestamp: int = Field(..., description="Unix timestamp of analysis")

class LivenessRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    threshold: Optional[float] = Field(0.5, description="Detection threshold (0.0 to 1.0)")

def decode_image(image_data) -> np.ndarray:
    """Decode image from various formats to numpy array"""
    try:
        if isinstance(image_data, str):
            # Base64 string
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        elif isinstance(image_data, bytes):
            # Direct bytes
            image = Image.open(io.BytesIO(image_data))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        else:
            raise ValueError("Unsupported image format")
            
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@router_liveness.post("/analyze", response_model=LivenessResponse, dependencies=[Depends(verify_api_key)])
async def analyze_liveness_upload(
    file: UploadFile = File(..., description="Image file to analyze"),
    threshold: Optional[float] = Form(0.5, description="Detection threshold")
):
    """
    Analyze uploaded image file for liveness detection
    
    Args:
        file: Image file (JPG, PNG, etc.)
        threshold: Detection threshold (0.0 to 1.0)
    
    Returns:
        LivenessResponse with analysis results
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and decode image
        image_bytes = await file.read()
        image = decode_image(image_bytes)
        
        # Perform liveness detection
        result = liveness_detector.detect_liveness(image, threshold)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return LivenessResponse(
            is_live=result['is_live'],
            confidence=result['confidence'],
            spoof_type=result.get('spoof_type'),
            analysis_details=result['details'],
            processing_time_ms=processing_time,
            timestamp=int(time.time() * 1000)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in liveness analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router_liveness.post("/analyze-base64", response_model=LivenessResponse, dependencies=[Depends(verify_api_key)])
async def analyze_liveness_base64(request: LivenessRequest):
    """
    Analyze base64 encoded image for liveness detection
    
    Args:
        request: LivenessRequest with base64 image and threshold
    
    Returns:
        LivenessResponse with analysis results
    """
    start_time = time.time()
    
    try:
        # Decode base64 image
        image = decode_image(request.image_base64)
        
        # Perform liveness detection
        result = liveness_detector.detect_liveness(image, request.threshold)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return LivenessResponse(
            is_live=result['is_live'],
            confidence=result['confidence'],
            spoof_type=result.get('spoof_type'),
            analysis_details=result['details'],
            processing_time_ms=processing_time,
            timestamp=int(time.time() * 1000)
        )
        
    except Exception as e:
        logger.error(f"Error in base64 liveness analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router_liveness.post("/batch-analyze", dependencies=[Depends(verify_api_key)])
async def batch_analyze_liveness(
    files: list[UploadFile] = File(..., description="Multiple image files to analyze"),
    threshold: Optional[float] = Form(0.5, description="Detection threshold")
):
    """
    Analyze multiple images for liveness detection
    
    Args:
        files: List of image files
        threshold: Detection threshold (0.0 to 1.0)
    
    Returns:
        List of analysis results
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            start_time = time.time()
            
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "Invalid file type",
                    "success": False
                })
                continue
            
            # Read and decode image
            image_bytes = await file.read()
            image = decode_image(image_bytes)
            
            # Perform liveness detection
            result = liveness_detector.detect_liveness(image, threshold)
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            results.append({
                "filename": file.filename,
                "is_live": result['is_live'],
                "confidence": result['confidence'],
                "spoof_type": result.get('spoof_type'),
                "analysis_details": result['details'],
                "processing_time_ms": processing_time,
                "success": True
            })
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {
        "results": results,
        "total_files": len(files),
        "successful_analyses": len([r for r in results if r.get('success', False)]),
        "timestamp": int(time.time() * 1000)
    }

@router_liveness.get("/health")
async def health_check():
    """Health check endpoint for liveness detection service"""
    try:
        # Test with a small dummy image
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = liveness_detector.detect_liveness(dummy_image, 0.5)
        
        return {
            "status": "healthy",
            "service": "liveness_detection",
            "detector_ready": True,
            "timestamp": int(time.time() * 1000)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "liveness_detection",
                "detector_ready": False,
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            }
        )

@router_liveness.get("/info")
async def get_detector_info():
    """Get information about the liveness detector"""
    return {
        "detector_version": liveness_detector.get_version(),
        "supported_formats": ["JPG", "PNG", "JPEG", "BMP", "TIFF"],
        "detection_methods": liveness_detector.get_detection_methods(),
        "default_threshold": 0.5,
        "max_batch_size": 10,
        "timestamp": int(time.time() * 1000)
    }

@router_liveness.post("/configure")
async def configure_detector(
    threshold: Optional[float] = Form(None, description="Default threshold"),
    sensitivity: Optional[str] = Form(None, description="Detection sensitivity: low, medium, high")
):
    """
    Configure liveness detector parameters
    
    Args:
        threshold: Default detection threshold (0.0 to 1.0)
        sensitivity: Detection sensitivity level
    
    Returns:
        Configuration status
    """
    try:
        config_updated = {}
        
        if threshold is not None:
            if not 0.0 <= threshold <= 1.0:
                raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
            liveness_detector.set_threshold(threshold)
            config_updated['threshold'] = threshold
        
        if sensitivity is not None:
            if sensitivity not in ['low', 'medium', 'high']:
                raise HTTPException(status_code=400, detail="Sensitivity must be 'low', 'medium', or 'high'")
            liveness_detector.set_sensitivity(sensitivity)
            config_updated['sensitivity'] = sensitivity
        
        return {
            "status": "configured",
            "updated_parameters": config_updated,
            "current_config": liveness_detector.get_config(),
            "timestamp": int(time.time() * 1000)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring detector: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

