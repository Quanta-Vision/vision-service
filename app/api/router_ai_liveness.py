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

# Import from your services directory
from app.services.enhance_liveness_detection import LivenessDetector
from app.services.liveness_ai import ai_liveness_service, ai_combiner
from app.utils.auth import verify_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router_ai_liveness = APIRouter(tags=["AI Liveness Detection"])

# Initialize the traditional liveness detector
liveness_detector = LivenessDetector()

# Pydantic models for request/response
class AILivenessRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    ai_provider: str = Field("openai_gpt4v", description="AI provider to use")
    combine_traditional: bool = Field(True, description="Combine with traditional detection methods")
    threshold: Optional[float] = Field(0.5, description="Detection threshold for traditional methods")

class AILivenessResponse(BaseModel):
    is_live: bool = Field(..., description="Final decision: True if real, False if spoof")
    final_confidence: float = Field(..., description="Overall confidence score")
    decision_method: str = Field(..., description="How final decision was made")
    ai_decision: str = Field(..., description="AI model decision ('Real' or 'Spoof')")
    ai_confidence: float = Field(..., description="AI model confidence")
    ai_reasoning: Optional[str] = Field(None, description="AI model reasoning")
    traditional_decision: Optional[bool] = Field(None, description="Traditional algorithm decision")
    traditional_score: Optional[float] = Field(None, description="Traditional algorithm score")
    processing_time_ms: int = Field(..., description="Total processing time")
    timestamp: int = Field(..., description="Unix timestamp")

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

@router_ai_liveness.post("/ai-analyze", response_model=AILivenessResponse, dependencies=[Depends(verify_api_key)])
async def analyze_with_ai(
    file: UploadFile = File(..., description="Image file to analyze"),
    ai_provider: str = Form("openai_gpt4v", description="AI provider to use"),
    combine_traditional: bool = Form(True, description="Combine with traditional detection"),
    threshold: float = Form(0.5, description="Traditional detection threshold")
):
    """
    Analyze uploaded image file using AI models with optional traditional method combination
    
    Available AI providers:
    - openai_gpt4v: OpenAI GPT-4 Vision (best accuracy)
    - anthropic_claude: Anthropic Claude Vision (balanced)
    - google_gemini: Google Gemini Pro Vision (cost-effective)
    - ollama_llava: Local LLaVA model (free)
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and encode image
        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Traditional detection (if requested)
        traditional_result = None
        if combine_traditional:
            try:
                image_array = decode_image(image_bytes)
                traditional_result = liveness_detector.detect_liveness(image_array, threshold)
                logger.info(f"Traditional detection: {traditional_result['is_live']} (score: {traditional_result['liveness_score']:.3f})")
            except Exception as e:
                logger.warning(f"Traditional detection failed: {str(e)}")
                # Continue with AI-only analysis
        
        # AI model analysis
        logger.info(f"Starting AI analysis with provider: {ai_provider}")
        ai_result = await ai_liveness_service.analyze_image(image_base64, ai_provider)
        logger.info(f"AI decision: {ai_result['decision']} (confidence: {ai_result['confidence']:.3f})")
        
        # Combine results
        combined_result = ai_combiner.combine_results(ai_result, traditional_result)
        
        # Calculate total processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return AILivenessResponse(
            is_live=combined_result["is_live"],
            final_confidence=combined_result["final_confidence"],
            decision_method=combined_result["decision_method"],
            ai_decision=combined_result["ai_decision"],
            ai_confidence=combined_result["ai_confidence"],
            ai_reasoning=combined_result["ai_reasoning"],
            traditional_decision=combined_result["traditional_decision"],
            traditional_score=combined_result["traditional_score"],
            processing_time_ms=processing_time,
            timestamp=int(time.time() * 1000)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router_ai_liveness.post("/ai-analyze-base64", response_model=AILivenessResponse, dependencies=[Depends(verify_api_key)])
async def analyze_ai_base64(request: AILivenessRequest):
    """
    Analyze base64 encoded image using AI models
    """
    start_time = time.time()
    
    try:
        # Traditional detection (if requested)
        traditional_result = None
        if request.combine_traditional:
            try:
                image_array = decode_image(request.image_base64)
                traditional_result = liveness_detector.detect_liveness(image_array, request.threshold)
                logger.info(f"Traditional detection: {traditional_result['is_live']} (score: {traditional_result['liveness_score']:.3f})")
            except Exception as e:
                logger.warning(f"Traditional detection failed: {str(e)}")
        
        # AI model analysis
        logger.info(f"Starting AI analysis with provider: {request.ai_provider}")
        ai_result = await ai_liveness_service.analyze_image(request.image_base64, request.ai_provider)
        logger.info(f"AI decision: {ai_result['decision']} (confidence: {ai_result['confidence']:.3f})")
        
        # Combine results
        combined_result = ai_combiner.combine_results(ai_result, traditional_result)
        
        # Calculate total processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return AILivenessResponse(
            is_live=combined_result["is_live"],
            final_confidence=combined_result["final_confidence"],
            decision_method=combined_result["decision_method"],
            ai_decision=combined_result["ai_decision"],
            ai_confidence=combined_result["ai_confidence"],
            ai_reasoning=combined_result["ai_reasoning"],
            traditional_decision=combined_result["traditional_decision"],
            traditional_score=combined_result["traditional_score"],
            processing_time_ms=processing_time,
            timestamp=int(time.time() * 1000)
        )
        
    except Exception as e:
        logger.error(f"AI base64 analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router_ai_liveness.get("/ai-providers")
async def get_available_providers():
    """Get list of available AI providers and their status"""
    try:
        providers = ai_liveness_service.get_available_providers()
        return {
            "providers": providers,
            "service_info": ai_liveness_service.get_service_info(),
            "timestamp": int(time.time() * 1000)
        }
    except Exception as e:
        logger.error(f"Error getting providers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")

@router_ai_liveness.get("/health")
async def ai_health_check():
    """Health check for AI liveness detection service"""
    try:
        service_info = ai_liveness_service.get_service_info()
        providers = ai_liveness_service.get_available_providers()
        
        available_count = len([p for p in providers.values() if p["available"]])
        
        return {
            "status": "healthy" if service_info["ready"] else "degraded",
            "service": "ai_liveness_detection",
            "available_providers": available_count,
            "total_providers": len(providers),
            "traditional_detector_ready": True,
            "details": service_info,
            "timestamp": int(time.time() * 1000)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "ai_liveness_detection",
                "error": str(e),
                "timestamp": int(time.time() * 1000)
            }
        )

@router_ai_liveness.post("/test-ai-provider")
async def test_ai_provider(
    ai_provider: str = Form(..., description="AI provider to test"),
    test_image: UploadFile = File(None, description="Optional test image (uses default if not provided)")
):
    """
    Test a specific AI provider with a sample image
    """
    try:
        # Use provided image or create a test image
        if test_image and test_image.content_type.startswith('image/'):
            image_bytes = await test_image.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        else:
            # Create a simple test image
            test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', test_img)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Test the provider
        start_time = time.time()
        result = await ai_liveness_service.analyze_image(image_base64, ai_provider)
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "provider": ai_provider,
            "test_successful": True,
            "result": result,
            "processing_time_ms": processing_time,
            "timestamp": int(time.time() * 1000)
        }
        
    except Exception as e:
        logger.error(f"Provider test failed for {ai_provider}: {str(e)}")
        return {
            "provider": ai_provider,
            "test_successful": False,
            "error": str(e),
            "timestamp": int(time.time() * 1000)
        }

@router_ai_liveness.get("/info")
async def get_ai_info():
    """Get information about the AI liveness detection service"""
    try:
        service_info = ai_liveness_service.get_service_info()
        providers = ai_liveness_service.get_available_providers()
        
        return {
            "service_info": service_info,
            "providers": providers,
            "endpoints": [
                "/ai-analyze - Upload file for AI analysis",
                "/ai-analyze-base64 - Base64 image analysis", 
                "/ai-providers - List available providers",
                "/test-ai-provider - Test specific provider",
                "/health - Service health check"
            ],
            "example_usage": {
                "curl": """curl -X POST "http://localhost:8000/ai-spoof-detect/ai-analyze" \\
  -F "file=@image.jpg" \\
  -F "ai_provider=openai_gpt4v" \\
  -F "combine_traditional=true" \\
  -F "threshold=0.5" """,
                "python": """
import requests

files = {'file': open('image.jpg', 'rb')}
data = {
    'ai_provider': 'openai_gpt4v',
    'combine_traditional': True,
    'threshold': 0.5
}

response = requests.post(
    'http://localhost:8000/ai-spoof-detect/ai-analyze',
    files=files,
    data=data
)

result = response.json()
print(f"Is Live: {result['is_live']}")
print(f"AI Decision: {result['ai_decision']}")
print(f"Confidence: {result['final_confidence']}")
"""
            },
            "timestamp": int(time.time() * 1000)
        }
        
    except Exception as e:
        logger.error(f"Error getting AI info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get info: {str(e)}")

# Batch processing endpoint
@router_ai_liveness.post("/batch-ai-analyze")
async def batch_analyze_ai(
    files: list[UploadFile] = File(..., description="Multiple image files to analyze"),
    ai_provider: str = Form("google_gemini", description="AI provider for batch processing"),
    combine_traditional: bool = Form(False, description="Combine with traditional detection"),
    max_batch_size: int = Form(5, description="Maximum batch size")
):
    """
    Batch process multiple images with AI analysis
    Note: Uses Google Gemini by default for cost-effectiveness in batch processing
    """
    if len(files) > max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size cannot exceed {max_batch_size}"
        )
    
    results = []
    total_start_time = time.time()
    
    for i, file in enumerate(files):
        file_start_time = time.time()
        
        try:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid file type",
                    "file_index": i
                })
                continue
            
            # Read and encode image
            image_bytes = await file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Traditional detection (if requested)
            traditional_result = None
            if combine_traditional:
                try:
                    image_array = decode_image(image_bytes)
                    traditional_result = liveness_detector.detect_liveness(image_array, 0.5)
                except Exception as e:
                    logger.warning(f"Traditional detection failed for {file.filename}: {str(e)}")
            
            # AI analysis
            ai_result = await ai_liveness_service.analyze_image(image_base64, ai_provider)
            
            # Combine results
            combined_result = ai_combiner.combine_results(ai_result, traditional_result)
            
            # Calculate processing time for this file
            file_processing_time = int((time.time() - file_start_time) * 1000)
            
            results.append({
                "filename": file.filename,
                "file_index": i,
                "success": True,
                "is_live": combined_result["is_live"],
                "final_confidence": combined_result["final_confidence"],
                "decision_method": combined_result["decision_method"],
                "ai_decision": combined_result["ai_decision"],
                "ai_confidence": combined_result["ai_confidence"],
                "ai_reasoning": combined_result["ai_reasoning"],
                "traditional_decision": combined_result["traditional_decision"],
                "processing_time_ms": file_processing_time
            })
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "file_index": i,
                "success": False,
                "error": str(e),
                "processing_time_ms": int((time.time() - file_start_time) * 1000)
            })
    
    total_processing_time = int((time.time() - total_start_time) * 1000)
    successful_analyses = len([r for r in results if r.get('success', False)])
    
    return {
        "batch_results": results,
        "summary": {
            "total_files": len(files),
            "successful_analyses": successful_analyses,
            "failed_analyses": len(files) - successful_analyses,
            "ai_provider_used": ai_provider,
            "traditional_detection_used": combine_traditional,
            "total_processing_time_ms": total_processing_time
        },
        "timestamp": int(time.time() * 1000)
    }
