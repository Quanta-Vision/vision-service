# Setup Instructions for AI Liveness Detection

# 1. INSTALL REQUIRED PACKAGES
"""
Add these to your requirements.txt:

openai>=1.0.0
anthropic>=0.18.0
aiohttp>=3.8.0
"""

# 2. UPDATE YOUR MAIN.PY
"""
Add this import and router to your main.py:

from app.api.routes_ai_liveness import router_ai_liveness

# Add this line with your other router inclusions:
app.include_router(router_ai_liveness, prefix="/ai-spoof-detect")
"""

# 3. ENVIRONMENT VARIABLES
"""
Create a .env file or set these environment variables:

# OpenAI API Key (for GPT-4 Vision)
OPENAI_API_KEY=sk-your_openai_api_key_here

# Anthropic API Key (for Claude Vision) 
ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key_here

# Google API Key (for Gemini Pro Vision)
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Ollama endpoint (for local models)
OLLAMA_ENDPOINT=http://localhost:11434
"""

# 4. COMPLETE MAIN.PY EXAMPLE
MAIN_PY_EXAMPLE = """
import time
from fastapi.responses import JSONResponse
import uvicorn
from app.api.routes_iam import router_iam
from app.api.routes import router
from app.api.routes_recognite import router_v2
from app.api.routes_counter import router_counter
from app.api.routes_liveness import router_liveness
from app.api.routes_ai_liveness import router_ai_liveness  # NEW: AI Liveness Detection
from fastapi import FastAPI, Request
from app.core.config import PORT
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

os.makedirs("images", exist_ok=True)

app = FastAPI(title="Vision API")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(router)
app.include_router(router_iam, prefix="/iam")
app.include_router(router_v2, prefix="/v2")
app.include_router(router_counter, prefix="/counter")
app.include_router(router_liveness, prefix="/spoof-detect")
app.include_router(router_ai_liveness, prefix="/ai-spoof-detect")  # NEW: AI Router

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": int(time.time() * 1000)
        }
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)
"""

# 5. API TESTING EXAMPLES

# Test with curl
CURL_EXAMPLES = {
    "upload_file": """
# Test with file upload
curl -X POST "http://localhost:8000/ai-spoof-detect/ai-analyze" \\
  -F "file=@test_image.jpg" \\
  -F "ai_provider=openai_gpt4v" \\
  -F "combine_traditional=true" \\
  -F "threshold=0.5"
""",
    
    "base64_test": """
# Test with base64 (replace with actual base64 string)
curl -X POST "http://localhost:8000/ai-spoof-detect/ai-analyze-base64" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_base64": "your_base64_image_here",
    "ai_provider": "google_gemini",
    "combine_traditional": true,
    "threshold": 0.5
  }'
""",
    
    "check_providers": """
# Check available AI providers
curl -X GET "http://localhost:8000/ai-spoof-detect/ai-providers"
""",
    
    "health_check": """
# Check service health
curl -X GET "http://localhost:8000/ai-spoof-detect/health"
""",
    
    "test_provider": """
# Test specific AI provider
curl -X POST "http://localhost:8000/ai-spoof-detect/test-ai-provider" \\
  -F "ai_provider=anthropic_claude"
"""
}

# Python client example
PYTHON_CLIENT_EXAMPLE = """
import requests
import base64
import json

class AILivenessClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.ai_endpoint = f"{base_url}/ai-spoof-detect"
    
    def analyze_image_file(self, image_path, ai_provider="openai_gpt4v", combine_traditional=True):
        \"\"\"Analyze image file with AI detection\"\"\"
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'ai_provider': ai_provider,
                    'combine_traditional': combine_traditional,
                    'threshold': 0.5
                }
                
                response = requests.post(
                    f"{self.ai_endpoint}/ai-analyze",
                    files=files,
                    data=data
                )
                
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_base64_image(self, image_base64, ai_provider="google_gemini", combine_traditional=True):
        \"\"\"Analyze base64 image with AI detection\"\"\"
        try:
            payload = {
                'image_base64': image_base64,
                'ai_provider': ai_provider,
                'combine_traditional': combine_traditional,
                'threshold': 0.5
            }
            
            response = requests.post(
                f"{self.ai_endpoint}/ai-analyze-base64",
                json=payload
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_available_providers(self):
        \"\"\"Get list of available AI providers\"\"\"
        try:
            response = requests.get(f"{self.ai_endpoint}/ai-providers")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'error': str(e)}
    
    def check_health(self):
        \"\"\"Check service health\"\"\"
        try:
            response = requests.get(f"{self.ai_endpoint}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'error': str(e)}

# Usage example
if __name__ == "__main__":
    client = AILivenessClient()
    
    # Check health first
    health = client.check_health()
    print("Health:", health)
    
    # Get available providers
    providers = client.get_available_providers()
    print("Available providers:", providers)
    
    # Analyze an image (replace with actual image path)
    result = client.analyze_image_file(
        "test_image.jpg", 
        ai_provider="openai_gpt4v",
        combine_traditional=True
    )
    
    if 'error' not in result:
        print(f"Is Live: {result['is_live']}")
        print(f"AI Decision: {result['ai_decision']}")
        print(f"Final Confidence: {result['final_confidence']:.2f}")
        print(f"Decision Method: {result['decision_method']}")
        print(f"AI Reasoning: {result['ai_reasoning']}")
        if result['traditional_decision'] is not None:
            print(f"Traditional Decision: {result['traditional_decision']}")
    else:
        print(f"Error: {result['error']}")
"""

# 6. LOCAL OLLAMA SETUP (Optional)
OLLAMA_SETUP = """
# Install Ollama (for local AI model)
# Visit: https://ollama.ai/download

# Or use Docker:
docker run -d -p 11434:11434 --name ollama ollama/ollama

# Install LLaVA model
ollama pull llava

# Test if working
curl http://localhost:11434/api/generate \\
  -d '{
    "model": "llava",
    "prompt": "Describe this image",
    "images": ["base64_image_here"],
    "stream": false
  }'
"""

# 7. COST OPTIMIZATION TIPS
COST_OPTIMIZATION = """
# Cost Optimization Strategies:

1. Provider Selection by Use Case:
   - Development/Testing: Use Ollama LLaVA (free)
   - High Accuracy Needed: OpenAI GPT-4V
   - Balanced Cost/Performance: Anthropic Claude
   - High Volume: Google Gemini

2. Batch Processing:
   - Use Google Gemini for batch operations (cheaper)
   - Process multiple images in single requests when possible

3. Caching:
   - Implement Redis caching for identical images
   - Cache results for development/testing

4. Fallback Strategy:
   - Primary: Preferred AI model
   - Fallback: Cheaper AI model
   - Emergency: Traditional detection only

5. Smart Provider Selection:
   - Use cheaper models for obvious cases
   - Use premium models for borderline cases
   - Implement confidence-based routing
"""

# 8. TROUBLESHOOTING
TROUBLESHOOTING = """
# Common Issues and Solutions:

1. "OpenAI API not configured"
   - Set OPENAI_API_KEY environment variable
   - Verify API key is valid and has credits

2. "Anthropic API not configured"
   - Set ANTHROPIC_API_KEY environment variable
   - Ensure API key format: sk-ant-...

3. "Google API error"
   - Set GOOGLE_API_KEY environment variable
   - Enable Vertex AI API in Google Cloud Console

4. "Ollama not available"
   - Start Ollama service: ollama serve
   - Pull LLaVA model: ollama pull llava
   - Check endpoint: http://localhost:11434

5. "Traditional detection failed"
   - Check import path: app.services.enhance_liveness_detection
   - Ensure services/__init__.py exists
   - Verify LivenessDetector class works

6. Import errors:
   - Install missing packages: pip install openai anthropic aiohttp
   - Check Python path configuration
   - Ensure all files are in correct directories
"""

print("=== AI Liveness Detection Setup Complete ===")
print("1. Install packages from requirements.txt")
print("2. Set environment variables for AI providers")
print("3. Update main.py with AI router")
print("4. Test endpoints using curl or Python client")
print("5. Check troubleshooting section if issues occur")