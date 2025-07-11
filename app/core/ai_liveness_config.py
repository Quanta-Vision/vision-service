# ai_liveness_config.py - Configuration for AI models

import os
from typing import Dict, Any

class AIConfig:
    """Configuration for AI liveness detection"""
    
    # API Keys (set these in your environment variables or directly here)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") 
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAIeWJCX4YH_u5DbJgIrulPOhrwLwuWm3k")  # Your Gemini key
    GOOGLE_API_URL = os.getenv("GOOGLE_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent")
    
    # Model configurations
    OPENAI_MODEL = "gpt-4o"  # Updated model name
    ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"  # Updated model
    GOOGLE_MODEL = "gemini-2.0-flash"  # Your specific Gemini model
    OLLAMA_MODEL = "llava"
    OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    
    # Timeout settings
    AI_REQUEST_TIMEOUT = 30  # seconds
    
    # Confidence thresholds
    MIN_AI_CONFIDENCE = 0.6
    MIN_TRADITIONAL_CONFIDENCE = 0.5
    
    # Decision combination weights
    AI_WEIGHT = 0.6
    TRADITIONAL_WEIGHT = 0.4
    
    # Provider priority order (most reliable first)
    PROVIDER_PRIORITY = ["google_gemini", "anthropic_claude", "openai_gpt4v", "ollama_llava"]
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, bool]:
        """Check which providers are available"""
        return {
            "openai_gpt4v": bool(cls.OPENAI_API_KEY),
            "anthropic_claude": bool(cls.ANTHROPIC_API_KEY),
            "google_gemini": bool(cls.GOOGLE_API_KEY),
            "ollama_llava": True  # Assume available if endpoint configured
        }
    
    @classmethod
    def get_provider_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get detailed provider information"""
        return {
            "openai_gpt4v": {
                "name": "OpenAI GPT-4o Vision",
                "model": cls.OPENAI_MODEL,
                "available": bool(cls.OPENAI_API_KEY),
                "cost_per_image": "$0.005-0.015",
                "speed": "2-4 seconds",
                "accuracy": "Very High"
            },
            "anthropic_claude": {
                "name": "Anthropic Claude 3.5 Sonnet",
                "model": cls.ANTHROPIC_MODEL,
                "available": bool(cls.ANTHROPIC_API_KEY),
                "cost_per_image": "$0.008-0.020", 
                "speed": "2-4 seconds",
                "accuracy": "Very High"
            },
            "google_gemini": {
                "name": "Google Gemini 2.0 Flash",
                "model": cls.GOOGLE_MODEL,
                "available": bool(cls.GOOGLE_API_KEY),
                "cost_per_image": "$0.001-0.005",
                "speed": "1-2 seconds",
                "accuracy": "High",
                "notes": "Your configured model - fastest and cheapest"
            },
            "ollama_llava": {
                "name": "Ollama LLaVA (Local)",
                "model": cls.OLLAMA_MODEL,
                "available": True,
                "cost_per_image": "Free",
                "speed": "3-8 seconds",
                "accuracy": "Medium-High"
            }
        }

# Environment setup helper
def setup_environment():
    """Set up environment variables if not already set"""
    import os
    
    # Set your API keys
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = "AIzaSyAIeWJCX4YH_u5DbJgIrulPOhrwLwuWm3k"
    
    if not os.getenv("GOOGLE_API_URL"):
        os.environ["GOOGLE_API_URL"] = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    print("✅ Google Gemini configuration loaded")
    print(f"🔑 API Key: {os.getenv('GOOGLE_API_KEY')[:20]}...")
    print(f"🌐 API URL: {os.getenv('GOOGLE_API_URL')}")

# Call setup when module is imported
setup_environment()

# Usage examples with your specific configuration
USAGE_EXAMPLES = {
    "curl_gemini": """
# Test with your Gemini API
curl -X POST "http://localhost:8000/ai-spoof-detect/ai-analyze" \\
  -F "file=@image.jpg" \\
  -F "ai_provider=google_gemini" \\
  -F "combine_traditional=true" \\
  -F "threshold=0.5"
""",
    
    "python_gemini": """
import requests

# Test with Gemini (your default provider)
def test_gemini_liveness(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'ai_provider': 'google_gemini',  # Your configured provider
            'combine_traditional': True,
            'threshold': 0.5
        }
        
        response = requests.post(
            'http://localhost:8000/ai-spoof-detect/ai-analyze',
            files=files,
            data=data
        )
        
        result = response.json()
        return result

# Test the API
result = test_gemini_liveness("test_image.jpg")
print(f"Is Live: {result['is_live']}")
print(f"Confidence: {result['final_confidence']}")
print(f"Provider Used: {result.get('successful_provider', 'google_gemini')}")
"""
}

# Requirements for your setup
REQUIREMENTS_TXT = """
# Add these to your requirements.txt:

# Core AI packages
openai>=1.0.0
anthropic>=0.18.0
aiohttp>=3.8.0

# For Google API (if using google-generativeai package)
google-generativeai>=0.3.0

# Additional utilities
asyncio-throttle>=1.0.0
python-dotenv>=1.0.0  # For loading .env files
"""

print("🚀 AI Liveness Configuration Ready!")
print("📋 Next steps:")
print("1. Add the requirements to your requirements.txt")
print("2. Update your services/liveness_ai.py with the new Google API configuration") 
print("3. Test with: curl or Python client using 'google_gemini' provider")

# requirements_ai.txt - Additional packages needed for AI integration
"""
# Add these to your requirements.txt:

# OpenAI
openai>=1.0.0

# Anthropic Claude
anthropic>=0.18.0

# Google AI
google-generativeai>=0.3.0

# Additional utilities
aiohttp>=3.8.0
asyncio-throttle>=1.0.0
"""

# .env file template
"""
# Create a .env file in your project root with these variables:

# OpenAI API Key (for GPT-4 Vision)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (for Claude Vision)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google API Key (for Gemini Pro Vision)
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Ollama endpoint (if using local models)
OLLAMA_ENDPOINT=http://localhost:11434
"""

# Updated main.py to include the new AI router
"""
# Add this to your main.py:

from app.api.routes_ai_liveness import router_ai_liveness

# Include the AI liveness router
app.include_router(router_ai_liveness, prefix="/ai-spoof-detect")
"""

# Docker setup for Ollama (optional local model)
OLLAMA_DOCKER_COMPOSE = """
# docker-compose.yml for local Ollama setup
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    command: serve

volumes:
  ollama_data:

# After starting: docker exec -it ollama_container ollama pull llava
"""

# Example usage
USAGE_EXAMPLES = {
    "curl_file_upload": """
# Upload file with AI analysis
curl -X POST "http://localhost:8000/ai-spoof-detect/ai-analyze" \\
  -F "file=@image.jpg" \\
  -F "ai_provider=openai_gpt4v" \\
  -F "combine_traditional=true" \\
  -F "threshold=0.5"
""",
    
    "curl_base64": """
# Base64 analysis
curl -X POST "http://localhost:8000/ai-spoof-detect/ai-analyze-base64" \\
  -H "Content-Type: application/json" \\
  -d '{
    "image_base64": "base64_encoded_image_here",
    "ai_provider": "anthropic_claude",
    "combine_traditional": true,
    "threshold": 0.5
  }'
""",
    
    "python_client": """
import requests
import base64

# Example Python client
def analyze_image_with_ai(image_path, ai_provider="openai_gpt4v"):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'ai_provider': ai_provider,
            'combine_traditional': True,
            'threshold': 0.5
        }
        
        response = requests.post(
            'http://localhost:8000/ai-spoof-detect/ai-analyze',
            files=files,
            data=data
        )
        
        return response.json()

# Usage
result = analyze_image_with_ai("test_image.jpg", "openai_gpt4v")
print(f"Is Live: {result['is_live']}")
print(f"AI Decision: {result['ai_decision']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['ai_reasoning']}")
"""
}

# Cost considerations
COST_INFO = {
    "openai_gpt4v": {
        "cost_per_image": "$0.01-0.02",
        "speed": "2-5 seconds",
        "accuracy": "Very High",
        "notes": "Best overall performance, detailed reasoning"
    },
    "anthropic_claude": {
        "cost_per_image": "$0.008-0.015", 
        "speed": "2-4 seconds",
        "accuracy": "High",
        "notes": "Good balance of cost and performance"
    },
    "google_gemini": {
        "cost_per_image": "$0.002-0.008",
        "speed": "1-3 seconds", 
        "accuracy": "High",
        "notes": "Most cost-effective cloud option"
    },
    "ollama_llava": {
        "cost_per_image": "Free",
        "speed": "3-8 seconds",
        "accuracy": "Medium-High",
        "notes": "Local model, no API costs, requires GPU"
    }
}

# Performance optimization tips
OPTIMIZATION_TIPS = """
1. Image Preprocessing:
   - Resize images to max 1024px for faster processing
   - Use JPEG compression to reduce payload size
   - Consider image quality vs speed tradeoffs

2. Caching:
   - Cache AI results for identical images
   - Use Redis or similar for result caching
   - Implement hash-based deduplication

3. Async Processing:
   - Use async/await for concurrent requests
   - Implement request queuing for high load
   - Consider rate limiting for API costs

4. Fallback Strategy:
   - Primary: Best AI model (GPT-4V/Claude)
   - Secondary: Faster AI model (Gemini)
   - Fallback: Traditional detection only
   - Emergency: Simple rule-based detection

5. Cost Management:
   - Monitor API usage and costs
   - Implement usage quotas
   - Use local models for development/testing
   - Batch similar requests when possible
"""