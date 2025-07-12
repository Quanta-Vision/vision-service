# ai_liveness_config.py - Configuration for AI models

import os
from typing import Dict, Any

class AIConfig:
    """Configuration for AI liveness detection"""
    
    # API Keys (set these in your environment variables)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") 
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model configurations
    OPENAI_MODEL = "gpt-4-vision-preview"
    ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    GOOGLE_MODEL = "gemini-pro-vision"
    OLLAMA_MODEL = "llava"
    
    # Timeout settings
    AI_REQUEST_TIMEOUT = 30  # seconds
    
    # Confidence thresholds
    MIN_AI_CONFIDENCE = 0.6
    MIN_TRADITIONAL_CONFIDENCE = 0.5
    
    # Decision combination weights
    AI_WEIGHT = 0.6
    TRADITIONAL_WEIGHT = 0.4

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