import asyncio
import base64
import json
import logging
import os
import time
from typing import Dict, Any, Optional
import aiohttp
import openai
import anthropic
import requests

logger = logging.getLogger(__name__)

class AILivenessService:
    """AI-powered liveness detection service"""
    
    def __init__(self):
        self.setup_clients()
        self.prompt_template = self._get_detection_prompt()
    
    def setup_clients(self):
        """Initialize AI model clients"""
        try:
            # OpenAI GPT-4V
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if self.openai_api_key:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI GPT-4V client initialized")
            else:
                self.openai_client = None
                logger.warning("OpenAI API key not found")
            
            # Anthropic Claude
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if self.anthropic_api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                logger.info("Anthropic Claude client initialized")
            else:
                self.anthropic_client = None
                logger.warning("Anthropic API key not found")
            
            # Google Gemini
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            if self.google_api_key:
                logger.info("Google Gemini API key found")
            else:
                logger.warning("Google API key not found")
            
            # Ollama (local)
            self.ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
            
        except Exception as e:
            logger.error(f"Error setting up AI clients: {str(e)}")
    
    def _get_detection_prompt(self) -> str:
        """Get the optimized prompt for liveness detection"""
        return """Analyze this image to determine if it's a 'Real' photo taken directly with a camera or a 'Spoof' (recaptured from a screen/display or printed photo).

Key indicators to look for:

SPOOF indicators:
- Screen artifacts: pixel grids, moiré patterns, RGB subpixels
- Display characteristics: uniform backlighting, color banding, refresh lines
- Print artifacts: dot matrix patterns, paper texture, halftone dots
- Digital processing: over-sharpening, compression artifacts, unnatural smoothing
- Lighting uniformity: artificial, too-perfect lighting
- Resolution inconsistencies: upscaling artifacts, pixelation

REAL indicators:
- Natural skin texture and pores
- Proper lighting gradients and shadows
- Camera sensor noise patterns
- Natural color variations
- Sharp, authentic edges
- Depth of field effects

Respond with ONLY this JSON format:
{
    "decision": "Real" or "Spoof",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of key indicators"
}"""
    
    async def analyze_with_openai_gpt4v(self, image_base64: str) -> Dict[str, Any]:
        """Analyze image using OpenAI GPT-4V"""
        if not self.openai_client:
            raise Exception("OpenAI API not configured")
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4o",  # Updated to current model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.prompt_template},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300,
                    temperature=0.1
                )
            )
            
            result_text = response.choices[0].message.content
            return self._parse_ai_response(result_text, "openai_gpt4v")
            
        except Exception as e:
            logger.error(f"OpenAI GPT-4V analysis failed: {str(e)}")
            raise Exception(f"OpenAI analysis failed: {str(e)}")
    
    async def analyze_with_anthropic_claude(self, image_base64: str) -> Dict[str, Any]:
        """Analyze image using Anthropic Claude"""
        if not self.anthropic_client:
            raise Exception("Anthropic API not configured")
        
        try:
            message = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",  # Updated to latest model
                    max_tokens=200,
                    temperature=0.1,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_base64
                                    }
                                },
                                {"type": "text", "text": self.prompt_template}
                            ]
                        }
                    ]
                )
            )
            
            result_text = message.content[0].text
            return self._parse_ai_response(result_text, "anthropic_claude")
            
        except Exception as e:
            logger.error(f"Anthropic Claude analysis failed: {str(e)}")
            raise Exception(f"Claude analysis failed: {str(e)}")
    
    async def analyze_with_google_gemini(self, image_base64: str) -> Dict[str, Any]:
        """Analyze image using Google Gemini"""
        if not self.google_api_key:
            raise Exception("Google API not configured")
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key={self.google_api_key}"
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": self.prompt_template},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 200
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result["candidates"][0]["content"]["parts"][0]["text"]
                        return self._parse_ai_response(text, "google_gemini")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Gemini API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Google Gemini analysis failed: {str(e)}")
            raise Exception(f"Gemini analysis failed: {str(e)}")
    
    async def analyze_with_ollama_llava(self, image_base64: str) -> Dict[str, Any]:
        """Analyze image using local Ollama LLaVA model"""
        try:
            url = f"{self.ollama_endpoint}/api/generate"
            
            payload = {
                "model": "llava",
                "prompt": self.prompt_template,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result.get("response", "")
                        return self._parse_ai_response(text, "ollama_llava")
                    else:
                        raise Exception(f"Ollama not available: {response.status}")
                        
        except Exception as e:
            logger.error(f"Ollama LLaVA analysis failed: {str(e)}")
            raise Exception(f"Local model analysis failed: {str(e)}")
    
    def _parse_ai_response(self, response_text: str, provider: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Try to parse JSON response
            if "{" in response_text and "}" in response_text:
                # Extract JSON part
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_text = response_text[start:end]
                
                try:
                    result = json.loads(json_text)
                    return {
                        "decision": result.get("decision", "Unknown"),
                        "confidence": float(result.get("confidence", 0.5)),
                        "reasoning": result.get("reasoning", "No reasoning provided"),
                        "provider": provider,
                        "raw_response": response_text
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback parsing for non-JSON responses
            text_lower = response_text.lower()
            
            # Determine decision
            if "real" in text_lower and "spoof" not in text_lower:
                decision = "Real"
                confidence = 0.7
            elif "spoof" in text_lower and "real" not in text_lower:
                decision = "Spoof"
                confidence = 0.7
            elif "real" in text_lower and "spoof" in text_lower:
                # Both mentioned, look for stronger indicators
                if text_lower.count("spoof") > text_lower.count("real"):
                    decision = "Spoof"
                    confidence = 0.6
                else:
                    decision = "Real"
                    confidence = 0.6
            else:
                decision = "Unknown"
                confidence = 0.5
            
            return {
                "decision": decision,
                "confidence": confidence,
                "reasoning": response_text[:200],
                "provider": provider,
                "raw_response": response_text
            }
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return {
                "decision": "Unknown",
                "confidence": 0.5,
                "reasoning": f"Error parsing response: {str(e)}",
                "provider": provider,
                "raw_response": response_text
            }
    
    async def analyze_image(self, image_base64: str, provider: str = "openai_gpt4v") -> Dict[str, Any]:
        """
        Main method to analyze image with specified AI provider
        
        Args:
            image_base64: Base64 encoded image
            provider: AI provider to use
            
        Returns:
            Analysis result dictionary
        """
        start_time = time.time()
        
        try:
            if provider == "openai_gpt4v":
                result = await self.analyze_with_openai_gpt4v(image_base64)
            elif provider == "anthropic_claude":
                result = await self.analyze_with_anthropic_claude(image_base64)
            elif provider == "google_gemini":
                result = await self.analyze_with_google_gemini(image_base64)
            elif provider == "ollama_llava":
                result = await self.analyze_with_ollama_llava(image_base64)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            # Add timing information
            processing_time = int((time.time() - start_time) * 1000)
            result["processing_time_ms"] = processing_time
            result["timestamp"] = int(time.time() * 1000)
            
            return result
            
        except Exception as e:
            logger.error(f"AI analysis failed with provider {provider}: {str(e)}")
            raise
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available AI providers and their status"""
        providers = {
            "openai_gpt4v": {
                "name": "OpenAI GPT-4o (Vision)",
                "model": "gpt-4o",
                "available": self.openai_client is not None,
                "description": "Latest OpenAI vision model with improved capabilities",
                "cost_estimate": "$0.0025-0.01 per image",
                "speed": "2-4 seconds"
            },
            "anthropic_claude": {
                "name": "Anthropic Claude 3.5 Sonnet",
                "model": "claude-3-5-sonnet-20241022",
                "available": self.anthropic_client is not None,
                "description": "Latest Claude with enhanced vision capabilities",
                "cost_estimate": "$0.003-0.012 per image", 
                "speed": "2-4 seconds"
            },
            "google_gemini": {
                "name": "Google Gemini Pro Vision",
                "model": "gemini-pro-vision",
                "available": self.google_api_key is not None,
                "description": "Google's multimodal AI model",
                "cost_estimate": "$0.001-0.005 per image",
                "speed": "1-3 seconds"
            },
            "ollama_llava": {
                "name": "Ollama LLaVA (Local)",
                "model": "llava",
                "available": True,  # Assume available if endpoint configured
                "description": "Local vision language model",
                "cost_estimate": "Free (local)",
                "speed": "3-8 seconds"
            }
        }
        
        return providers
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        available_providers = [
            name for name, info in self.get_available_providers().items()
            if info["available"]
        ]
        
        return {
            "service": "AI Liveness Detection",
            "version": "1.0.0",
            "available_providers": available_providers,
            "total_providers": len(self.get_available_providers()),
            "ready": len(available_providers) > 0
        }

class AILivenessCombiner:
    """Service to combine AI and traditional detection results"""
    
    @staticmethod
    def combine_results(ai_result: Dict[str, Any], traditional_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Combine AI and traditional detection results
        
        Args:
            ai_result: Result from AI analysis
            traditional_result: Result from traditional detection (optional)
            
        Returns:
            Combined result with final decision
        """
        ai_is_live = ai_result["decision"].lower() == "real"
        ai_confidence = ai_result["confidence"]
        
        if traditional_result is None:
            # AI only
            return {
                "is_live": ai_is_live,
                "final_confidence": ai_confidence,
                "decision_method": "AI_ONLY",
                "ai_decision": ai_result["decision"],
                "ai_confidence": ai_confidence,
                "ai_reasoning": ai_result.get("reasoning", ""),
                "traditional_decision": None,
                "traditional_score": None
            }
        
        traditional_is_live = traditional_result["is_live"]
        traditional_confidence = traditional_result["confidence"]
        
        # Combine decisions
        if ai_is_live == traditional_is_live:
            # Both agree - high confidence
            combined_confidence = (ai_confidence + traditional_confidence) / 2
            final_decision = ai_is_live
            decision_method = "CONSENSUS"
        else:
            # Disagreement - use more confident result
            if ai_confidence > traditional_confidence:
                final_decision = ai_is_live
                combined_confidence = ai_confidence * 0.9  # Slight penalty for disagreement
                decision_method = "AI_PREFERRED"
            else:
                final_decision = traditional_is_live
                combined_confidence = traditional_confidence * 0.9
                decision_method = "TRADITIONAL_PREFERRED"
        
        return {
            "is_live": final_decision,
            "final_confidence": combined_confidence,
            "decision_method": decision_method,
            "ai_decision": ai_result["decision"],
            "ai_confidence": ai_confidence,
            "ai_reasoning": ai_result.get("reasoning", ""),
            "traditional_decision": traditional_is_live,
            "traditional_score": traditional_result["liveness_score"]
        }

# Global service instance
ai_liveness_service = AILivenessService()
ai_combiner = AILivenessCombiner()