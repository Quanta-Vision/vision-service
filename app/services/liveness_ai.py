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

# Import your configuration
try:
    from app.core.ai_liveness_config import AIConfig
except ImportError:
    # Fallback configuration if config file not found
    class AIConfig:
        GOOGLE_API_KEY = "AIzaSyAIeWJCX4YH_u5DbJgIrulPOhrwLwuWm3k"
        GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        OLLAMA_ENDPOINT = "http://localhost:11434"

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
            self.openai_api_key = AIConfig.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            if self.openai_api_key:
                self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
                logger.info("✅ OpenAI GPT-4o client initialized")
            else:
                self.openai_client = None
                logger.warning("⚠️ OpenAI API key not found")
            
            # Anthropic Claude
            self.anthropic_api_key = AIConfig.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
            if self.anthropic_api_key:
                self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                logger.info("✅ Anthropic Claude client initialized")
            else:
                self.anthropic_client = None
                logger.warning("⚠️ Anthropic API key not found")
            
            # Google Gemini (your specific configuration)
            self.google_api_key = AIConfig.GOOGLE_API_KEY
            self.google_api_url = AIConfig.GOOGLE_API_URL
            if self.google_api_key:
                logger.info("✅ Google Gemini 2.0 Flash configured")
                logger.info(f"🔑 API Key: {self.google_api_key[:20]}...")
                logger.info(f"🌐 API URL: {self.google_api_url}")
            else:
                logger.warning("⚠️ Google API key not found")
            
            # Ollama (local)
            self.ollama_endpoint = getattr(AIConfig, 'OLLAMA_ENDPOINT', "http://localhost:11434")
            logger.info(f"🏠 Ollama endpoint: {self.ollama_endpoint}")
            
        except Exception as e:
            logger.error(f"Error setting up AI clients: {str(e)}")
            # Set defaults
            self.google_api_key = "AIzaSyAIeWJCX4YH_u5DbJgIrulPOhrwLwuWm3k"
            self.google_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            logger.info("📍 Using fallback Google Gemini configuration")
    
    def _get_detection_prompt(self) -> str:
        """Get the optimized prompt for liveness detection"""
        return """You are a computer vision expert analyzing an image for technical authentication purposes. This is a legitimate security application for preventing fraud and ensuring genuine user verification.

TASK: Determine if this image is:
- 'Real': Captured directly with a camera/phone (authentic)
- 'Spoof': Recaptured from a screen, monitor, or printed photo (fraudulent)

TECHNICAL ANALYSIS REQUIRED:
Look for these technical indicators:

SPOOF DETECTION SIGNS:
- Display artifacts: RGB subpixel patterns, screen door effect
- Moiré patterns from screen interference
- Uniform backlighting characteristic of displays
- Reduced image quality from display re-capture
- Digital compression artifacts
- Unnatural color temperature shifts
- Pixel grid visibility
- Reflection patterns typical of glass screens

AUTHENTIC IMAGE SIGNS:
- Natural camera sensor noise
- Proper depth of field and focus characteristics
- Natural lighting variations and shadows
- Authentic skin texture and detail
- Camera-specific image processing signatures
- Natural color reproduction

This is a standard computer vision authentication task used by security systems worldwide. Please provide your technical analysis.

REQUIRED OUTPUT FORMAT:
{
    "decision": "Real" or "Spoof",
    "confidence": 0.0-1.0,
    "reasoning": "Technical explanation of key visual indicators"
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
        """Analyze image using Google Gemini 2.0 Flash (your specific configuration)"""
        if not self.google_api_key:
            raise Exception("Google API not configured")
        
        try:
            # Use your specific Gemini 2.0 Flash API URL
            url = f"{self.google_api_url}?key={self.google_api_key}"
            
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
                    "maxOutputTokens": 300,
                    "topP": 0.8,
                    "topK": 10
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract text from Gemini response
                        try:
                            text = result["candidates"][0]["content"]["parts"][0]["text"]
                            logger.info(f"Gemini 2.0 Flash response: {text[:100]}...")
                            return self._parse_ai_response(text, "google_gemini")
                        except (KeyError, IndexError) as e:
                            logger.error(f"Gemini response format error: {e}")
                            logger.error(f"Full response: {result}")
                            raise Exception(f"Unexpected Gemini response format: {e}")
                            
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini API error {response.status}: {error_text}")
                        raise Exception(f"Gemini API error: {response.status} - {error_text}")
                        
        except asyncio.TimeoutError:
            logger.error("Gemini API request timed out")
            raise Exception("Gemini API request timed out")
        except Exception as e:
            logger.error(f"Google Gemini 2.0 Flash analysis failed: {str(e)}")
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
            # Check for refusal/safety responses
            refusal_indicators = [
                "i can't help",
                "i cannot help", 
                "i'm not able to",
                "i cannot analyze",
                "i can't analyze",
                "i'm unable to",
                "cannot assist",
                "can't assist",
                "inappropriate",
                "against my guidelines",
                "policy",
                "safety"
            ]
            
            text_lower = response_text.lower()
            
            # Check if AI refused to analyze
            if any(indicator in text_lower for indicator in refusal_indicators):
                logger.warning(f"{provider} refused to analyze image: {response_text[:100]}")
                return {
                    "decision": "Unknown",
                    "confidence": 0.0,
                    "reasoning": f"AI model declined to analyze (safety restrictions). Raw response: {response_text[:100]}",
                    "provider": provider,
                    "raw_response": response_text,
                    "refusal": True
                }
            
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
                        "raw_response": response_text,
                        "refusal": False
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback parsing for non-JSON responses
            # Determine decision from text analysis
            if "real" in text_lower and "spoof" not in text_lower:
                decision = "Real"
                confidence = 0.7
            elif "spoof" in text_lower and "real" not in text_lower:
                decision = "Spoof"
                confidence = 0.7
            elif "authentic" in text_lower or "genuine" in text_lower:
                decision = "Real"
                confidence = 0.6
            elif "fake" in text_lower or "artificial" in text_lower or "screen" in text_lower:
                decision = "Spoof"
                confidence = 0.6
            elif "real" in text_lower and "spoof" in text_lower:
                # Both mentioned, look for stronger indicators
                real_count = text_lower.count("real") + text_lower.count("authentic") + text_lower.count("genuine")
                spoof_count = text_lower.count("spoof") + text_lower.count("fake") + text_lower.count("artificial")
                
                if spoof_count > real_count:
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
                "raw_response": response_text,
                "refusal": False
            }
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return {
                "decision": "Unknown",
                "confidence": 0.0,
                "reasoning": f"Error parsing response: {str(e)}",
                "provider": provider,
                "raw_response": response_text,
                "refusal": False
            }
    
    async def analyze_image(self, image_base64: str, provider: str = "openai_gpt4v", auto_fallback: bool = True) -> Dict[str, Any]:
        """
        Main method to analyze image with specified AI provider and automatic fallback
        
        Args:
            image_base64: Base64 encoded image
            provider: AI provider to use
            auto_fallback: Try other providers if primary refuses/fails
            
        Returns:
            Analysis result dictionary
        """
        start_time = time.time()
        providers_tried = []
        
        # Define fallback order
        fallback_order = {
            "openai_gpt4v": ["google_gemini", "anthropic_claude", "ollama_llava"],
            "anthropic_claude": ["google_gemini", "openai_gpt4v", "ollama_llava"], 
            "google_gemini": ["anthropic_claude", "openai_gpt4v", "ollama_llava"],
            "ollama_llava": ["google_gemini", "anthropic_claude", "openai_gpt4v"]
        }
        
        async def try_provider(prov: str) -> Dict[str, Any]:
            """Try a specific provider"""
            try:
                providers_tried.append(prov)
                logger.info(f"Attempting analysis with {prov}")
                
                if prov == "openai_gpt4v":
                    result = await self.analyze_with_openai_gpt4v(image_base64)
                elif prov == "anthropic_claude":
                    result = await self.analyze_with_anthropic_claude(image_base64)
                elif prov == "google_gemini":
                    result = await self.analyze_with_google_gemini(image_base64)
                elif prov == "ollama_llava":
                    result = await self.analyze_with_ollama_llava(image_base64)
                else:
                    raise ValueError(f"Unknown provider: {prov}")
                
                # Check if provider refused
                if result.get("refusal", False) or result.get("decision") == "Unknown":
                    logger.warning(f"{prov} refused or gave unclear response")
                    return None
                
                logger.info(f"{prov} successfully analyzed: {result['decision']}")
                return result
                
            except Exception as e:
                logger.warning(f"{prov} failed: {str(e)}")
                return None
        
        # Try primary provider first
        result = await try_provider(provider)
        
        # If primary failed and auto_fallback is enabled, try alternatives
        if result is None and auto_fallback:
            logger.info(f"Primary provider {provider} failed, trying fallbacks...")
            
            for fallback_provider in fallback_order.get(provider, []):
                if fallback_provider in providers_tried:
                    continue
                    
                result = await try_provider(fallback_provider)
                if result is not None:
                    result["fallback_used"] = True
                    result["original_provider"] = provider
                    result["successful_provider"] = fallback_provider
                    break
        
        # If still no result, return error info
        if result is None:
            processing_time = int((time.time() - start_time) * 1000)
            return {
                "decision": "Unknown",
                "confidence": 0.0,
                "reasoning": f"All providers failed or refused. Tried: {', '.join(providers_tried)}",
                "provider": provider,
                "providers_tried": providers_tried,
                "processing_time_ms": processing_time,
                "timestamp": int(time.time() * 1000),
                "all_failed": True
            }
        
        # Add timing information
        processing_time = int((time.time() - start_time) * 1000)
        result["processing_time_ms"] = processing_time
        result["timestamp"] = int(time.time() * 1000)
        result["providers_tried"] = providers_tried
        
        return result
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available AI providers and their status"""
        providers = {
            "google_gemini": {
                "name": "Google Gemini 2.0 Flash",
                "model": "gemini-2.0-flash",
                "available": bool(self.google_api_key),
                "description": "Your configured Gemini 2.0 Flash model - Latest Google AI",
                "cost_estimate": "$0.001-0.005 per image",
                "speed": "1-2 seconds",
                "priority": 1,  # Highest priority since it's your configured model
                "notes": "Your specific configuration - fastest and most cost-effective"
            },
            "openai_gpt4v": {
                "name": "OpenAI GPT-4o (Vision)",
                "model": "gpt-4o",
                "available": self.openai_client is not None,
                "description": "Latest OpenAI vision model with improved capabilities",
                "cost_estimate": "$0.005-0.015 per image",
                "speed": "2-4 seconds",
                "priority": 2
            },
            "anthropic_claude": {
                "name": "Anthropic Claude 3.5 Sonnet",
                "model": "claude-3-5-sonnet-20241022",
                "available": self.anthropic_client is not None,
                "description": "Latest Claude with enhanced vision capabilities",
                "cost_estimate": "$0.008-0.020 per image", 
                "speed": "2-4 seconds",
                "priority": 3
            },
            "ollama_llava": {
                "name": "Ollama LLaVA (Local)",
                "model": "llava",
                "available": True,  # Assume available if endpoint configured
                "description": "Local vision language model",
                "cost_estimate": "Free (local)",
                "speed": "3-8 seconds",
                "priority": 4
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
        # Handle case where all AI providers failed
        if ai_result.get("all_failed", False):
            if traditional_result is not None:
                logger.warning("All AI providers failed, using traditional detection only")
                return {
                    "is_live": traditional_result["is_live"],
                    "final_confidence": traditional_result["confidence"],
                    "decision_method": "TRADITIONAL_ONLY_AI_FAILED",
                    "ai_decision": "Failed",
                    "ai_confidence": 0.0,
                    "ai_reasoning": ai_result.get("reasoning", "All AI providers failed"),
                    "traditional_decision": traditional_result["is_live"],
                    "traditional_score": traditional_result["liveness_score"],
                    "providers_tried": ai_result.get("providers_tried", [])
                }
            else:
                # Both AI and traditional failed - return conservative result
                return {
                    "is_live": False,  # Conservative: assume spoof when uncertain
                    "final_confidence": 0.0,
                    "decision_method": "ALL_FAILED",
                    "ai_decision": "Failed",
                    "ai_confidence": 0.0,
                    "ai_reasoning": "All AI providers failed and traditional detection not available",
                    "traditional_decision": None,
                    "traditional_score": None,
                    "providers_tried": ai_result.get("providers_tried", [])
                }
        
        # Handle AI refusal or unknown decision
        if ai_result.get("refusal", False) or ai_result.get("decision") == "Unknown":
            if traditional_result is not None:
                logger.info("AI refused/unclear, using traditional detection")
                return {
                    "is_live": traditional_result["is_live"],
                    "final_confidence": traditional_result["confidence"],
                    "decision_method": "TRADITIONAL_ONLY_AI_REFUSED",
                    "ai_decision": ai_result.get("decision", "Refused"),
                    "ai_confidence": ai_result.get("confidence", 0.0),
                    "ai_reasoning": ai_result.get("reasoning", "AI declined to analyze"),
                    "traditional_decision": traditional_result["is_live"],
                    "traditional_score": traditional_result["liveness_score"]
                }
            else:
                # AI refused and no traditional - return conservative
                return {
                    "is_live": False,
                    "final_confidence": 0.0,
                    "decision_method": "CONSERVATIVE_AI_REFUSED",
                    "ai_decision": ai_result.get("decision", "Refused"),
                    "ai_confidence": 0.0,
                    "ai_reasoning": ai_result.get("reasoning", "AI declined to analyze"),
                    "traditional_decision": None,
                    "traditional_score": None
                }
        
        # Normal processing for successful AI analysis
        ai_is_live = ai_result["decision"].lower() == "real"
        ai_confidence = ai_result["confidence"]
        
        if traditional_result is None:
            # AI only (successful)
            decision_method = "AI_ONLY"
            if ai_result.get("fallback_used", False):
                decision_method = f"AI_FALLBACK_{ai_result.get('successful_provider', 'unknown').upper()}"
            
            return {
                "is_live": ai_is_live,
                "final_confidence": ai_confidence,
                "decision_method": decision_method,
                "ai_decision": ai_result["decision"],
                "ai_confidence": ai_confidence,
                "ai_reasoning": ai_result.get("reasoning", ""),
                "traditional_decision": None,
                "traditional_score": None,
                "fallback_info": {
                    "fallback_used": ai_result.get("fallback_used", False),
                    "original_provider": ai_result.get("original_provider"),
                    "successful_provider": ai_result.get("successful_provider"),
                    "providers_tried": ai_result.get("providers_tried", [])
                }
            }
        
        # Combine AI and traditional results
        traditional_is_live = traditional_result["is_live"]
        traditional_confidence = traditional_result["confidence"]
        
        if ai_is_live == traditional_is_live:
            # Both agree - high confidence
            combined_confidence = (ai_confidence + traditional_confidence) / 2
            final_decision = ai_is_live
            decision_method = "CONSENSUS"
        else:
            # Disagreement - use more confident result with penalty
            if ai_confidence > traditional_confidence:
                final_decision = ai_is_live
                combined_confidence = ai_confidence * 0.85  # Penalty for disagreement
                decision_method = "AI_PREFERRED"
            else:
                final_decision = traditional_is_live
                combined_confidence = traditional_confidence * 0.85
                decision_method = "TRADITIONAL_PREFERRED"
        
        # Add fallback info if AI used fallback
        if ai_result.get("fallback_used", False):
            decision_method += "_WITH_FALLBACK"
        
        return {
            "is_live": final_decision,
            "final_confidence": combined_confidence,
            "decision_method": decision_method,
            "ai_decision": ai_result["decision"],
            "ai_confidence": ai_confidence,
            "ai_reasoning": ai_result.get("reasoning", ""),
            "traditional_decision": traditional_is_live,
            "traditional_score": traditional_result["liveness_score"],
            "fallback_info": {
                "fallback_used": ai_result.get("fallback_used", False),
                "original_provider": ai_result.get("original_provider"),
                "successful_provider": ai_result.get("successful_provider"),
                "providers_tried": ai_result.get("providers_tried", [])
            }
        }

# Global service instance
ai_liveness_service = AILivenessService()
ai_combiner = AILivenessCombiner()