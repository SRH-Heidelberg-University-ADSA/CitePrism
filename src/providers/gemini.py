import os
import json
from typing import Dict, Any
import google.genai as genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from src.providers.base import LLMProvider
from src.utils.exceptions import LLMProviderError
from src.utils.logger import logger

class GeminiProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 1. Get API Key
        self.api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise LLMProviderError("Gemini/Google API Key is missing.")
            
        # 2. Configure the client (using new API)
        self.client = genai.Client(api_key=self.api_key)
        
        # 3. Set Model
        self.model_name = config.get("model", "gemini-2.0-flash")
        
        # 4. Store configuration
        self.temperature = config.get("temperature", 0.1)
        self.max_output_tokens = config.get("max_new_tokens", 8192)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract_structured_data(self, text: str, prompt_instruction: str) -> Dict:
        """
        Uses Google Gemini to generate structured JSON response.
        """
        try:
            logger.info(f"Sending request to Gemini: {self.model_name}")

            # Construct the prompt
            prompt = f"""
            You are a strict data extraction assistant.
            {prompt_instruction}

            TEXT TO ANALYZE:
            {text}

            IMPORTANT: Return ONLY valid JSON with no additional text, markdown, or explanations.
            """
            
            # Generate content with JSON mode
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
            )

            # Extract text and parse JSON
            response_text = response.text.strip()
            
            # Clean up any potential markdown
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Find JSON in response
            start = response_text.find('{')
            end = response_text.rfind('}')
            
            if start != -1 and end != -1:
                json_str = response_text[start:end+1]
                return json.loads(json_str)
            else:
                # Try direct parse
                return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response: {response_text}")
            logger.error(f"JSON Decode Error: {str(e)}")
            # Try to extract JSON with more aggressive cleaning
            return self._clean_and_parse_json(response_text)
        except Exception as e:
            logger.error(f"Gemini API Error: {str(e)}")
            raise LLMProviderError(f"API request failed: {str(e)}")

    def _clean_and_parse_json(self, text: str) -> Dict:
        """
        Fallback cleaner for problematic JSON.
        """
        import re
        
        # Remove all markdown code blocks
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Remove any text before first { or after last }
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
            
            # Fix common JSON issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            json_str = re.sub(r'}\s*{', '},{', json_str)  # Fix multiple objects
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Even after cleaning, JSON is invalid: {json_str[:200]}...")
                raise
        else:
            raise json.JSONDecodeError("No JSON found in response", text, 0)