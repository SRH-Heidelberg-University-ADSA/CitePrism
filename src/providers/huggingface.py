# src/providers/huggingface.py
import os
import json
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from huggingface_hub import InferenceClient  # <--- Using the official library
from src.providers.base import LLMProvider
from src.utils.exceptions import LLMProviderError
from src.utils.logger import logger
class HuggingFaceProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = self.api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise LLMProviderError("Hugging Face API Key is missing.")
        # Initialize the official client
        self.client = InferenceClient(api_key=self.api_key)
        self.model_id = config.get("model", "mistralai/Mistral-7B-Instruct-v0.3")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract_structured_data(self, text: str, prompt_instruction: str) -> Dict:
        """
        Uses HuggingFace InferenceClient to generate response.
        """
        # Prompt formatting for Chat Models
        messages = [
            {"role": "system", "content": "You are a strict data extraction assistant. Return ONLY valid JSON. Do not write code blocks."},
            {"role": "user", "content": f"{prompt_instruction}\n\nTEXT TO ANALYZE:\n{text}"}
        ]
        try:
            logger.info(f"Sending request to Hugging Face: {self.model_id}")
            # Using chat_completion which handles special tokens automatically
            response = self.client.chat_completion(
                model=self.model_id,
                messages=messages,
                max_tokens=self.config.get("max_new_tokens", 4096),
                temperature=self.config.get("temperature", 0.1)
            )
            # Extract content
            content = response.choices[0].message.content
            return self._clean_and_parse_json(content)
        except Exception as e:
            logger.error(f"Hugging Face API Error: {str(e)}")
            raise LLMProviderError(f"API request failed: {str(e)}")
    def _clean_and_parse_json(self, text: str) -> Dict:
        """
        Cleans Markdown code blocks and parses JSON.
        """
        import re
        # Remove ```json and ``` if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        # Attempt to find the first '{' and last '}' to strip extra chatter
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        return json.loads(text)