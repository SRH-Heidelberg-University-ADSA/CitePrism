# src/providers/openai.py
import os
import json
import time
from typing import Dict, Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.providers.base import LLMProvider
from src.utils.exceptions import LLMProviderError
from src.utils.logger import logger


class OpenAIProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
        if not self.api_key:
            raise LLMProviderError("OpenAI/ChatGPT API Key is missing.")

        self.client = OpenAI(api_key=self.api_key)

        self.model_name = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_new_tokens", 4096)
        self.timeout = config.get("timeout", 60)

        # ✅ debug controls
        self.debug_llm = bool(config.get("debug_llm", False)) or os.getenv("CITEPRISM_DEBUG_LLM") == "1"
        self.dump_dir = config.get("llm_dump_dir", "./llm_dumps")
        if self.debug_llm:
            os.makedirs(self.dump_dir, exist_ok=True)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def extract_structured_data(self, text: str, prompt_instruction: str) -> Dict:
        """
        Uses OpenAI to generate structured JSON response.
        ✅ Logs raw response (and optionally dumps to disk) when debug_llm enabled.
        """
        response_text = ""
        try:
            logger.info(f"Sending request to OpenAI: {self.model_name}")

            # keep prompts tight (less chance of hallucinated wrapper text)
            messages = [
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Return ONLY valid JSON. No markdown. No explanations.",
                },
                {
                    "role": "user",
                    "content": f"{prompt_instruction}\n\nTEXT:\n{text}",
                },
            ]

            if self.debug_llm:
                logger.info(f"[LLM DEBUG] prompt_preview={prompt_instruction[:300]!r}")
                logger.info(f"[LLM DEBUG] text_chars={len(text)} text_preview={text[:300]!r}")

            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                timeout=self.timeout,
            )

            response_text = (resp.choices[0].message.content or "").strip()

            if self.debug_llm:
                logger.info("===== RAW LLM RESPONSE (BEGIN) =====")
                logger.info(response_text[:4000])  # avoid log explosion
                logger.info("===== RAW LLM RESPONSE (END) =====")

                # dump full raw response to disk for inspection
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = os.path.join(self.dump_dir, f"openai_{ts}_{int(time.time()*1000)}.json.txt")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(response_text)

            parsed = json.loads(response_text)
            return parsed

        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from OpenAI response")
            logger.error(f"Raw response text (first 1000 chars): {response_text[:1000]}")
            raise LLMProviderError(f"Invalid JSON from OpenAI: {e}")

        except Exception as e:
            logger.error(f"OpenAI API Error: {e}", exc_info=True)
            raise LLMProviderError(f"API request failed: {e}")
