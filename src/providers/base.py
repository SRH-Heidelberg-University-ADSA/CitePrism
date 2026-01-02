# src/providers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.model = config.get("model")

    @abstractmethod
    def extract_structured_data(self, text: str, prompt_instruction: str) -> Dict:
        """
        Extract structured data from text.
        
        Args:
            text: The context text to process.
            prompt_instruction: Specific instruction (e.g., "Extract authors").
            
        Returns:
            Dict: Parsed JSON response.
        """
        pass