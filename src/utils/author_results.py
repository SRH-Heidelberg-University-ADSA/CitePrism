import os
import json
import logging
import re
from pathlib import Path
from typing import Optional, Literal, Dict, List
from dotenv import load_dotenv
from thefuzz import fuzz

# Import LLM Clients
from google import genai
from google.genai import types
from openai import OpenAI

# --- 1. CONFIGURATION LOADING ---
load_dotenv()

class Config:
    # LLM Provider selection
    LLM_PROVIDER: Literal["openai", "google"] = os.getenv("LLM_PROVIDER", "google")
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

    # Model names
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

    # File paths
    INPUT_DIR: Path = Path(os.getenv("INPUT_DIR", "output"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "final_disambiguated"))
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 2. LLM SERVICE WRAPPER ---
class LLMService:
    def __init__(self):
        self.provider = Config.LLM_PROVIDER
        if self.provider == "google":
            if not Config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY is missing in .env")
            self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
            self.model = Config.GOOGLE_MODEL
            logger.info(f"Initialized Google Gemini service using {self.model}")
        else:
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is missing in .env")
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.model = Config.OPENAI_MODEL
            logger.info(f"Initialized OpenAI service using {self.model}")

    def get_canonical_name(self, group: List[str]) -> str:
        """Asks the LLM to pick the best/canonical name from a fuzzy group."""
        prompt = (
            f"Review these name variations for the same researcher found in academic papers:\n{group}\n\n"
            "Return ONLY the most complete and formal 'Canonical' version of the name. "
            "Do not include any explanation or markdown blocks."
        )

        try:
            if self.provider == "google":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(max_output_tokens=60)
                )
                return response.text.strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Resolution failed: {e}")
            return group[0] # Fallback to first name in list

# --- 3. AUTHOR RESOLVER ---
class AuthorResolver:
    def __init__(self):
        self.llm = LLMService()

    def extract_authors(self, data: dict) -> List[str]:
        """Collects unique author strings from meta-data and references."""
        authors = set()
        
        # From Manuscript Metadata
        for a in data.get("manuscript_metadata", {}).get("authors", []):
            authors.add(a)

        # From Enriched References
        for ref in data.get("enriched_references", []):
            # Check original parsed names
            orig = ref.get("original_data", {}).get("parsed", {}).get("authors", [])
            for a in orig: authors.add(a)
            
            # Check external metadata (OpenAlex/Crossref)
            ext = ref.get("external_metadata", {}).get("authors", [])
            for a in ext:
                name = a.get("display_name") if isinstance(a, dict) else a
                if name: authors.add(name)
        
        return sorted(list(authors))

    def group_fuzzy_names(self, names: List[str], threshold=70) -> List[List[str]]:
        """Lowered threshold to 70 to catch variations like 'Initials vs Full Name'."""
        groups = []
        visited = set()
        
        # Log the count so you know the script is actually seeing authors
        logger.info(f"Total raw author strings found: {len(names)}")
        
        for i, name1 in enumerate(names):
            if name1 in visited: continue
            group = [name1]
            visited.add(name1)
            for name2 in names[i+1:]:
                if name2 not in visited:
                    # token_sort_ratio is great for "Surname, First" vs "First Surname"
                    score = fuzz.token_sort_ratio(name1, name2)
                    if score >= threshold:
                        group.append(name2)
                        visited.add(name2)
            groups.append(group)
        
        # Log how many groups were created
        logger.info(f"Created {len(groups)} author groups. (Groups > 1 will be sent to Gemini)")
        return groups

    def process_file(self, filename: str):
        input_path = Config.INPUT_DIR / filename
        if not input_path.exists():
            logger.error(f"File {filename} not found in {Config.INPUT_DIR}")
            return

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Analyzing authors in {filename}...")
        raw_names = self.extract_authors(data)
        fuzzy_groups = self.group_fuzzy_names(raw_names)
        
        # Disambiguate with LLM
        resolution_map = {}
        for group in fuzzy_groups:
            if len(group) > 1:
                canonical = self.llm.get_canonical_name(group)
                for name in group:
                    resolution_map[name] = canonical
                logger.info(f"Merged: {group} -> {canonical}")
            else:
                resolution_map[group[0]] = group[0]

        # Output resolution map
        output_file = Config.OUTPUT_DIR / f"author_map_{filename}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(resolution_map, f, indent=2)
            
        logger.info(f"✅ Disambiguation complete. Map saved to {output_file}")

# --- 4. MAIN ---
if __name__ == "__main__":
    resolver = AuthorResolver()
    # Path logic to process your specific file
    resolver.process_file("paper3_parsed_2.5.json")