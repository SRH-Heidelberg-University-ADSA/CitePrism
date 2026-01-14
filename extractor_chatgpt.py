"""
CitePrism Phase 1: Academic Paper Parser
==========================================
This script extracts text from academic PDFs and uses LLMs (GPT-4o or Gemini 1.5)
to parse manuscript structure, citations, and references into structured JSON.

Author: CitePrism Team
Version: 1.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv  

# Load environment variables from .env file immediately
load_dotenv() 

# PDF extraction libraries
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
except ImportError:
    pdfminer_extract = None

# LLM libraries
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR OUTPUT VALIDATION
# ============================================================================

class ParsedReference(BaseModel):
    """Structured data for a single parsed reference."""
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None


class Reference(BaseModel):
    """Complete reference entry with raw and parsed data."""
    ref_id: str
    raw_text: str
    parsed: ParsedReference


class CitationInText(BaseModel):
    """In-text citation with context window."""
    marker: str
    context_window: str


class Metadata(BaseModel):
    """Manuscript metadata."""
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None


class ManuscriptStructure(BaseModel):
    """Complete manuscript parsing output."""
    metadata: Metadata
    citations_in_text: List[CitationInText] = Field(default_factory=list)
    references_list: List[Reference] = Field(default_factory=list)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for CitePrism Phase 1."""

    # LLM Provider: "openai" or "google"
    LLM_PROVIDER: Literal["openai", "google"] = os.getenv("LLM_PROVIDER", "openai")

    # API Keys (load from environment variables)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

    # Model names
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")

    # Token limits for API calls
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "16000"))

    # File paths
    INPUT_DIR: Path = Path("input")
    OUTPUT_DIR: Path = Path("output")
    DEBUG_DIR: Path = Path("debug")  # Save raw responses for debugging

    # PDF extraction method: "pypdf" or "pdfminer"
    PDF_EXTRACTOR: Literal["pypdf", "pdfminer"] = "pypdf"


# ============================================================================
# SYSTEM PROMPT (EXACTLY AS SPECIFIED)
# ============================================================================

SYSTEM_PROMPT = """You are an expert academic editor and parser. Your goal is to convert raw manuscript text into a structured JSON format.

Task 1: Metadata Extraction Extract the `title`, `authors` (list), and `abstract` of the manuscript.

Task 2: Section Segmentation Identify main headers and extract text content. Return a list of objects with `section_title` and `section_content`.

Task 3: Citation Context Extraction Locate every in-text citation marker (e.g., `[1]`, `(Smith, 2020)`). For EACH citation:
1. Extract the Context Window (the sentence with the citation ± 1 sentence before/after).
2. Extract the raw `citation_marker`.

Task 4: Reference List Parsing (Crucial for API Lookup) Locate the 'References' or 'Bibliography' section. For every item in the list, you MUST parse the raw string into structured fields.
* `ref_id`: The marker used in the text (e.g., '1', '[1]', 'Smith 2020').
* `raw_text`: The complete original reference string.
* `title`: The title of the cited work.
* `authors`: A list of the cited authors.
* `year`: The publication year (integer).
* `venue`: The journal, conference, or publisher name.
* `doi`: The DOI string if present (e.g., '10.1145/...') OR null.

Output Format: Return ONLY valid JSON adhering to this schema:
```json
{
  "metadata": {
    "title": "String",
    "authors": ["String"],
    "abstract": "String"
  },
  "citations_in_text": [
    {
      "marker": "[1]",
      "context_window": "Previous sentence... Target... Following..."
    }
  ],
  "references_list": [
    {
      "ref_id": "[1]",
      "raw_text": "[1] J. Smith, 'Deep Learning', Nature, 2020.",
      "parsed": {
        "title": "Deep Learning",
        "authors": ["J. Smith"],
        "year": 2020,
        "venue": "Nature",
        "doi": null
      }
    }
  ]
}
```

Do not truncate text. If the text is too long, summarize section content but KEEP citations and reference lists exact."""


# ============================================================================
# PDF EXTRACTION
# ============================================================================

def extract_text_from_pdf(pdf_path: Path, method: str = "pypdf") -> str:
    """
    Extract raw text from a PDF file using specified method.

    Args:
        pdf_path: Path to the PDF file
        method: Extraction method - "pypdf" or "pdfminer"

    Returns:
        Extracted text as string

    Raises:
        ValueError: If extraction method is unavailable
        Exception: If PDF extraction fails
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")

    if method == "pypdf":
        if pypdf is None:
            raise ValueError("pypdf library not installed. Install with: pip install pypdf")

        try:
            text_parts = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"PDF has {total_pages} pages")

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    logger.debug(f"Extracting page {page_num}/{total_pages}")
                    text_parts.append(page.extract_text())

            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters")
            return full_text

        except Exception as e:
            logger.error(f"pypdf extraction failed: {e}")
            raise

    elif method == "pdfminer":
        if pdfminer_extract is None:
            raise ValueError("pdfminer.six library not installed. Install with: pip install pdfminer.six")

        try:
            full_text = pdfminer_extract(str(pdf_path))
            logger.info(f"Extracted {len(full_text)} characters")
            return full_text

        except Exception as e:
            logger.error(f"pdfminer extraction failed: {e}")
            raise

    else:
        raise ValueError(f"Unknown extraction method: {method}")


# ============================================================================
# LLM INTERFACE - UNIFIED ABSTRACTION
# ============================================================================

class LLMInterface:
    """Unified interface for different LLM providers."""

    def parse_manuscript(self, text: str, debug_path: Optional[Path] = None) -> Dict:
        """
        Send manuscript text to LLM for parsing.

        Args:
            text: Raw manuscript text
            debug_path: Optional path to save raw LLM response

        Returns:
            Parsed structure as dictionary
        """
        raise NotImplementedError


class OpenAIInterface(LLMInterface):
    """OpenAI GPT-4o implementation."""

    def __init__(self, api_key: str, model: str = "gpt-4o", max_tokens: int = 16000):
        if OpenAI is None:
            raise ValueError("openai library not installed. Install with: pip install openai")

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        logger.info(f"Initialized OpenAI client with model: {model}")
        logger.info(f"Max output tokens: {max_tokens}")

    def parse_manuscript(self, text: str, debug_path: Optional[Path] = None) -> Dict:
        """Parse manuscript using OpenAI GPT-4o."""
        logger.info("Sending request to OpenAI API...")

        try:
            # Use structured output to ensure JSON response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Parse this manuscript:\n\n{text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent parsing
                max_tokens=self.max_tokens  # Increased token limit
            )

            # Extract JSON from response
            json_text = response.choices[0].message.content

            # Save raw response for debugging
            if debug_path:
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(json_text)
                logger.info(f"Saved raw LLM response to: {debug_path}")

            # Check if response was truncated
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                logger.warning("⚠ Response was truncated due to token limit!")
                logger.warning(f"⚠ Consider increasing MAX_TOKENS (current: {self.max_tokens})")

            parsed_data = json.loads(json_text)

            logger.info("✓ Successfully received and parsed OpenAI response")
            return parsed_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from OpenAI response: {e}")
            if debug_path and debug_path.exists():
                logger.error(f"Check the raw response in: {debug_path}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GoogleInterface(LLMInterface):
    """Google Gemini 1.5 implementation."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro", max_tokens: int = 16000):
        if genai is None:
            raise ValueError("google-generativeai library not installed. Install with: pip install google-generativeai")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
                "max_output_tokens": max_tokens
            }
        )
        self.max_tokens = max_tokens
        logger.info(f"Initialized Google Gemini client with model: {model}")
        logger.info(f"Max output tokens: {max_tokens}")

    def parse_manuscript(self, text: str, debug_path: Optional[Path] = None) -> Dict:
        """Parse manuscript using Google Gemini."""
        logger.info("Sending request to Google Gemini API...")

        try:
            # Combine system prompt and user message
            full_prompt = f"{SYSTEM_PROMPT}\n\nParse this manuscript:\n\n{text}"

            response = self.model.generate_content(full_prompt)
            json_text = response.text

            # Save raw response for debugging
            if debug_path:
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(json_text)
                logger.info(f"Saved raw LLM response to: {debug_path}")

            parsed_data = json.loads(json_text)

            logger.info("✓ Successfully received and parsed Gemini response")
            return parsed_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from Gemini response: {e}")
            if debug_path and debug_path.exists():
                logger.error(f"Check the raw response in: {debug_path}")
            raise
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            raise


def get_llm_interface(provider: str, config: Config) -> LLMInterface:
    """
    Factory function to get the appropriate LLM interface.

    Args:
        provider: "openai" or "google"
        config: Configuration object

    Returns:
        LLM interface instance
    """
    if provider == "openai":
        return OpenAIInterface(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            max_tokens=config.MAX_TOKENS
        )
    elif provider == "google":
        return GoogleInterface(
            api_key=config.GOOGLE_API_KEY,
            model=config.GOOGLE_MODEL,
            max_tokens=config.MAX_TOKENS
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_pdf(
    pdf_path: Path,
    output_path: Path,
    config: Config
) -> ManuscriptStructure:
    """
    Complete pipeline: Extract PDF → Parse with LLM → Validate → Save.

    Args:
        pdf_path: Path to input PDF file
        output_path: Path to save JSON output
        config: Configuration object

    Returns:
        Validated ManuscriptStructure object
    """
    logger.info("=" * 80)
    logger.info(f"Processing: {pdf_path.name}")
    logger.info("=" * 80)

    # Step 1: Extract text from PDF
    try:
        raw_text = extract_text_from_pdf(pdf_path, method=config.PDF_EXTRACTOR)
        if not raw_text or len(raw_text.strip()) < 100:
            raise ValueError("Extracted text is too short or empty")
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise

    # Step 2: Initialize LLM interface
    try:
        llm = get_llm_interface(config.LLM_PROVIDER, config)
    except Exception as e:
        logger.error(f"Failed to initialize LLM interface: {e}")
        raise

    # Step 3: Parse manuscript with LLM
    debug_path = config.DEBUG_DIR / f"{pdf_path.stem}_raw_response.json"
    try:
        parsed_dict = llm.parse_manuscript(raw_text, debug_path=debug_path)
    except Exception as e:
        logger.error(f"LLM parsing failed: {e}")
        raise

    # Step 4: Validate output with Pydantic
    try:
        manuscript = ManuscriptStructure(**parsed_dict)
        logger.info("✓ Output validation successful")
        logger.info(f"  - Title: {manuscript.metadata.title}")
        logger.info(f"  - Authors: {len(manuscript.metadata.authors)}")
        logger.info(f"  - Citations: {len(manuscript.citations_in_text)}")
        logger.info(f"  - References: {len(manuscript.references_list)}")
    except Exception as e:
        logger.error(f"Output validation failed: {e}")
        raise

    # Step 5: Save to JSON file
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                manuscript.model_dump(),
                f,
                indent=2,
                ensure_ascii=False
            )
        logger.info(f"✓ Saved output to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        raise

    return manuscript


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the script."""
    import sys

    # Initialize configuration
    config = Config()

    # Check dependencies
    if config.PDF_EXTRACTOR == "pypdf" and pypdf is None:
        logger.error("pypdf not installed. Run: pip install pypdf")
        sys.exit(1)
    elif config.PDF_EXTRACTOR == "pdfminer" and pdfminer_extract is None:
        logger.error("pdfminer.six not installed. Run: pip install pdfminer.six")
        sys.exit(1)

    if config.LLM_PROVIDER == "openai" and OpenAI is None:
        logger.error("openai not installed. Run: pip install openai")
        sys.exit(1)
    elif config.LLM_PROVIDER == "google" and genai is None:
        logger.error("google-generativeai not installed. Run: pip install google-generativeai")
        sys.exit(1)

    # Create directories
    config.INPUT_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.DEBUG_DIR.mkdir(exist_ok=True)

    # Find PDF files in input directory
    pdf_files = list(config.INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {config.INPUT_DIR}")
        logger.info(f"Please place PDF files in the '{config.INPUT_DIR}' directory")
        sys.exit(0)

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    logger.info(f"Using LLM provider: {config.LLM_PROVIDER}")

    # Process each PDF
    results = []
    for pdf_path in pdf_files:
        output_path = config.OUTPUT_DIR / f"{pdf_path.stem}_parsed.json"

        try:
            manuscript = process_pdf(pdf_path, output_path, config)
            results.append({"file": pdf_path.name, "status": "success"})
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            results.append({"file": pdf_path.name, "status": "failed", "error": str(e)})

    # Summary
    logger.info("=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Successful: {success_count}/{len(results)}")

    for result in results:
        status_icon = "✓" if result["status"] == "success" else "✗"
        logger.info(f"{status_icon} {result['file']}")


if __name__ == "__main__":
    main()