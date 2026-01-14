import os
import json
import logging
from pathlib import Path
import re
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
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# JSON repair library (optional but recommended)
try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not HAS_JSON_REPAIR:
    logger.warning("json-repair not installed. For better JSON parsing, install with: pip install json-repair")


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
    parsed: ParsedReference


class CitationInText(BaseModel):
    """In-text citation with context window."""
    marker: str
    context_window: str


class Metadata(BaseModel):
    """Manuscript metadata."""
    title: Optional[str] = None
    doi: Optional[str] = None
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

    # LLM Provider: Changed default to "google" for Gemini 2.5 usage
    LLM_PROVIDER: Literal["openai", "google"] = os.getenv("LLM_PROVIDER", "google")

    # API Keys (load from environment variables)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

    # Model names
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    # Updated to Gemini 2.5 Flash (Latest efficient model with 1M context)
    GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

    # Token limits for API calls (Increased for larger documents)
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "32000"))

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

Do not truncate text. If the text is too long, summarize section content but KEEP citations and reference lists exact.
"""


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
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF extraction fails
    """
    # Check if file exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Check if file is readable
    if not os.access(pdf_path, os.R_OK):
        raise PermissionError(f"Cannot read PDF file: {pdf_path}")
    
    logger.info(f"Extracting text from PDF: {pdf_path}")

    if method == "pypdf":
        if pypdf is None:
            raise ValueError("pypdf library not installed. Install with: pip install pypdf")

        try:
            text_parts = []
            with open(pdf_path, 'rb') as file:
                try:
                    pdf_reader = pypdf.PdfReader(file)
                except pypdf.errors.PdfReadError as e:
                    raise ValueError(f"Invalid or corrupted PDF file: {e}")
                except Exception as e:
                    raise ValueError(f"Failed to read PDF file: {e}")
                
                total_pages = len(pdf_reader.pages)
                
                if total_pages == 0:
                    raise ValueError("PDF file has no pages")
                
                logger.info(f"PDF has {total_pages} pages")

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        logger.debug(f"Extracting page {page_num}/{total_pages}")
                        page_text = page.extract_text()
                        
                        if page_text:
                            text_parts.append(page_text)
                        else:
                            logger.warning(f"Page {page_num} appears to be empty or unreadable")
                    
                    except Exception as e:
                        logger.error(f"Failed to extract text from page {page_num}: {e}")
                        # Continue with other pages
                        continue

            if not text_parts:
                raise ValueError("No text could be extracted from any page in the PDF")
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from {len(text_parts)} pages")
            
            return full_text

        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            logger.error(f"pypdf extraction failed: {e}")
            raise Exception(f"PDF extraction error: {e}")

    elif method == "pdfminer":
        if pdfminer_extract is None:
            raise ValueError("pdfminer.six library not installed. Install with: pip install pdfminer.six")

        try:
            full_text = pdfminer_extract(str(pdf_path))
            
            if not full_text or len(full_text.strip()) < 10:
                raise ValueError("Extracted text is empty or too short")
            
            logger.info(f"Extracted {len(full_text)} characters")
            return full_text

        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            logger.error(f"pdfminer extraction failed: {e}")
            raise Exception(f"PDF extraction error: {e}")

    else:
        raise ValueError(f"Unknown extraction method: {method}. Use 'pypdf' or 'pdfminer'")


# ============================================================================
# DOI EXTRACTION
# ============================================================================

def extract_doi_from_text(text: str) -> Optional[str]:
    """
    Extract the first DOI found in the given text using regex.

    Args:
        text: Raw text (e.g., full manuscript text)

    Returns:
        DOI string if found, otherwise None
    """
    try:
        # Standard Crossref-recommended DOI pattern
        doi_pattern = re.compile(
            r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b',
            re.IGNORECASE
        )

        match = doi_pattern.search(text)
        doi = match.group(0) if match else None
        
        if doi:
            logger.debug(f"Extracted DOI: {doi}")
        
        return doi
    
    except Exception as e:
        logger.warning(f"Error during DOI extraction: {e}")
        return None


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

        try:
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self.max_tokens = max_tokens
            logger.info(f"Initialized OpenAI client with model: {model}")
            logger.info(f"Max output tokens: {max_tokens}")
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

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
                max_tokens=self.max_tokens
            )

            # Extract JSON from response
            json_text = response.choices[0].message.content

            if not json_text:
                raise ValueError("OpenAI returned empty response")

            # Save raw response for debugging
            if debug_path:
                try:
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        f.write(json_text)
                    logger.info(f"Saved raw LLM response to: {debug_path}")
                except Exception as e:
                    logger.warning(f"Failed to save debug file: {e}")

            # Check if response was truncated
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                logger.warning("⚠ Response was truncated due to token limit!")
                logger.warning(f"⚠ Consider increasing MAX_TOKENS (current: {self.max_tokens})")

            # Parse JSON
            try:
                parsed_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from OpenAI response: {e}")
                
                # Try to repair JSON if library is available
                if HAS_JSON_REPAIR:
                    logger.info("Attempting to repair JSON...")
                    try:
                        repaired_json = repair_json(json_text)
                        parsed_data = json.loads(repaired_json)
                        logger.info("✓ Successfully repaired and parsed JSON")
                    except Exception as repair_error:
                        logger.error(f"JSON repair failed: {repair_error}")
                        raise e
                else:
                    raise e

            logger.info("✓ Successfully received and parsed OpenAI response")
            return parsed_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            if debug_path and debug_path.exists():
                logger.error(f"Check the raw response in: {debug_path}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GoogleInterface(LLMInterface):
    """
    Google GenAI Interface using 'google-genai' SDK (v2) with robust error handling.
    """
    def __init__(self, api_key: str, model: str, max_tokens: int):
        if genai is None or types is None:
            raise ValueError("google-genai library not installed. Run: pip install google-genai")
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        try:
            self.client = genai.Client(api_key=api_key)
            self.model = model
            self.max_tokens = max_tokens
            logger.info(f"Initialized Google GenAI Client with model: {model}")
            logger.info(f"Max output tokens: {max_tokens}")
        except Exception as e:
            raise ValueError(f"Failed to initialize Google GenAI client: {e}")

    def parse_manuscript(self, text: str, debug_path: Optional[Path] = None) -> Dict:
        logger.info("Sending request to Google Gemini API (v2 SDK)...")
        
        try:
            # Use types.Part(text=...) constructor (FIXED)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=SYSTEM_PROMPT),
                            types.Part(text=f"Parse this manuscript:\n\n{text}")
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ManuscriptStructure,  # Pass Pydantic model directly
                    temperature=0.1,
                    max_output_tokens=self.max_tokens  # FIXED: Added max_output_tokens
                )
            )

            if not response or not response.text:
                raise ValueError("Google GenAI returned empty response")

            # Save raw response for debugging
            if debug_path:
                try:
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    logger.info(f"Saved raw LLM response to: {debug_path}")
                except Exception as e:
                    logger.warning(f"Failed to save debug file: {e}")

            # Try multiple parsing strategies
            parsed_data = None
            
            # Strategy 1: Use the SDK's built-in parsed object (most reliable)
            if hasattr(response, 'parsed') and response.parsed:
                try:
                    logger.info("Using SDK's parsed object")
                    parsed_data = response.parsed.model_dump()
                    logger.info("✓ Successfully parsed using SDK's built-in parser")
                except Exception as e:
                    logger.warning(f"Failed to use SDK's parsed object: {e}")
            
            # Strategy 2: Clean and parse the text manually
            if parsed_data is None:
                logger.info("Attempting manual JSON parsing")
                try:
                    clean_text = self._clean_json_response(response.text)
                    
                    # Save cleaned JSON for debugging
                    if debug_path:
                        try:
                            clean_path = debug_path.parent / f"{debug_path.stem}_cleaned.json"
                            with open(clean_path, 'w', encoding='utf-8') as f:
                                f.write(clean_text)
                            logger.info(f"Saved cleaned JSON to: {clean_path}")
                        except Exception as e:
                            logger.warning(f"Failed to save cleaned JSON: {e}")
                    
                    parsed_data = json.loads(clean_text)
                    logger.info("✓ Successfully parsed using manual JSON parsing")
                
                except json.JSONDecodeError as e:
                    logger.error(f"Manual JSON parsing failed: {e}")
                    
                    # Strategy 3: Try JSON repair if available
                    if HAS_JSON_REPAIR:
                        logger.info("Attempting JSON repair...")
                        try:
                            repaired_json = repair_json(clean_text)
                            parsed_data = json.loads(repaired_json)
                            logger.info("✓ Successfully repaired and parsed JSON")
                        except Exception as repair_error:
                            logger.error(f"JSON repair failed: {repair_error}")
                            raise e
                    else:
                        logger.error("Install json-repair for better error recovery: pip install json-repair")
                        raise e
            
            if parsed_data is None:
                raise ValueError("Failed to parse response with all available strategies")
                
            logger.info("✓ Successfully parsed Gemini response")
            return parsed_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            if debug_path and debug_path.exists():
                logger.error(f"Check raw response in: {debug_path}")
                # Try to identify the problematic area
                try:
                    with open(debug_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Show context around the error
                        if hasattr(e, 'pos') and e.pos:
                            start = max(0, e.pos - 100)
                            end = min(len(content), e.pos + 100)
                            logger.error(f"Error context: ...{content[start:end]}...")
                except Exception:
                    pass
            raise
        except Exception as e:
            logger.error(f"Google GenAI API error: {e}")
            raise
    
    def _clean_json_response(self, text: str) -> str:
        """
        Clean the JSON response from potential formatting issues.
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned JSON string
        """
        try:
            # Remove markdown code blocks
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'\s*```', '', text)
            
            # Remove any leading/trailing whitespace
            text = text.strip()
            
            return text
        
        except Exception as e:
            logger.warning(f"Error during JSON cleaning: {e}")
            return text


def get_llm_interface(provider: str, config: Config) -> LLMInterface:
    """
    Factory function to get the appropriate LLM interface.

    Args:
        provider: "openai" or "google"
        config: Configuration object

    Returns:
        LLM interface instance
        
    Raises:
        ValueError: If provider is unknown or configuration is invalid
    """
    try:
        if provider == "openai":
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment variables")
            
            return OpenAIInterface(
                api_key=config.OPENAI_API_KEY,
                model=config.OPENAI_MODEL,
                max_tokens=config.MAX_TOKENS
            )
        elif provider == "google":
            if not config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not set in environment variables")
            
            return GoogleInterface(
                api_key=config.GOOGLE_API_KEY,
                model=config.GOOGLE_MODEL,
                max_tokens=config.MAX_TOKENS
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'google'")
    
    except Exception as e:
        logger.error(f"Failed to initialize LLM interface: {e}")
        raise


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
        
    Raises:
        Various exceptions for different failure modes
    """
    logger.info("=" * 80)
    logger.info(f"Processing: {pdf_path.name}")
    logger.info("=" * 80)

    # Step 1: Extract text from PDF
    try:
        raw_text = extract_text_from_pdf(pdf_path, method=config.PDF_EXTRACTOR)
        
        if not raw_text or len(raw_text.strip()) < 100:
            raise ValueError("Extracted text is too short or empty (less than 100 characters)")
        
        logger.info(f"✓ Successfully extracted {len(raw_text)} characters")
        
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: {e}")
        raise
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        raise
    except ValueError as e:
        logger.error(f"PDF extraction validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise

    # Step 2: Initialize LLM interface
    try:
        llm = get_llm_interface(config.LLM_PROVIDER, config)
    except ValueError as e:
        logger.error(f"LLM configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize LLM interface: {e}")
        raise

    # Step 3: Parse manuscript with LLM
    debug_path = config.DEBUG_DIR / f"{pdf_path.stem}_raw_response.json"
    try:
        parsed_dict = llm.parse_manuscript(raw_text, debug_path=debug_path)
        
        if not parsed_dict:
            raise ValueError("LLM returned empty parsed data")
        
        logger.info("✓ Successfully received parsed data from LLM")
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.error(f"Check debug file for details: {debug_path}")
        raise
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
        logger.error(f"Parsed data structure does not match expected schema")
        raise

    # Step 5: Extract DOI from text (regex-based, no LLM)
    try:
        doi = extract_doi_from_text(raw_text)
        if doi:
            # Dynamically attach DOI to metadata without altering existing logic
            manuscript.metadata.doi = doi
            logger.info(f"✓ Extracted DOI from text: {doi}")
        else:
            logger.info("  - No DOI found in text")
    except Exception as e:
        logger.warning(f"DOI extraction failed (non-critical): {e}")
        # Don't raise - DOI extraction is optional
        
    # Step 6: Save to JSON file
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
        
        # Verify file was written correctly
        if not output_path.exists():
            raise IOError(f"Output file was not created: {output_path}")
        
        file_size = output_path.stat().st_size
        if file_size == 0:
            raise IOError(f"Output file is empty: {output_path}")
        
        logger.info(f"  - Output file size: {file_size} bytes")
        
    except PermissionError as e:
        logger.error(f"Permission denied when saving output: {e}")
        raise
    except IOError as e:
        logger.error(f"I/O error when saving output: {e}")
        raise
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

    logger.info("=" * 80)
    logger.info("CitePrism Phase 1 - Manuscript Parser")
    logger.info("=" * 80)

    # Initialize configuration
    try:
        config = Config()
        logger.info(f"Configuration loaded successfully")
        logger.info(f"  - LLM Provider: {config.LLM_PROVIDER}")
        logger.info(f"  - Model: {config.GOOGLE_MODEL if config.LLM_PROVIDER == 'google' else config.OPENAI_MODEL}")
        logger.info(f"  - Max Tokens: {config.MAX_TOKENS}")
        logger.info(f"  - PDF Extractor: {config.PDF_EXTRACTOR}")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Check dependencies
    try:
        if config.PDF_EXTRACTOR == "pypdf" and pypdf is None:
            logger.error("pypdf not installed. Run: pip install pypdf")
            sys.exit(1)
        elif config.PDF_EXTRACTOR == "pdfminer" and pdfminer_extract is None:
            logger.error("pdfminer.six not installed. Run: pip install pdfminer.six")
            sys.exit(1)

        if config.LLM_PROVIDER == "openai":
            if OpenAI is None:
                logger.error("openai not installed. Run: pip install openai")
                sys.exit(1)
            if not config.OPENAI_API_KEY:
                logger.error("OPENAI_API_KEY environment variable not set")
                sys.exit(1)
                
        elif config.LLM_PROVIDER == "google":
            if genai is None or types is None:
                logger.error("google-genai not installed. Run: pip install google-genai")
                sys.exit(1)
            if not config.GOOGLE_API_KEY:
                logger.error("GOOGLE_API_KEY environment variable not set")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        sys.exit(1)

    # Create directories
    try:
        config.INPUT_DIR.mkdir(exist_ok=True)
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        config.DEBUG_DIR.mkdir(exist_ok=True)
        logger.info("✓ Directory structure verified")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)

    # Find PDF files in input directory
    try:
        pdf_files = list(config.INPUT_DIR.glob("*.pdf"))
    except Exception as e:
        logger.error(f"Failed to scan input directory: {e}")
        sys.exit(1)

    if not pdf_files:
        logger.warning(f"No PDF files found in {config.INPUT_DIR}")
        logger.info(f"Please place PDF files in the '{config.INPUT_DIR}' directory")
        sys.exit(0)

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    logger.info("")

    # Process each PDF
    results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] Starting processing: {pdf_path.name}")
        output_path = config.OUTPUT_DIR / f"{pdf_path.stem}_parsed_2.5.json"

        try:
            manuscript = process_pdf(pdf_path, output_path, config)
            results.append({
                "file": pdf_path.name,
                "status": "success",
                "output": str(output_path)
            })
            logger.info(f"✓ [{i}/{len(pdf_files)}] Successfully processed: {pdf_path.name}")
            
        except FileNotFoundError as e:
            error_msg = f"File not found: {e}"
            logger.error(f"✗ [{i}/{len(pdf_files)}] {pdf_path.name}: {error_msg}")
            results.append({"file": pdf_path.name, "status": "failed", "error": error_msg})
            
        except PermissionError as e:
            error_msg = f"Permission denied: {e}"
            logger.error(f"✗ [{i}/{len(pdf_files)}] {pdf_path.name}: {error_msg}")
            results.append({"file": pdf_path.name, "status": "failed", "error": error_msg})
            
        except ValueError as e:
            error_msg = f"Validation error: {e}"
            logger.error(f"✗ [{i}/{len(pdf_files)}] {pdf_path.name}: {error_msg}")
            results.append({"file": pdf_path.name, "status": "failed", "error": error_msg})
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {e}"
            logger.error(f"✗ [{i}/{len(pdf_files)}] {pdf_path.name}: {error_msg}")
            results.append({"file": pdf_path.name, "status": "failed", "error": error_msg})
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ [{i}/{len(pdf_files)}] {pdf_path.name}: {error_msg}")
            results.append({"file": pdf_path.name, "status": "failed", "error": error_msg})
        
        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count
    
    logger.info(f"Total: {len(results)} file(s)")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("")

    if success_count > 0:
        logger.info("Successful files:")
        for result in results:
            if result["status"] == "success":
                logger.info(f"  ✓ {result['file']} → {result['output']}")
    
    if failed_count > 0:
        logger.info("")
        logger.info("Failed files:")
        for result in results:
            if result["status"] == "failed":
                logger.info(f"  ✗ {result['file']}")
                logger.info(f"    Error: {result['error']}")
    
    logger.info("=" * 80)
    
    # Exit with appropriate code
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()