"""
CitePrism Gemini Extractor - Pipeline Integration
==================================================
Self-contained extractor with all logic from extractor_new_2.py
Provides a clean interface for the pipeline orchestrator.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field

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
# SYSTEM PROMPT (EXACTLY AS SPECIFIED IN ORIGINAL)
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
# PDF EXTRACTION (FROM ORIGINAL)
# ============================================================================

def extract_text_from_pdf(pdf_path: Path, method: str = "pypdf") -> str:
    """Extract raw text from a PDF file using specified method."""
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
# DOI EXTRACTION (FROM ORIGINAL)
# ============================================================================

def extract_doi_from_text(text: str) -> Optional[str]:
    """Extract the first DOI found in the given text using regex."""
    doi_pattern = re.compile(
        r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b',
        re.IGNORECASE
    )
    match = doi_pattern.search(text)
    return match.group(0) if match else None


# ============================================================================
# LLM INTERFACES (FROM ORIGINAL)
# ============================================================================

class GoogleInterface:
    """Google Gemini LLM interface."""
    
    def __init__(self, api_key: str, model: str, max_tokens: int):
        if genai is None:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model,
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
                "max_output_tokens": max_tokens
            }
        )
        self.max_tokens = max_tokens
        logger.info(f"Initialized Google Gemini client with model: {model}")
        logger.info(f"Max output tokens: {max_tokens}")
    
    def _repair_truncated_json(self, json_text: str) -> str:
        """Attempt to repair truncated JSON by closing incomplete structures."""
        # Count opening and closing brackets
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        open_brackets = json_text.count('[')
        close_brackets = json_text.count(']')
        
        # Find last complete position
        last_complete_obj = json_text.rfind('},')
        last_complete_arr = json_text.rfind('],')
        
        # Truncate at last complete structure
        if last_complete_obj > last_complete_arr and last_complete_obj > 0:
            json_text = json_text[:last_complete_obj + 1]
        elif last_complete_arr > 0:
            json_text = json_text[:last_complete_arr + 1]
        
        # Close any unclosed strings (look for odd number of quotes in last line)
        lines = json_text.split('\n')
        if lines:
            last_line = lines[-1]
            quote_count = last_line.count('"')
            if quote_count % 2 == 1:  # Odd number means unclosed string
                # Remove the incomplete line
                json_text = '\n'.join(lines[:-1])
        
        # Close arrays and objects
        while open_brackets > close_brackets:
            json_text += '\n]'
            close_brackets += 1
        
        while open_braces > close_braces:
            json_text += '\n}'
            close_braces += 1
        
        return json_text
    
    def parse_manuscript(self, text: str, debug_path: Optional[Path] = None) -> Dict:
        """Parse manuscript using Google Gemini."""
        logger.info("Sending request to Google Gemini API...")
        
        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\nParse this manuscript:\n\n{text}"
            response = self.model.generate_content(full_prompt)
            json_text = response.text
            
            # Save raw response for debugging
            if debug_path:
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(json_text)
                logger.info(f"Saved raw LLM response to: {debug_path}")
            
            # Try to parse JSON
            try:
                parsed_data = json.loads(json_text)
                logger.info("✓ Successfully received and parsed Gemini response")
                return parsed_data
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}. Attempting to repair truncated JSON...")
                
                # Attempt to repair
                repaired_json = self._repair_truncated_json(json_text)
                
                # Save repaired version
                if debug_path:
                    repair_path = debug_path.parent / f"{debug_path.stem}_repaired.json"
                    with open(repair_path, 'w', encoding='utf-8') as f:
                        f.write(repaired_json)
                    logger.info(f"Saved repaired JSON to: {repair_path}")
                
                # Try parsing repaired JSON
                try:
                    parsed_data = json.loads(repaired_json)
                    logger.info("✓ Successfully parsed repaired JSON")
                    return parsed_data
                except json.JSONDecodeError as e2:
                    logger.error(f"Still failed after repair: {e2}")
                    raise Exception(
                        f"Gemini response is truncated or malformed. "
                        f"Original error: {e}. "
                        f"Repair failed: {e2}. "
                        f"Try increasing MAX_TOKENS in config or use a longer context model."
                    )
        
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            raise


class OpenAIInterface:
    """OpenAI ChatGPT interface."""
    
    def __init__(self, api_key: str, model: str, max_tokens: int):
        if OpenAI is None:
            raise ImportError("openai not installed. Run: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        logger.info(f"Initialized OpenAI client with model: {model}")
    
    def parse_manuscript(self, text: str, debug_path: Optional[Path] = None) -> Dict:
        """Parse manuscript using OpenAI."""
        logger.info("Sending request to OpenAI API...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Parse this manuscript:\n\n{text}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                temperature=0.1
            )
            
            json_text = response.choices[0].message.content
            
            if debug_path:
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, 'w', encoding='utf-8') as f:
                    f.write(json_text)
                logger.info(f"Saved raw LLM response to: {debug_path}")
            
            parsed_data = json.loads(json_text)
            logger.info("✓ Successfully received and parsed OpenAI response")
            return parsed_data
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


# ============================================================================
# MAIN EXTRACTOR CLASS (PIPELINE INTERFACE)
# ============================================================================

class GeminiExtractor:
    """
    Self-contained Gemini extractor for CitePrism pipeline.
    Contains all logic from extractor_new_2.py.
    """
    
    def __init__(self, config):
        """Initialize the Gemini extractor."""
        self.config = config
        
        # Get configuration
        self.provider = getattr(config, 'LLM_PROVIDER', 'google')
        self.model = getattr(config, 'GOOGLE_MODEL', 'gemini-2.0-flash-exp')
        self.max_tokens = getattr(config, 'MAX_TOKENS', 100000)  # Increased default for longer papers
        self.pdf_extractor = getattr(config, 'PDF_EXTRACTOR', 'pypdf')
        
        # API keys
        self.google_api_key = getattr(config, 'GOOGLE_API_KEY', None)
        self.openai_api_key = getattr(config, 'OPENAI_API_KEY', None)
        
        # Debug directory
        self.debug_dir = Path("data/debug/extractor_responses")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"GeminiExtractor initialized:")
        logger.info(f"  - Provider: {self.provider}")
        logger.info(f"  - Model: {self.model}")
        logger.info(f"  - Max tokens: {self.max_tokens}")
    
    def _get_llm_interface(self):
        """Get the appropriate LLM interface."""
        if self.provider == "openai":
            return OpenAIInterface(
                api_key=self.openai_api_key,
                model=self.model,
                max_tokens=self.max_tokens
            )
        elif self.provider == "google":
            return GoogleInterface(
                api_key=self.google_api_key,
                model=self.model,
                max_tokens=self.max_tokens
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
    
    def extract(self, pdf_path: Path, progress_bar=None) -> Dict:
        """
        Extract structured data from a PDF using LLM.
        
        Args:
            pdf_path: Path to the PDF file
            progress_bar: Optional Streamlit progress bar object
            
        Returns:
            Dictionary with parsed manuscript data
        """
        logger.info("=" * 80)
        logger.info(f"Processing: {pdf_path.name}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Extract text from PDF
            if progress_bar:
                progress_bar.progress(0.10, text="Extracting text from PDF...")
            
            logger.info(f"Step 1: Extracting text using {self.pdf_extractor}...")
            raw_text = extract_text_from_pdf(pdf_path, method=self.pdf_extractor)
            
            if not raw_text or len(raw_text.strip()) < 100:
                raise ValueError("Extracted text is too short or empty")
            
            logger.info(f"  ✓ Extracted {len(raw_text)} characters")
            
            # Step 2: Initialize LLM interface
            if progress_bar:
                progress_bar.progress(0.15, text="Connecting to LLM API...")
            
            logger.info(f"Step 2: Initializing {self.provider} LLM interface...")
            llm = self._get_llm_interface()
            
            # Step 3: Parse manuscript with LLM
            if progress_bar:
                progress_bar.progress(0.20, text="Analyzing manuscript with LLM (this may take 30-60s)...")
            
            logger.info("Step 3: Sending text to LLM for parsing...")
            debug_path = self.debug_dir / f"{pdf_path.stem}_raw_response.json"
            
            parsed_dict = llm.parse_manuscript(raw_text, debug_path=debug_path)
            logger.info("  ✓ Successfully received LLM response")
            
            # Step 4: Validate output with Pydantic
            if progress_bar:
                progress_bar.progress(0.35, text="Validating extracted data...")
            
            logger.info("Step 4: Validating parsed data...")
            manuscript = ManuscriptStructure(**parsed_dict)
            
            logger.info("  ✓ Output validation successful")
            logger.info(f"    - Title: {manuscript.metadata.title}")
            logger.info(f"    - Authors: {len(manuscript.metadata.authors)}")
            logger.info(f"    - Citations: {len(manuscript.citations_in_text)}")
            logger.info(f"    - References: {len(manuscript.references_list)}")
            
            # Step 5: Extract DOI from text (regex-based)
            if progress_bar:
                progress_bar.progress(0.38, text="Extracting metadata...")
            
            logger.info("Step 5: Extracting DOI from text...")
            doi = extract_doi_from_text(raw_text)
            if doi:
                manuscript.metadata.__dict__["doi"] = doi
                logger.info(f"  ✓ Extracted DOI: {doi}")
            else:
                logger.info("  ! No DOI found in text")
            
            # Step 6: Convert to pipeline-compatible format
            if progress_bar:
                progress_bar.progress(0.40, text="Finalizing parsed data...")
            
            logger.info("Step 6: Converting to pipeline format...")
            result = manuscript.model_dump()
            
            logger.info("=" * 80)
            logger.info("EXTRACTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            if progress_bar:
                progress_bar.progress(0.40, text=f"Extraction failed: {str(e)[:50]}...")
            raise
    
    def extract_with_cache(self, pdf_path: Path, cache_key: str, 
                          db_manager, progress_bar=None, 
                          force_reprocess: bool = False) -> Dict:
        """
        Extract with intelligent caching support.
        
        Args:
            pdf_path: Path to PDF file
            cache_key: Unique cache identifier
            db_manager: Database manager for caching
            progress_bar: Optional progress bar
            force_reprocess: If True, ignore cache
            
        Returns:
            Parsed manuscript data
        """
        # Check cache first (unless force_reprocess is True)
        if not force_reprocess:
            cached = db_manager.get_cached_response('parsing', cache_key)
            if cached:
                logger.info(f"✓ Using cached parsing result for {pdf_path.name}")
                if progress_bar:
                    progress_bar.progress(0.40, text="Using cached parsed data...")
                return cached
        
        # If not cached or force reprocess, extract fresh
        logger.info(f"{'Force re-parsing' if force_reprocess else 'Cache miss - parsing'} {pdf_path.name}")
        result = self.extract(pdf_path, progress_bar)
        
        # Cache the result
        db_manager.cache_api_response('parsing', cache_key, result)
        logger.info(f"  ✓ Cached parsing result for future use")
        
        return result