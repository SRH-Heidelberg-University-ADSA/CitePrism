"""
Gemini Extractor - Wrapped for CitePrism Pipeline
Wraps the existing extractor logic into a class interface.
"""

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
# SYSTEM PROMPT
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
# HELPER FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_path: Path, method: str = "pypdf") -> str:
    """Extract raw text from a PDF file using specified method."""
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
                        continue

            if not text_parts:
                raise ValueError("No text could be extracted from any page in the PDF")
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from {len(text_parts)} pages")
            
            return full_text

        except ValueError:
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
            raise
        except Exception as e:
            logger.error(f"pdfminer extraction failed: {e}")
            raise Exception(f"PDF extraction error: {e}")

    else:
        raise ValueError(f"Unknown extraction method: {method}. Use 'pypdf' or 'pdfminer'")


def extract_doi_from_text(text: str) -> Optional[str]:
    """Extract the first DOI found in the given text using regex."""
    try:
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
# GEMINI EXTRACTOR CLASS (Main Interface)
# ============================================================================

class GeminiExtractor:
    """
    Gemini-based PDF extractor for CitePrism pipeline.
    Wraps existing extraction logic into a class interface.
    """
    
    def __init__(self, config):
        """
        Initialize the Gemini extractor.
        
        Args:
            config: Configuration object with API keys and settings
        """
        self.config = config
        
        # Get API key and model from config
        self.api_key = config.GOOGLE_API_KEY
        self.model = config.GOOGLE_MODEL
        self.max_tokens = config.MAX_TOKENS
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set in configuration")
        
        # Set up logs directory for debug output
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        logger.info(f"Debug logs will be saved to: {self.logs_dir}")
        logger.info(f"Max output tokens configured: {self.max_tokens}")
        
        # Initialize Gemini client
        try:
            
            self.client = genai.Client(
                api_key=self.api_key
            )
            
            logger.info(f"Initialized Google GenAI Client with model: {self.model}")
        except Exception as e:
            raise ValueError(f"Failed to initialize Google GenAI client: {e}")
    
    def extract(self, pdf_path: Path) -> Dict:
        """
        Extract structured data from PDF using Gemini LLM.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with parsed manuscript data
        """
        logger.info(f"Starting extraction for: {pdf_path.name}")
        
        # Step 1: Extract text from PDF
        try:
            raw_text = extract_text_from_pdf(pdf_path, method="pypdf")
            
            if not raw_text or len(raw_text.strip()) < 100:
                raise ValueError("Extracted text is too short or empty (less than 100 characters)")
            
            logger.info(f"[OK] Successfully extracted {len(raw_text)} characters")
            
            # Count approximate number of references in raw text for validation
            expected_ref_count = self._estimate_reference_count(raw_text)
            logger.info(f"Estimated {expected_ref_count} references in source document")
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
        
        # Step 2: Parse with Gemini LLM
        debug_path = self.logs_dir / f"{pdf_path.stem}_llm_response.json"
        try:
            parsed_dict = self._parse_with_gemini(raw_text, debug_path=debug_path)
            
            if not parsed_dict:
                raise ValueError("LLM returned empty parsed data")
            
            logger.info("[OK] Successfully received parsed data from LLM")
            
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            raise
        
        # Step 3: Validate with Pydantic
        try:
            parsed_dict = self._normalize_citations(parsed_dict)
            manuscript = ManuscriptStructure(**parsed_dict)
            logger.info("[OK] Output validation successful")
            logger.info(f"  - Title: {manuscript.metadata.title}")
            logger.info(f"  - Authors: {len(manuscript.metadata.authors)}")
            logger.info(f"  - Citations: {len(manuscript.citations_in_text)}")
            logger.info(f"  - References: {len(manuscript.references_list)}")
            
            # Step 3.5: Validate completeness
            self._validate_completeness(
                manuscript,
                expected_ref_count,
                pdf_path.name
            )
            
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            raise
        
        # Step 4: Extract DOI
        try:
            doi = extract_doi_from_text(raw_text)
            if doi:
                manuscript.metadata.doi = doi
                logger.info(f"[OK] Extracted DOI from text: {doi}")
        except Exception as e:
            logger.warning(f"DOI extraction failed (non-critical): {e}")
        
        # Return as dictionary
        return manuscript.model_dump()
    
    def _normalize_citations(self, data: dict) -> dict:
        """
        Ensure citations_in_text is a list of dicts, not ints.
        """
        citations = data.get("citations_in_text", [])

        fixed = []
        for item in citations:
            if isinstance(item, dict):
                fixed.append(item)
            elif isinstance(item, int):
                # salvage minimal structure
                fixed.append({
                    "marker": f"[{item}]",
                    "context_window": ""
                })
            else:
                # drop completely invalid entries
                continue

        data["citations_in_text"] = fixed
        return data
    
    def _estimate_reference_count(self, text: str) -> int:
        """
        Estimate the number of references in the document by looking for 
        common reference section patterns.
        
        Args:
            text: Raw manuscript text
            
        Returns:
            Estimated reference count
        """
        try:
            # Look for "References" or "Bibliography" section
            ref_section_pattern = r'(?i)(references|bibliography)\s*\n'
            match = re.search(ref_section_pattern, text)
            
            if match:
                # Get text after references section
                ref_section = text[match.end():]
                
                # Count numbered references like [1], [2], etc.
                numbered_refs = len(re.findall(r'^\s*\[\d+\]', ref_section, re.MULTILINE))
                
                # Count Author-Year style references
                author_year_refs = len(re.findall(r'^\s*\w+,\s+[A-Z]\..*?\(\d{4}\)', ref_section, re.MULTILINE))
                
                # Return the higher count
                estimated = max(numbered_refs, author_year_refs)
                
                if estimated > 0:
                    return estimated
            
            # Fallback: count citation markers in text
            citation_markers = len(re.findall(r'\[\d+\]', text))
            unique_markers = len(set(re.findall(r'\[(\d+)\]', text)))
            
            return max(unique_markers, citation_markers // 3)  # Conservative estimate
            
        except Exception as e:
            logger.warning(f"Error estimating reference count: {e}")
            return 0
    
    def _validate_completeness(
        self,
        manuscript: ManuscriptStructure,
        expected_ref_count: int,
        filename: str
    ):
        """
        Validate that the extraction is complete and comprehensive.
        
        Args:
            manuscript: Parsed manuscript structure
            expected_ref_count: Estimated number of references
            filename: Name of the PDF file being processed
        """
        actual_ref_count = len(manuscript.references_list)
        citation_count = len(manuscript.citations_in_text)
        
        # Check 1: Reference count validation
        if expected_ref_count > 0:
            completeness_ratio = actual_ref_count / expected_ref_count
            
            if completeness_ratio < 0.5:
                logger.error(
                    f"⚠️ CRITICAL: Only {actual_ref_count}/{expected_ref_count} references extracted "
                    f"({completeness_ratio:.1%} completeness)"
                )
                logger.error(f"⚠️ Response appears to be INCOMPLETE or TRUNCATED!")
                
            elif completeness_ratio < 0.8:
                logger.warning(
                    f"⚠️ WARNING: Only {actual_ref_count}/{expected_ref_count} references extracted "
                    f"({completeness_ratio:.1%} completeness)"
                )
                logger.warning(f"⚠️ Some references may be missing")
                
            else:
                logger.info(
                    f"[OK] Reference extraction looks complete: {actual_ref_count}/{expected_ref_count} "
                    f"({completeness_ratio:.1%})"
                )
        
        # Check 2: Minimum reference count sanity check
        if actual_ref_count < 5:
            logger.warning(
                f"⚠️ WARNING: Very few references extracted ({actual_ref_count}). "
                f"This may indicate an incomplete extraction."
            )
        
        # Check 3: Citation vs Reference mismatch
        if citation_count > actual_ref_count * 1.5:
            logger.warning(
                f"⚠️ WARNING: More citations ({citation_count}) than references ({actual_ref_count}). "
                f"Some references may be missing from the reference list."
            )
        
        # Check 4: Metadata completeness
        if not manuscript.metadata.title:
            logger.warning("⚠️ WARNING: No title extracted")
        
        if not manuscript.metadata.authors:
            logger.warning("⚠️ WARNING: No authors extracted")
        
        if not manuscript.metadata.abstract:
            logger.warning("⚠️ WARNING: No abstract extracted")
        
        # Save validation report to logs
        validation_report = {
            "filename": filename,
            "expected_references": expected_ref_count,
            "extracted_references": actual_ref_count,
            "completeness_ratio": actual_ref_count / expected_ref_count if expected_ref_count > 0 else 0,
            "citations_extracted": citation_count,
            "has_title": manuscript.metadata.title is not None,
            "has_authors": len(manuscript.metadata.authors) > 0,
            "has_abstract": manuscript.metadata.abstract is not None,
            "validation_status": "COMPLETE" if actual_ref_count >= expected_ref_count * 0.8 else "INCOMPLETE"
        }
        
        validation_path = self.logs_dir / f"{Path(filename).stem}_validation.json"
        try:
            with open(validation_path, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, indent=2)
            logger.info(f"Saved validation report to: {validation_path}")
        except Exception as e:
            logger.warning(f"Failed to save validation report: {e}")
    
    def _parse_with_gemini(self, text: str, debug_path: Optional[Path] = None) -> Dict:
        """
        Parse manuscript text using Gemini LLM.
        Streamlit-safe version:
        - No response_schema (prevents timeout)
        - Post-hoc Pydantic validation (original behavior)
        """
        logger.info("Sending request to Google Gemini API...")
        logger.info(f"Input text length: {len(text)} characters")
        logger.info(f"Max output tokens: {self.max_tokens}")

        try:
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
                    temperature=0.1,
                    max_output_tokens=self.max_tokens
                )
            )

            if not response or not response.text:
                raise ValueError("Google GenAI returned empty response")

            # Save raw response for debugging
            if debug_path:
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                logger.info(f"[OK] Saved raw LLM response to: {debug_path}")
                logger.info(f"  Response length: {len(response.text)} characters")

            # Log truncation info (important for long papers)
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
                if finish_reason == 2:
                    logger.error("⚠️ CRITICAL: Gemini response truncated (MAX_TOKENS)")
                elif finish_reason == 1:
                    logger.info("[OK] Gemini response completed normally")

            # ---------- JSON PARSING (ORIGINAL STRATEGY) ----------
            try:
                clean_text = self._clean_json_response(response.text)
                parsed_dict = json.loads(clean_text)

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")

                if HAS_JSON_REPAIR:
                    logger.info("Attempting JSON repair...")
                    repaired = repair_json(clean_text)
                    parsed_dict = json.loads(repaired)
                    logger.info("[OK] JSON repaired successfully")
                else:
                    raise

            return parsed_dict

        except Exception as e:
            logger.error(f"Gemini parsing failed: {e}")
            raise

    
    def _clean_json_response(self, text: str) -> str:
        """Clean JSON response from potential formatting issues."""
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