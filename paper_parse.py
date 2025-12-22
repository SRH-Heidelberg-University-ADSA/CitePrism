from __future__ import annotations

import fitz
import os
import json
import re
import logging
import statistics
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import time

from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# SETUP
# =====================================================

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('citeprism.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ROOT = Path.cwd()
PDF_DIR = ROOT / "research_papers"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# CONFIGURATION FROM .ENV
# =====================================================

LLM_BACKEND = os.getenv("LLM_BACKEND", "hf").strip().lower()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()

# Hugging Face Configuration
HF_API_KEY = os.getenv("HF_API_KEY", "").strip()
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-2b-it").strip()
HF_ENDPOINT = "https://router.huggingface.co/hf-inference"

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()

# =====================================================
# DATA MODELS
# =====================================================

@dataclass
class Section:
    title: str
    content: str
    start_page: int
    end_page: int

@dataclass
class Citation:
    number: int
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    title: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None

@dataclass
class Author:
    name: str
    affiliation: str = ""
    email: Optional[str] = None

@dataclass
class PaperMetadata:
    title: str = ""
    authors: List[Author] = field(default_factory=list)
    journal: str = ""
    volume: Optional[str] = None
    issue: Optional[str] = None
    year: Optional[int] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    received_date: Optional[str] = None
    accepted_date: Optional[str] = None
    available_online: Optional[str] = None

# =====================================================
# UTILITIES
# =====================================================

def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    try:
        vec = TfidfVectorizer().fit_transform([a, b])
        return cosine_similarity(vec[0], vec[1])[0][0]
    except Exception:
        return 0.0

def clean_json_response(text: str) -> str:
    """Clean up the response to extract valid JSON"""
    if not text:
        return "{}"
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Find JSON object
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx >= 0 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]
    
    # Fix common JSON issues
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    return text

def extract_email(text: str) -> Optional[str]:
    """Extract email address from text"""
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    return match.group() if match else None

# =====================================================
# ADVANCED HEURISTIC PARSER
# =====================================================

class AdvancedHeuristicParser:
    def __init__(self):
        self.ABSTRACT_ALIASES = {"abstract", "summary", "overview"}
        self.INTRO_RE = re.compile(r'^\s*(\d+\.?\s+)?introduction\b', re.I)
        self.SECTION_RE = re.compile(r'^\s*(\d+(\.\d+)*)\s+(.+)$')
        self.CITATION_RE = re.compile(r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]')
        self.AUTHOR_RE = re.compile(r'([A-Z][a-zA-Z\s\-]+(?:,\s*[A-Z][a-zA-Z\s\-]+)*)')
        self.AFFILIATION_RE = re.compile(r'^[a-z]\s*([A-Z].+?)(?:,\s*[A-Z].+?)*$')
        self.DOI_RE = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.I)
        self.YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
        
    def parse(self, pdf: Path) -> Dict[str, Any]:
        """Parse PDF using advanced heuristic rules"""
        doc = fitz.open(pdf)
        full_text = ""
        lines_with_metadata = []
        
        # Extract text with formatting and position information
        for p in range(min(len(doc), 10)):  # Limit to first 10 pages for speed
            page = doc[p]
            blocks = page.get_text("dict").get("blocks", [])
            
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    line_text = " ".join(
                        span["text"].strip()
                        for span in line["spans"]
                        if span["text"].strip()
                    )
                    if line_text:
                        try:
                            avg_size = statistics.mean(span["size"] for span in line["spans"])
                            is_bold = any(span.get("flags", 0) & 2**4 for span in line["spans"])  # Bold flag
                            lines_with_metadata.append({
                                "page": p,
                                "text": line_text,
                                "size": avg_size,
                                "is_bold": is_bold,
                                "bbox": line.get("bbox", [0, 0, 0, 0])
                            })
                            full_text += line_text + "\n"
                        except:
                            lines_with_metadata.append({
                                "page": p,
                                "text": line_text,
                                "size": 10,
                                "is_bold": False,
                                "bbox": [0, 0, 0, 0]
                            })
                            full_text += line_text + "\n"
        
        # Extract metadata from first page
        first_page_lines = [l for l in lines_with_metadata if l["page"] == 0]
        
        # Parse metadata
        metadata = self._extract_metadata(first_page_lines, full_text)
        
        # Extract abstract
        abstract = self._extract_abstract_advanced(lines_with_metadata)
        
        # Extract sections
        sections = self._extract_sections_advanced(lines_with_metadata, len(doc))
        
        # Extract keywords
        keywords = self._extract_keywords(first_page_lines)
        
        # Extract citations (simplified - from references section)
        citations = self._extract_citations_from_text(full_text)
        
        return {
            "metadata": metadata,
            "abstract": abstract,
            "sections": sections,
            "keywords": keywords,
            "citations_count": len(citations),
            "total_pages": len(doc)
        }
    
    def _extract_metadata(self, first_page_lines: List[Dict], full_text: str) -> Dict[str, Any]:
        """Extract paper metadata from first page"""
        metadata = {
            "title": "",
            "authors": [],
            "journal": "",
            "volume": "",
            "issue": "",
            "year": None,
            "pages": "",
            "doi": "",
            "received_date": "",
            "accepted_date": "",
            "available_online": ""
        }
        
        # Look for title (usually largest text on first page)
        if first_page_lines:
            # Title is often the largest text on page 0
            title_candidates = sorted(
                [line for line in first_page_lines if line["size"] > 12],
                key=lambda x: x["size"],
                reverse=True
            )
            if title_candidates:
                metadata["title"] = title_candidates[0]["text"]
        
        # Look for journal information (often contains "Journal", "Proceedings", etc.)
        journal_patterns = [r'Journal\s+[A-Z]', r'Proceedings', r'Conference', r'Transactions']
        for line in first_page_lines:
            for pattern in journal_patterns:
                if re.search(pattern, line["text"], re.I):
                    metadata["journal"] = line["text"]
                    break
        
        # Look for authors (names followed by affiliations with superscripts)
        authors_text = ""
        for i, line in enumerate(first_page_lines):
            if re.search(r'@|\.edu|\.ac\.|university|institute', line["text"], re.I):
                # This might be author/affiliation line
                authors_text = first_page_lines[i-1]["text"] if i > 0 else line["text"]
                break
        
        if authors_text:
            # Simple parsing - split by comma and clean
            authors = [a.strip() for a in authors_text.split(',') if a.strip()]
            metadata["authors"] = [{"name": author, "affiliation": "", "email": ""} for author in authors[:5]]
        
        # Look for DOI
        doi_match = self.DOI_RE.search(full_text[:2000])
        if doi_match:
            metadata["doi"] = doi_match.group()
        
        # Look for year
        year_match = self.YEAR_RE.search(full_text[:1000])
        if year_match:
            try:
                metadata["year"] = int(year_match.group())
            except:
                pass
        
        # Look for dates
        date_patterns = [
            r'Received\s+(\d+\s+\w+\s+\d{4})',
            r'Accepted\s+(\d+\s+\w+\s+\d{4})',
            r'Available online\s+(\d+\s+\w+\s+\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, full_text, re.I)
            if match:
                if "Received" in pattern:
                    metadata["received_date"] = match.group(1)
                elif "Accepted" in pattern:
                    metadata["accepted_date"] = match.group(1)
                elif "Available online" in pattern:
                    metadata["available_online"] = match.group(1)
        
        return metadata
    
    def _extract_abstract_advanced(self, lines: List[Dict]) -> str:
        """Extract abstract using multiple strategies"""
        abstract_lines = []
        in_abstract = False
        found_abstract_header = False
        
        for line in lines[:100]:  # Check first 100 lines
            text = line["text"].lower()
            
            # Check for abstract header
            if any(alias in text for alias in self.ABSTRACT_ALIASES) and not found_abstract_header:
                in_abstract = True
                found_abstract_header = True
                continue
            
            # Check for end of abstract (usually keywords or introduction)
            if in_abstract:
                if "keywords" in text or "key words" in text or self.INTRO_RE.match(text):
                    break
                if text.strip() and not text.startswith(('1.', '1 ', 'fig', 'table')):
                    abstract_lines.append(line["text"])
        
        if abstract_lines:
            return " ".join(abstract_lines)
        
        # Fallback: first substantial paragraph after title
        word_count = 0
        abstract_fallback = []
        for line in lines[:50]:  # Check first 50 lines
            if line["page"] == 0 and line["text"].strip():
                words = line["text"].split()
                word_count += len(words)
                abstract_fallback.append(line["text"])
                if word_count >= 150:
                    break
        
        return " ".join(abstract_fallback) if abstract_fallback else ""
    
    def _extract_sections_advanced(self, lines: List[Dict], total_pages: int) -> List[Dict[str, Any]]:
        """Extract sections with improved detection"""
        sections = []
        current_section = None
        current_content = []
        current_start_page = 0
        
        for i, line in enumerate(lines):
            text = line["text"].strip()
            page = line["page"]
            
            # Check if this is a section header
            is_header = False
            header_text = ""
            
            # Pattern 1: Numbered sections (1., 1.1, 2., etc.)
            numbered_match = self.SECTION_RE.match(text)
            if numbered_match:
                is_header = True
                header_text = text
            
            # Pattern 2: Common section titles
            common_sections = [
                'introduction', 'methodology', 'methods', 'results',
                'discussion', 'conclusion', 'references', 'acknowledgements',
                'abstract', 'keywords'
            ]
            
            text_lower = text.lower()
            if any(section in text_lower for section in common_sections):
                is_header = True
                header_text = text
            
            # Pattern 3: Large/bold text that's not too long
            if (line["size"] > 13 or line["is_bold"]) and len(text.split()) < 15 and not text.endswith('.'):
                is_header = True
                header_text = text
            
            if is_header and header_text:
                # Save previous section
                if current_section and current_content:
                    sections.append({
                        "title": current_section,
                        "content": " ".join(current_content),
                        "start_page": current_start_page,
                        "end_page": lines[i-1]["page"] if i > 0 else 0
                    })
                
                # Start new section
                current_section = header_text
                current_content = []
                current_start_page = page
            elif current_section:
                # Add content to current section
                current_content.append(text)
        
        # Add the last section
        if current_section and current_content:
            sections.append({
                "title": current_section,
                "content": " ".join(current_content),
                "start_page": current_start_page,
                "end_page": total_pages - 1
            })
        
        return sections
    
    def _extract_keywords(self, first_page_lines: List[Dict]) -> List[str]:
        """Extract keywords from first page"""
        keywords = []
        found_keywords = False
        
        for line in first_page_lines:
            text = line["text"].lower()
            
            if "keywords" in text or "key words" in text:
                found_keywords = True
                # Extract the actual keywords (text after "Keywords:")
                keyword_text = re.sub(r'.*keywords?\s*:\s*', '', text, flags=re.I)
                if keyword_text:
                    # Split by commas, semicolons, or other delimiters
                    keywords = [k.strip() for k in re.split(r'[;,•]', keyword_text) if k.strip()]
                break
        
        # Fallback: look for common keyword patterns
        if not keywords:
            for line in first_page_lines:
                text = line["text"]
                # Look for comma-separated lists of 2-4 word phrases
                if ',' in text and len(text.split()) < 30:
                    potential_keywords = [k.strip() for k in text.split(',')]
                    if all(len(k.split()) <= 4 for k in potential_keywords):
                        keywords = potential_keywords[:10]  # Limit to 10
                        break
        
        return keywords
    
    def _extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract citation references from text (simplified)"""
        citations = []
        citation_patterns = [
            r'\[\s*(\d+)\s*\]',  # [1], [2], etc.
            r'\(\s*(\d+)\s*\)',  # (1), (2), etc.
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    citation_num = int(match.group(1))
                    if citation_num > 0:
                        citations.append({
                            "number": citation_num,
                            "text": f"[{citation_num}]"
                        })
                except:
                    continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            if citation["number"] not in seen:
                seen.add(citation["number"])
                unique_citations.append(citation)
        
        return sorted(unique_citations, key=lambda x: x["number"])

# =====================================================
# BLOCK EXTRACTION FOR LLM
# =====================================================

def extract_blocks_for_llm(pdf: Path, max_pages: int = 5) -> List[Dict[str, Any]]:
    """Extract structured blocks for LLM processing"""
    doc = fitz.open(pdf)
    blocks = []
    
    for p in range(min(len(doc), max_pages)):
        page = doc[p]
        text_blocks = page.get_text("blocks")  # Get text blocks with coordinates
        
        for block in text_blocks:
            # block format: (x0, y0, x1, y1, text, block_no, block_type)
            if len(block) >= 6:
                text = block[4].strip()
                if text and len(text) > 10:  # Skip very short blocks
                    blocks.append({
                        "page": p,
                        "text": text,
                        "bbox": block[:4],
                        "block_type": "text" if block[6] == 0 else "image" if block[6] == 1 else "other"
                    })
    
    return blocks[:30]  # Limit to 30 blocks to avoid token limits

# =====================================================
# LLM PARSERS
# =====================================================

def create_llm_prompt(blocks: List[Dict[str, Any]], paper_text_sample: str = "") -> str:
    """Create prompt for LLM parsing"""
    
    sample_text = paper_text_sample[:2000] if paper_text_sample else ""
    blocks_str = json.dumps(blocks[:15], indent=2)  # Limit blocks
    
    return f"""You are an expert research paper parser. Extract structured information from the provided paper content.

CRITICAL INSTRUCTIONS:
1. Return ONLY valid JSON, no other text
2. Use null for missing information
3. Page numbers are 0-indexed (first page = 0)

REQUIRED OUTPUT FORMAT:
{{
  "metadata": {{
    "title": "string or null",
    "authors": [{{"name": "string", "affiliation": "string", "email": "string or null"}}],
    "journal": "string or null",
    "year": "integer or null",
    "doi": "string or null",
    "received_date": "string or null",
    "accepted_date": "string or null",
    "available_online": "string or null"
  }},
  "abstract": "string or null",
  "keywords": ["string1", "string2", ...],
  "sections": [
    {{
      "title": "string",
      "content": "string",
      "start_page": integer,
      "end_page": integer
    }}
  ],
  "summary": {{
    "total_pages": integer,
    "section_count": integer,
    "has_abstract": boolean,
    "has_keywords": boolean
  }}
}}

PAPER CONTENT SAMPLE:
{sample_text}

STRUCTURED BLOCKS:
{blocks_str}

OUTPUT JSON:"""

def llm_parse_huggingface(blocks: List[Dict[str, Any]], paper_text: str = "") -> Dict[str, Any]:
    """Parse using Hugging Face Inference API"""
    if not HF_API_KEY:
        logger.error("Hugging Face API key not configured")
        return {"error": "API key not configured"}
    
    prompt = create_llm_prompt(blocks, paper_text)
    
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Try different endpoints
    endpoints = [
        (HF_ENDPOINT, {"model": HF_MODEL}),
        (f"https://api-inference.huggingface.co/models/{HF_MODEL}", {})
    ]
    
    for endpoint, extra_params in endpoints:
        try:
            logger.info(f"Trying endpoint: {endpoint}")
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1500,
                    "temperature": 0.1,
                    "return_full_text": False,
                    **extra_params
                }
            }
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse response
                if isinstance(result, list):
                    text = result[0].get("generated_text", "") if result else ""
                elif isinstance(result, dict):
                    text = result.get("generated_text", "")
                else:
                    text = str(result)
                
                # Clean and parse JSON
                cleaned_text = clean_json_response(text)
                try:
                    return json.loads(cleaned_text)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {e}")
                    logger.debug(f"Raw response: {text[:500]}")
                    
                    # Try to extract JSON
                    json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group())
                        except:
                            pass
                
                # If we got here but response was 200, at least we tried
                return {"error": "Failed to parse LLM response", "raw_response": text[:500]}
            
            elif response.status_code in [503, 429]:
                logger.warning(f"Endpoint busy/loading: {response.status_code}")
                time.sleep(2)
                continue
            else:
                logger.warning(f"Endpoint failed: {response.status_code} - {response.text[:200]}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request to {endpoint} failed: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error with {endpoint}: {e}")
            continue
    
    logger.error("All Hugging Face endpoints failed")
    return {"error": "All API endpoints failed"}

def llm_parse_openai(blocks: List[Dict[str, Any]], paper_text: str = "") -> Dict[str, Any]:
    """Parse using OpenAI API"""
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not configured")
        return {"error": "API key not configured"}
    
    prompt = create_llm_prompt(blocks, paper_text)
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a research paper parser. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
        else:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text[:200]}")
            return {"error": f"API error: {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenAI request failed: {e}")
        return {"error": f"Request failed: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse OpenAI response: {e}")
        return {"error": "Failed to parse response"}

def llm_parse_ollama(blocks: List[Dict[str, Any]], paper_text: str = "") -> Dict[str, Any]:
    """Parse using local Ollama"""
    prompt = create_llm_prompt(blocks, paper_text)
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 2000
        },
        "format": "json"
    }
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=150
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("response", "")
            cleaned = clean_json_response(content)
            return json.loads(cleaned)
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text[:200]}")
            return {"error": f"API error: {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request failed: {e}")
        return {"error": f"Request failed: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama response: {e}")
        return {"error": "Failed to parse response"}

def llm_parse(blocks: List[Dict[str, Any]], paper_text: str = "", backend: str = None) -> Dict[str, Any]:
    """Main LLM parser that routes to appropriate backend"""
    if backend is None:
        backend = LLM_BACKEND
    
    logger.info(f"Using LLM backend: {backend}")
    
    if backend == "openai":
        return llm_parse_openai(blocks, paper_text)
    elif backend == "ollama":
        return llm_parse_ollama(blocks, paper_text)
    elif backend == "hf":
        return llm_parse_huggingface(blocks, paper_text)
    else:
        logger.error(f"Unknown LLM backend: {backend}")
        return {"error": f"Unknown backend: {backend}"}

# =====================================================
# MERGER AND VALIDATOR
# =====================================================

def merge_and_validate_results(heuristic_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
    """Merge heuristic and LLM results with validation"""
    
    # Initialize merged result
    merged = {
        "metadata": {},
        "abstract": "",
        "keywords": [],
        "sections": [],
        "processing_info": {
            "heuristic_used": True,
            "llm_used": "error" not in llm_result,
            "merge_strategy": "preference_based",
            "confidence": "medium"
        }
    }
    
    # Helper function to get best value
    def get_best_value(h_val, l_val, field_name=""):
        if field_name in ["title", "abstract"]:
            # For important fields, prefer longer, more complete text
            if l_val and isinstance(l_val, str) and len(l_val.strip()) > 20:
                return l_val
            return h_val or l_val or ""
        elif field_name == "keywords":
            # Prefer non-empty list
            if l_val and isinstance(l_val, list) and len(l_val) > 0:
                return l_val
            return h_val or l_val or []
        elif field_name == "sections":
            # Prefer LLM sections if they look reasonable
            if (l_val and isinstance(l_val, list) and len(l_val) > 1 and 
                all("title" in s and "content" in s for s in l_val)):
                return l_val
            return h_val or l_val or []
        else:
            return l_val or h_val or ""
    
    # Merge metadata
    h_meta = heuristic_result.get("metadata", {})
    l_meta = llm_result.get("metadata", {}) if "error" not in llm_result else {}
    
    merged["metadata"] = {
        "title": get_best_value(h_meta.get("title", ""), l_meta.get("title", ""), "title"),
        "authors": l_meta.get("authors", []) or h_meta.get("authors", []),
        "journal": l_meta.get("journal", "") or h_meta.get("journal", ""),
        "year": l_meta.get("year") or h_meta.get("year"),
        "doi": l_meta.get("doi", "") or h_meta.get("doi", ""),
        "received_date": l_meta.get("received_date", "") or h_meta.get("received_date", ""),
        "accepted_date": l_meta.get("accepted_date", "") or h_meta.get("accepted_date", ""),
        "available_online": l_meta.get("available_online", "") or h_meta.get("available_online", "")
    }
    
    # Merge abstract
    h_abstract = heuristic_result.get("abstract", "")
    l_abstract = llm_result.get("abstract", "") if "error" not in llm_result else ""
    merged["abstract"] = get_best_value(h_abstract, l_abstract, "abstract")
    
    # Merge keywords
    h_keywords = heuristic_result.get("keywords", [])
    l_keywords = llm_result.get("keywords", []) if "error" not in llm_result else []
    merged["keywords"] = get_best_value(h_keywords, l_keywords, "keywords")
    
    # Merge sections
    h_sections = heuristic_result.get("sections", [])
    l_sections = llm_result.get("sections", []) if "error" not in llm_result else []
    merged["sections"] = get_best_value(h_sections, l_sections, "sections")
    
    # Add statistics
    merged["statistics"] = {
        "total_pages": heuristic_result.get("total_pages", 0),
        "citations_count": heuristic_result.get("citations_count", 0),
        "section_count": len(merged["sections"]),
        "abstract_length": len(merged["abstract"]),
        "keyword_count": len(merged["keywords"])
    }
    
    # Calculate confidence score
    confidence_factors = []
    
    # Abstract confidence
    if len(merged["abstract"]) > 100:
        confidence_factors.append(1.0)
    elif len(merged["abstract"]) > 50:
        confidence_factors.append(0.7)
    else:
        confidence_factors.append(0.3)
    
    # Sections confidence
    if len(merged["sections"]) >= 3:
        confidence_factors.append(1.0)
    elif len(merged["sections"]) >= 1:
        confidence_factors.append(0.6)
    else:
        confidence_factors.append(0.2)
    
    # Metadata confidence
    metadata_score = 0
    if merged["metadata"]["title"]:
        metadata_score += 0.3
    if merged["metadata"]["authors"]:
        metadata_score += 0.3
    if merged["metadata"]["journal"]:
        metadata_score += 0.2
    if merged["metadata"]["year"]:
        metadata_score += 0.2
    confidence_factors.append(metadata_score)
    
    avg_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0
    
    if avg_confidence >= 0.8:
        merged["processing_info"]["confidence"] = "high"
    elif avg_confidence >= 0.5:
        merged["processing_info"]["confidence"] = "medium"
    else:
        merged["processing_info"]["confidence"] = "low"
    
    merged["processing_info"]["confidence_score"] = round(avg_confidence, 2)
    
    return merged

# =====================================================
# MAIN PROCESSING FUNCTION
# =====================================================

def process_single_paper(pdf_path: Path, backend: str = None) -> Dict[str, Any]:
    """Process a single PDF paper and return structured data"""
    
    start_time = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {pdf_path.name}")
    logger.info(f"{'='*60}")
    
    try:
        # Step 1: Heuristic parsing
        logger.info("Step 1: Running heuristic parser...")
        heuristic_parser = AdvancedHeuristicParser()
        heuristic_result = heuristic_parser.parse(pdf_path)
        
        logger.info(f"  ✓ Heuristic parsing complete")
        logger.info(f"  - Title extracted: {bool(heuristic_result.get('metadata', {}).get('title'))}")
        logger.info(f"  - Abstract length: {len(heuristic_result.get('abstract', ''))} chars")
        logger.info(f"  - Sections found: {len(heuristic_result.get('sections', []))}")
        logger.info(f"  - Keywords: {len(heuristic_result.get('keywords', []))}")
        
        # Step 2: Extract sample text for LLM
        logger.info("Step 2: Extracting text for LLM...")
        doc = fitz.open(pdf_path)
        sample_text = ""
        for p in range(min(3, len(doc))):  # First 3 pages
            sample_text += doc[p].get_text() + "\n"
        doc.close()
        
        # Step 3: Extract blocks for LLM
        blocks = extract_blocks_for_llm(pdf_path, max_pages=3)
        logger.info(f"  ✓ Extracted {len(blocks)} text blocks")
        
        # Step 4: LLM parsing
        logger.info(f"Step 3: Running LLM parser ({backend or LLM_BACKEND})...")
        llm_result = llm_parse(blocks, sample_text, backend)
        
        if "error" in llm_result:
            logger.warning(f"  ⚠️  LLM parsing had issues: {llm_result['error']}")
            # Use heuristic result only
            final_result = merge_and_validate_results(heuristic_result, {})
            final_result["processing_info"]["llm_used"] = False
            final_result["processing_info"]["llm_error"] = llm_result["error"]
        else:
            logger.info("  ✓ LLM parsing complete")
            logger.info(f"  - LLM extracted sections: {len(llm_result.get('sections', []))}")
            
            # Step 5: Merge results
            logger.info("Step 4: Merging results...")
            final_result = merge_and_validate_results(heuristic_result, llm_result)
            logger.info(f"  ✓ Merge complete (confidence: {final_result['processing_info']['confidence']})")
        
        # Step 6: Add processing metadata
        processing_time = time.time() - start_time
        final_result["processing_info"]["processing_time_seconds"] = round(processing_time, 2)
        final_result["processing_info"]["paper_filename"] = pdf_path.name
        final_result["processing_info"]["file_size_mb"] = round(pdf_path.stat().st_size / (1024 * 1024), 2)
        
        logger.info(f"\n✅ Successfully processed {pdf_path.name}")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        logger.info(f"   Final confidence: {final_result['processing_info']['confidence']}")
        
        return final_result
        
    except Exception as e:
        logger.error(f"\n❌ Failed to process {pdf_path.name}: {str(e)}")
        logger.exception("Detailed error traceback:")
        
        return {
            "error": str(e),
            "paper_filename": pdf_path.name,
            "processing_info": {
                "success": False,
                "error_message": str(e),
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
        }

# =====================================================
# BATCH PROCESSING AND MAIN
# =====================================================

def main():
    """Main entry point for batch processing"""
    
    print("\n" + "="*70)
    print("CITEPRISM - Research Paper Parser v2.0")
    print("="*70)
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  • LLM Backend: {LLM_BACKEND.upper()}")
    
    if LLM_BACKEND == "openai":
        print(f"  • OpenAI Model: {OPENAI_MODEL}")
        if not OPENAI_API_KEY:
            print(f"  ⚠️  Warning: OpenAI API key not set")
    elif LLM_BACKEND == "hf":
        print(f"  • Hugging Face Model: {HF_MODEL}")
        if not HF_API_KEY:
            print(f"  ⚠️  Warning: Hugging Face API key not set")
    elif LLM_BACKEND == "ollama":
        print(f"  • Ollama Model: {OLLAMA_MODEL}")
        print(f"  • Ollama URL: {OLLAMA_BASE_URL}")
    
    # Check PDF directory
    if not PDF_DIR.exists():
        print(f"\n❌ PDF directory not found: {PDF_DIR}")
        print(f"\nCreating directory and instructions...")
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {PDF_DIR}")
        print(f"\nPlease add your PDF files to this directory and run again.")
        print(f"Example: cp *.pdf {PDF_DIR}/")
        return
    
    # Find PDF files
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n❌ No PDF files found in {PDF_DIR}")
        print(f"\nAdd PDF files to the directory and try again.")
        return
    
    print(f"\nFound {len(pdf_files)} PDF file(s):")
    for i, pdf in enumerate(pdf_files, 1):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  {i:2d}. {pdf.name} ({size_mb:.1f} MB)")
    
    print(f"\nOutput will be saved to: {OUT_DIR}")
    print("-" * 70)
    
    # Process each PDF
    successful = []
    failed = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        
        result = process_single_paper(pdf_file, LLM_BACKEND)
        
        if "error" in result and not result.get("processing_info", {}).get("success", True):
            print(f"  ❌ Failed: {result['error'][:100]}...")
            failed.append(pdf_file.name)
        else:
            # Save result
            output_file = OUT_DIR / f"{pdf_file.stem}_parsed.json"
            
            # Pretty print with indentation
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Saved: {output_file.name}")
            
            # Show quick summary
            if "metadata" in result:
                meta = result["metadata"]
                title = meta.get('title', 'N/A')
                if len(title) > 60:
                    title = title[:57] + "..."
                print(f"     Title: {title}")
                print(f"     Authors: {len(meta.get('authors', []))}")
                print(f"     Sections: {len(result.get('sections', []))}")
            
            successful.append(pdf_file.name)
    
    # Summary
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Total papers: {len(pdf_files)}")
    print(f"Successful:   {len(successful)}")
    print(f"Failed:       {len(failed)}")
    
    if successful:
        print(f"\n✅ Output files saved in: {OUT_DIR.absolute()}")
        
        # Show example output structure
        if successful:
            first_output = OUT_DIR / f"{Path(successful[0]).stem}_parsed.json"
            if first_output.exists():
                print(f"\nExample output from {first_output.name}:")
                try:
                    with open(first_output, 'r', encoding='utf-8') as f:
                        example = json.load(f)
                    
                    print(f"  • Title: {example.get('metadata', {}).get('title', 'N/A')[:80]}...")
                    print(f"  • Abstract: {len(example.get('abstract', ''))} chars")
                    print(f"  • Sections: {len(example.get('sections', []))}")
                    keywords = example.get('keywords', [])
                    if keywords:
                        print(f"  • Keywords: {', '.join(keywords[:5])}")
                        if len(keywords) > 5:
                            print(f"    ... and {len(keywords) - 5} more")
                    
                except Exception as e:
                    print(f"  • Could not display example: {e}")
    
    if failed:
        print(f"\n⚠️  Failed files ({len(failed)}):")
        for f in failed:
            print(f"  • {f}")
        print(f"\nCheck citeprism.log for detailed error messages.")

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.exception("Stack trace:")
        print(f"\n❌ A fatal error occurred: {e}")
        print("Check citeprism.log for details.")