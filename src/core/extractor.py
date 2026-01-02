# src/core/extractor.py - IMPROVED VERSION
import re
import json
from typing import Dict, Any, List, Optional

from src.core.pdf_parser import PDFParser
from src.providers.base import LLMProvider
from src.utils.logger import logger


class ExtractionEngine:
    def __init__(self, parser: PDFParser, llm_provider: LLMProvider, config: Dict):
        self.parser = parser
        self.llm = llm_provider
        self.config = config
        
        self.ref_chunk_size = (
            config.get("extraction", {})
            .get("reference_extraction", {})
            .get("chunk_size", 4000)
        )
        
        # Quality thresholds
        self.min_references = 5
        self.min_structured_ratio = 0.3  # 30% should have title or DOI

    def process(self) -> Dict[str, Any]:
        logger.info(f"Starting extraction for file: {self.parser.file_path}")
        
        # Extract metadata
        regex_data = self._extract_regex_metadata()
        intro_text = self.parser.get_first_pages(2)
        llm_metadata = self._extract_llm_metadata(intro_text)
        abstract = self._extract_abstract(intro_text)
        
        # Extract references with validation and retry
        references = self._extract_references_with_retry()
        
        final_output = {
            "metadata": {
                "source": self.parser.file_path,
                "doi": regex_data.get("doi") or llm_metadata.get("doi"),
                "title": llm_metadata.get("title"),
                "authors": llm_metadata.get("authors"),
                "keywords": llm_metadata.get("keywords", []),
                "publication_date": llm_metadata.get("publication_date"),
            },
            "content": {
                "abstract": abstract,
                "full_text_length": len(self.parser.full_text),
            },
            "references": references,
            "extraction_quality": self._calculate_quality_metrics(references)
        }
        
        return final_output

    def _extract_references_with_retry(self) -> List[Dict]:
        """Extract references with multiple attempts and strategies."""
        
        # Attempt 1: Use detected reference section
        ref_text, method = self.parser.get_references_section_candidate()
        references = self._extract_references_enhanced(ref_text, method)
        
        if self._validate_references(references):
            return references
        
        logger.warning(f"First attempt failed validation ({len(references)} refs). Trying alternative...")
        
        # Attempt 2: Try different section if first failed
        if method != "fallback":
            fallback_text = self.parser._get_references_from_fulltext_fallback()
            references = self._extract_references_enhanced(fallback_text, "fallback_retry")
            
            if self._validate_references(references):
                return references
        
        # Attempt 3: Use full text with stricter parsing
        logger.warning("Attempting extraction from full text with strict parsing...")
        references = self._extract_references_strict(self.parser.full_text)
        
        return references

    def _validate_references(self, refs: List[Dict]) -> bool:
        """Check if extracted references meet quality thresholds."""
        if len(refs) < self.min_references:
            logger.warning(f"Too few references: {len(refs)} < {self.min_references}")
            return False
        
        # Count how many have structured data (title or DOI)
        structured = sum(1 for r in refs if r.get('title') or r.get('doi'))
        ratio = structured / len(refs) if refs else 0
        
        if ratio < self.min_structured_ratio:
            logger.warning(f"Low structured ratio: {ratio:.2%} < {self.min_structured_ratio:.2%}")
            return False
        
        logger.info(f"[OK] References passed validation: {len(refs)} refs, {ratio:.1%} structured")
        return True

    def _calculate_quality_metrics(self, refs: List[Dict]) -> Dict:
        """Calculate extraction quality metrics."""
        if not refs:
            return {"total": 0, "structured_ratio": 0, "with_doi": 0, "with_title": 0}
        
        return {
            "total": len(refs),
            "structured_ratio": sum(1 for r in refs if r.get('title') or r.get('doi')) / len(refs),
            "with_doi": sum(1 for r in refs if r.get('doi')),
            "with_title": sum(1 for r in refs if r.get('title')),
            "with_authors": sum(1 for r in refs if r.get('authors')),
            "with_year": sum(1 for r in refs if r.get('year')),
        }

    # -------------------------
    # Reference Extraction
    # -------------------------
    
    def _extract_references_enhanced(self, text: str, method: str) -> List[Dict]:
        """
        Two-stage extraction:
        1. Segment text into individual reference blocks (preprocessing)
        2. Send clean blocks to LLM for structuring
        """
        if not text or len(text) < 200:
            logger.warning(f"Reference text too short ({len(text)} chars)")
            return []
        
        # Clean text first
        text = self._clean_references_text(text)
        
        # Segment into individual references
        blocks = self._regex_segment_references(text)
        logger.info(f"Segmented into {len(blocks)} reference blocks via {method}")
        
        if len(blocks) < 3:
            logger.warning("Too few blocks after segmentation, trying line-by-line split")
            blocks = self._fallback_segment_by_lines(text)
        
        # Group blocks into chunks for LLM
        chunks = self._split_blocks_into_chunks(blocks, self.ref_chunk_size)
        logger.info(f"Sending {len(chunks)} chunks to LLM")
        
        all_refs = []
        
        # Enhanced prompt with examples
        prompt = self._get_reference_extraction_prompt()
        
        for idx, chunk in enumerate(chunks, start=1):
            logger.info(f"Processing chunk {idx}/{len(chunks)} ({len(chunk)} chars)")
            
            try:
                res = self.llm.extract_structured_data(chunk, prompt)
                refs = (res or {}).get("references", [])
                
                if not isinstance(refs, list):
                    logger.warning(f"Chunk {idx}: Invalid response format")
                    continue
                
                logger.info(f"Chunk {idx}: Extracted {len(refs)} references")
                
                for r in refs:
                    if not isinstance(r, dict):
                        continue
                    
                    # Validate and clean
                    cleaned = self._clean_reference_entry(r)
                    if cleaned and self._is_valid_reference(cleaned):
                        all_refs.append(cleaned)
                    
            except Exception as e:
                logger.error(f"Chunk {idx} failed: {e}", exc_info=True)
        
        # Deduplicate
        unique = self._deduplicate_references(all_refs)
        logger.info(f"Extracted {len(unique)} unique references (from {len(all_refs)} total)")
        
        return unique

    def _extract_references_strict(self, text: str) -> List[Dict]:
        """
        Strict extraction: parse only clear numbered references.
        Used as last resort when other methods fail.
        """
        # Find all numbered references
        pattern = r'\[(\d+)\]\s*(.{50,2000}?)(?=\[\d+\]|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        refs = []
        for num, ref_text in matches:
            ref_text = ref_text.strip()
            if len(ref_text) < 30:
                continue
            
            # Extract basic info with regex
            year_match = re.search(r'\b(19|20)\d{2}\b', ref_text)
            doi_match = re.search(r'10\.\d{4,9}/\S+', ref_text)
            
            refs.append({
                "raw_text": ref_text[:500],
                "title": None,
                "authors": None,
                "year": int(year_match.group()) if year_match else None,
                "journal": None,
                "doi": doi_match.group() if doi_match else None,
            })
        
        logger.info(f"Strict parsing found {len(refs)} numbered references")
        return refs

    def _get_reference_extraction_prompt(self) -> str:
        """Enhanced prompt with better instructions and examples."""
        return """Extract ALL academic references from the text below.

**IMPORTANT RULES:**
1. Each reference must have "raw_text" (the original text as-is)
2. Try to extract: title, authors, year, journal, DOI
3. If a field is unclear or missing, use null (not empty string)
4. For authors: extract as a single string (e.g., "Smith, J., Jones, A.")
5. For year: extract as a number (e.g., 2023, not "2023")
6. Clean up formatting but preserve content

**EXAMPLE INPUT:**
[1] Smith, J., & Jones, A. (2023). Deep learning methods. Nature, 500(1), 1-10. doi:10.1038/nature12345

**EXAMPLE OUTPUT:**
{
  "references": [
    {
      "raw_text": "[1] Smith, J., & Jones, A. (2023). Deep learning methods. Nature, 500(1), 1-10. doi:10.1038/nature12345",
      "title": "Deep learning methods",
      "authors": "Smith, J., Jones, A.",
      "year": 2023,
      "journal": "Nature",
      "doi": "10.1038/nature12345"
    }
  ]
}

Return ONLY valid JSON with this exact structure.
If NO references found, return {"references": []}.

TEXT TO PARSE:"""

    def _clean_reference_entry(self, ref: Dict) -> Optional[Dict]:
        """Clean and validate a single reference entry."""
        raw = (ref.get("raw_text") or "").strip()
        if not raw or len(raw) < 20:
            return None
        
        return {
            "raw_text": raw[:1000],  # Cap length
            "title": self._clean_string(ref.get("title")),
            "authors": self._clean_string(ref.get("authors")),
            "year": self._coerce_year(ref.get("year")),
            "journal": self._clean_string(ref.get("journal")),
            "doi": self._clean_doi(ref.get("doi")),
        }

    def _clean_string(self, s) -> Optional[str]:
        """Clean string field."""
        if not s or s in ['null', 'None', 'N/A']:
            return None
        s = str(s).strip()
        return s if len(s) > 2 else None

    def _clean_doi(self, doi) -> Optional[str]:
        """Clean and validate DOI."""
        if not doi:
            return None
        
        doi = str(doi).strip()
        
        # Remove common prefixes
        doi = re.sub(r'^(https?://)?((dx\.)?doi\.org/)?', '', doi, flags=re.IGNORECASE)
        
        # Validate format
        if re.match(r'^10\.\d{4,9}/\S+', doi):
            return doi
        
        return None

    def _clean_references_text(self, text: str) -> str:
        """Clean reference section text and pre-split glued references."""

        # -----------------------------------
        # FIX: Pre-split glued references
        # -----------------------------------

        # Case 1: ") [12]"
        text = re.sub(
            r'(\))\s*(\[\d{1,4}\])',
            r'\1\n\2',
            text
        )

        # Case 2: ". [12]"
        text = re.sub(
            r'(\.)\s*(\[\d{1,4}\])',
            r'\1\n\2',
            text
        )

        # Case 3: DOI followed by reference "[12]"
        text = re.sub(
            r'(10\.\d{4,9}/[^\s]+)\s*(\[\d{1,4}\])',
            r'\1\n\2',
            text,
            flags=re.IGNORECASE
        )

        # Case 4: Year followed by reference "2023 [12]"
        text = re.sub(
            r'(\b(19|20)\d{2}\b)\s*(\[\d{1,4}\])',
            r'\1\n\3',
            text
        )

        # Case 5: author/title text followed immediately by "[12]"
        text = re.sub(
            r'([a-zA-Z])\s*(\[\d{1,4}\])',
            r'\1\n\2',
            text
        )

        # 🚨 Case 6 (CRITICAL): force-split ANY mid-line reference index
        # This is the ultimate safety net
        text = re.sub(
            r'(?<!\n)(\[\d{1,4}\])',
            r'\n\1',
            text
        )

        # -----------------------------------
        # Existing cleanup logic
        # -----------------------------------
        lines = text.split("\n")
        cleaned = []

        for line in lines:
            s = line.strip()
            if not s:
                continue

            # Drop lone page numbers
            if s.isdigit() and len(s) <= 4:
                continue

            # Drop very short noise lines
            if len(s) < 15:
                continue

            cleaned.append(line)

        out = "\n".join(cleaned)

        # Collapse excessive blank lines
        out = re.sub(r"\n\s*\n+", "\n\n", out)

        return out.strip()

    def _regex_segment_references(self, text: str) -> List[str]:
        """
        Segment text into individual references using multiple patterns.
        """
        text = "\n" + text.strip()
        
        # Try numbered patterns: [1], 1., (1), 1)
        split_rx = re.compile(
            r"\n(?="
            r"(?:\[\d{1,4}\]\s*)"      # [1]
            r"|(?:^\d{1,4}\.\s+)"      # 1. at start of line
            r"|(?:\(\d{1,4}\)\s*)"     # (1)
            r"|(?:^\d{1,4}\)\s+)"      # 1) at start of line
            r")",
            re.MULTILINE
        )
        
        parts = [p.strip() for p in split_rx.split(text) if p.strip()]
        
        # Filter valid references (must be substantial)
        valid_parts = [p for p in parts if len(p) >= 50]
        
        if len(valid_parts) >= 5:
            logger.debug(f"Regex segmentation found {len(valid_parts)} references")
            return valid_parts
        
        # Fallback: split by double newlines
        logger.debug("Regex segmentation weak, trying paragraph split")
        blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
        return [b for b in blocks if len(b) >= 50]

    def _fallback_segment_by_lines(self, text: str) -> List[str]:
        """Emergency fallback: treat each substantial line as a reference."""
        lines = text.split('\n')
        blocks = []
        current = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current:
                    blocks.append(' '.join(current))
                    current = []
                continue
            
            # Start new block if line begins with number/bracket
            if re.match(r'^[\[\(]?\d+[\.\)\]]', line):
                if current:
                    blocks.append(' '.join(current))
                current = [line]
            else:
                current.append(line)
        
        if current:
            blocks.append(' '.join(current))
        
        return [b for b in blocks if len(b) >= 50]

    def _split_blocks_into_chunks(self, blocks: List[str], max_chars: int) -> List[str]:
        """Group reference blocks into chunks."""
        chunks = []
        cur = []
        cur_len = 0
        
        for b in blocks:
            if cur_len + len(b) + 2 > max_chars and cur:
                chunks.append("\n\n".join(cur))
                cur = [b]
                cur_len = len(b)
            else:
                cur.append(b)
                cur_len += len(b) + 2
        
        if cur:
            chunks.append("\n\n".join(cur))
        
        return chunks

    def _is_valid_reference(self, ref: Dict) -> bool:
        """Validate reference quality."""
        raw = (ref.get("raw_text") or "").strip()
        
        if len(raw) < 30:
            return False
        
        # Must have at least one strong indicator
        has_year = bool(ref.get("year"))
        has_doi = bool(ref.get("doi"))
        has_title = bool(ref.get("title"))
        has_journal = bool(ref.get("journal"))
        
        # Check raw text for patterns if structured fields missing
        if not (has_year or has_doi or has_title):
            has_year_pattern = bool(re.search(r'\b(19|20)\d{2}\b', raw))
            has_doi_pattern = bool(re.search(r'\b10\.\d{4,9}/\S+\b', raw))
            has_author_pattern = bool(re.search(r'[A-Z][a-z]+,\s*[A-Z]\.?', raw))
            
            return has_year_pattern or has_doi_pattern or has_author_pattern
        
        return True

    def _coerce_year(self, y):
        """Convert year to int or None."""
        if y is None:
            return None
        if isinstance(y, int):
            return y if 1900 <= y <= 2100 else None
        if isinstance(y, str):
            m = re.search(r'(19\d{2}|20\d{2})', str(y))
            return int(m.group(1)) if m else None
        return None

    def _deduplicate_references(self, references: List[Dict]) -> List[Dict]:
        """Remove duplicate references using multiple signals."""
        seen = set()
        unique = []
        
        for ref in references:
            raw = (ref.get("raw_text") or "").lower().strip()
            title = (ref.get("title") or "").lower().strip()
            doi = (ref.get("doi") or "").lower().strip()
            year = str(ref.get("year") or "")
            authors = (ref.get("authors") or "").lower().strip()
            
            # Create signatures for deduplication
            sigs = []
            
            # DOI is strongest signal
            if doi and len(doi) > 6:
                sigs.append(f"doi:{doi}")
            
            # Title + year
            if title and len(title) > 15:
                sigs.append(f"title:{title[:100]}:{year}")
            
            # Authors + year (for papers without DOI)
            if authors and len(authors) > 10 and year:
                author_sig = authors[:50]  # First author usually enough
                sigs.append(f"auth:{author_sig}:{year}")
            
            # Raw text as last resort
            if raw and len(raw) > 50:
                sigs.append(f"raw:{raw[:150]}")
            
            if not sigs:
                continue
            
            # Check if any signature already seen
            if any(s in seen for s in sigs):
                continue
            
            unique.append(ref)
            for s in sigs:
                seen.add(s)
        
        return unique

    # -------------------------
    # Metadata Extraction
    # -------------------------
    
    def _extract_regex_metadata(self) -> Dict[str, Any]:
        """Extract metadata using regex patterns."""
        text = self.parser.full_text
        data: Dict[str, Any] = {}
        
        # DOI extraction
        doi_match = re.search(r"\b(10\.\d{4,}/[-._;()/:a-zA-Z0-9]+)\b", text)
        if doi_match:
            data["doi"] = doi_match.group(1)
            logger.debug(f"DOI found via Regex: {data['doi']}")
        
        return data

    def _extract_llm_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata using LLM."""
        prompt = """Analyze this research paper text and extract metadata.

Return ONLY this JSON format:
{
  "title": "Full paper title",
  "authors": [
    {"name": "First Author", "affiliation": "Their institution or null"},
    {"name": "Second Author", "affiliation": "Their institution or null"}
  ],
  "keywords": ["keyword1", "keyword2"],
  "publication_date": "2023",
  "doi": "10.xxxx/xxxxx or null"
}

Rules:
- Extract EXACTLY as it appears in the paper
- If a field is unclear, use null
- For publication_date, extract year as string
- For authors, always include "name" and "affiliation" (can be null)
"""
        try:
            return self.llm.extract_structured_data(text, prompt)
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}

    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract abstract from paper."""
        prompt = """Extract the abstract from this research paper.

Return this JSON format:
{
  "abstract_text": "The complete abstract text here..."
}

If no abstract found, return {"abstract_text": null}.
Return ONLY JSON, no other text."""
        
        try:
            res = self.llm.extract_structured_data(text, prompt)
            abstract = res.get("abstract_text")
            
            if abstract and len(abstract) > 100:
                return abstract
            return None
            
        except Exception as e:
            logger.warning(f"Abstract extraction failed: {e}")
            return None