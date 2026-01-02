# src/core/pdf_parser.py - IMPROVED VERSION
import re
import pdfplumber
from typing import List, Optional, Dict, Tuple
from src.utils.exceptions import PDFProcessingException
from src.utils.logger import logger

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logger.warning("pdfminer.six not installed. Install with: pip install pdfminer.six")


class PDFParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.full_text: str = ""
        self.metadata: Dict = {}
        self.pages_text: List[str] = []

    def load(self) -> None:
        """Load PDF with multiple extraction methods as fallback."""
        try:
            logger.info(f"Loading PDF: {self.file_path}")
            
            # Try pdfplumber first (better layout preservation)
            success = self._load_with_pdfplumber()
            
            # Fallback to pdfminer if text quality is poor
            if not success or self._is_text_quality_poor():
                logger.warning("pdfplumber extraction poor, trying pdfminer...")
                self._load_with_pdfminer()
            
            if not self.full_text.strip():
                raise PDFProcessingException("PDF loaded but no text found (scanned image?).")
            
            logger.info(f"PDF Loaded. Total characters: {len(self.full_text)}")
            logger.debug(f"First 500 chars: {self.full_text[:500]}")

        except Exception as e:
            logger.error(f"Failed to process PDF {self.file_path}: {e}")
            raise PDFProcessingException(f"Error reading PDF: {e}")

    def _load_with_pdfplumber(self) -> bool:
        """Extract with pdfplumber."""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                self.metadata = pdf.metadata or {}
                pages = []
                
                for page in pdf.pages:
                    # Try layout-aware extraction first
                    text = page.extract_text(layout=True) or page.extract_text() or ""
                    pages.append(text)
                
                self.pages_text = pages
                self.full_text = "\n".join([p for p in pages if p])
                return True
                
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return False

    def _load_with_pdfminer(self) -> bool:
        """Fallback: Extract with pdfminer (better for complex PDFs)."""
        if not PDFMINER_AVAILABLE:
            return False
            
        try:
            self.full_text = pdfminer_extract(self.file_path)
            # Split into pages (approximate - pdfminer doesn't preserve page boundaries well)
            self.pages_text = self.full_text.split('\f')  # form feed = page break
            return True
            
        except Exception as e:
            logger.error(f"pdfminer extraction failed: {e}")
            return False

    def _is_text_quality_poor(self) -> bool:
        """Detect if extracted text has quality issues."""
        if len(self.full_text) < 100:
            return True
        
        # Check for concatenated words (lack of spaces)
        sample = self.full_text[:2000]
        words = sample.split()
        
        # If average word length > 15, likely concatenated
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len > 15:
                logger.warning(f"Poor text quality detected (avg word len: {avg_word_len:.1f})")
                return True
        
        # Check for lack of punctuation
        punct_count = sum(1 for c in sample if c in '.,;:!?')
        if punct_count < len(sample) / 100:  # Less than 1% punctuation
            logger.warning("Poor text quality: minimal punctuation")
            return True
        
        return False

    def get_first_pages(self, num_pages: int = 2) -> str:
        """Returns text from the first N pages."""
        if self.pages_text:
            return "\n".join(self.pages_text[:num_pages])
        
        # Fallback: reopen
        try:
            with pdfplumber.open(self.file_path) as pdf:
                text = ""
                for i in range(min(num_pages, len(pdf.pages))):
                    text += (pdf.pages[i].extract_text(layout=True) or "") + "\n"
                return text
        except Exception as e:
            raise PDFProcessingException(f"Error reading first pages: {e}")

    # -------------------------
    # IMPROVED Reference Detection
    # -------------------------
    
    def get_references_section_candidate(self) -> Tuple[str, str]:
        """
        Returns (references_text, detection_method).
        Uses multiple strategies with confidence scoring.
        """
        strategies = [
            self._detect_references_by_page_scan,
            self._detect_references_by_numbered_pattern,
            self._detect_references_by_keyword_density,
        ]
        
        best_result = ("", "none", 0.0)  # (text, method, confidence)
        
        for strategy in strategies:
            try:
                text, method, confidence = strategy()
                if confidence > best_result[2]:
                    best_result = (text, method, confidence)
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")
        
        if best_result[2] < 0.3:  # Low confidence
            logger.warning(f"Low confidence ({best_result[2]:.2f}) in reference detection, using fallback")
            return self._get_references_from_fulltext_fallback(), "fallback"
        
        logger.info(f"References detected via {best_result[1]} (confidence: {best_result[2]:.2f})")
        return best_result[0], best_result[1]

    def _detect_references_by_page_scan(self) -> Tuple[str, str, float]:
        """Scan pages from end to find 'References' heading."""
        if not self.pages_text:
            return "", "page_scan", 0.0
        
        for p_idx in range(len(self.pages_text) - 1, -1, -1):
            page = self.pages_text[p_idx] or ""
            lines = page.split("\n")
            
            for line_idx, line in enumerate(lines[:80]):
                if self._is_reference_heading_line(line):
                    # Extract from this line onward
                    collected = ["\n".join(lines[line_idx:])]
                    
                    # Add subsequent pages until stop heading
                    for next_p in range(p_idx + 1, len(self.pages_text)):
                        next_page = self.pages_text[next_p] or ""
                        if self._has_stop_heading(next_page):
                            break
                        collected.append(next_page)
                    
                    text = "\n".join(collected)[:30000]
                    confidence = self._calculate_reference_confidence(text)
                    return text, "page_scan", confidence
        
        return "", "page_scan", 0.0

    def _detect_references_by_numbered_pattern(self) -> Tuple[str, str, float]:
        """Find dense region of numbered references like [1], [2]..."""
        if not self.full_text:
            return "", "numbered", 0.0
        
        # Find all numbered reference patterns
        pattern = r'\[(\d+)\]|\((\d+)\)|\n(\d+)\.\s+'
        matches = list(re.finditer(pattern, self.full_text))
        
        if len(matches) < 5:
            return "", "numbered", 0.0
        
        # Find densest region (most matches per 5000 chars)
        window = 5000
        best_start = 0
        best_density = 0
        
        for m in matches[:len(matches)//2]:  # Start from first half
            start_pos = m.start()
            end_pos = min(start_pos + window, len(self.full_text))
            window_text = self.full_text[start_pos:end_pos]
            
            density = len(re.findall(pattern, window_text))
            if density > best_density:
                best_density = density
                best_start = start_pos
        
        if best_density < 10:  # Need at least 10 references in window
            return "", "numbered", 0.0
        
        # Extract from best_start to end (or until stop keyword)
        text = self.full_text[best_start:best_start + 45000]
        confidence = min(best_density / 50, 0.95)  # Cap at 0.95
        
        return text, "numbered", confidence

    def _detect_references_by_keyword_density(self) -> Tuple[str, str, float]:
        """Find section with highest density of reference keywords."""
        keywords = ['doi', 'journal', 'proceedings', 'arxiv', 'published', 'vol', 'pp']
        
        # Split text into chunks
        chunk_size = 5000
        chunks = []
        for i in range(0, len(self.full_text), chunk_size):
            chunks.append(self.full_text[i:i+chunk_size])
        
        # Score each chunk
        best_idx = 0
        best_score = 0
        
        for idx, chunk in enumerate(chunks[len(chunks)//2:]):  # Start from middle
            score = sum(chunk.lower().count(kw) for kw in keywords)
            if score > best_score:
                best_score = score
                best_idx = len(chunks)//2 + idx
        
        if best_score < 5:
            return "", "keyword", 0.0
        
        # Extract from best chunk to end
        start = best_idx * chunk_size
        text = self.full_text[start:start + 30000]
        confidence = min(best_score / 30, 0.85)
        
        return text, "keyword", confidence

    def _calculate_reference_confidence(self, text: str) -> float:
        """Calculate confidence score for reference section."""
        if len(text) < 200:
            return 0.0
        
        indicators = [
            (r'\[\d+\]', 0.3),  # [1], [2]
            (r'\(\d{4}\)', 0.2),  # (2021)
            (r'\bdoi:', 0.25),
            (r'\bvol\.\s*\d+', 0.15),
            (r',\s*pp\.\s*\d+', 0.1),
        ]
        
        score = 0.0
        for pattern, weight in indicators:
            count = len(re.findall(pattern, text, re.IGNORECASE))
            score += min(count / 20, 1.0) * weight
        
        return min(score, 1.0)

    def _is_reference_heading_line(self, line: str) -> bool:
        """Check if line is a reference heading."""
        line = line.strip().lower()
        if len(line) > 80 or len(line) < 4:
            return False
        
        patterns = [
            r'^\s*\d*\.?\s*references?\s*$',
            r'^\s*\d*\.?\s*bibliography\s*$',
            r'^\s*\d*\.?\s*works?\s+cited\s*$',
            r'^\s*\d*\.?\s*literature\s+cited\s*$',
        ]
        
        return any(re.match(p, line) for p in patterns)

    def _has_stop_heading(self, text: str) -> bool:
        """Check if text contains stop headings (appendix, acknowledgments, etc)."""
        stop_keywords = ['appendix', 'acknowledgment', 'acknowledgement', 'author contribution', 'funding', 'supplementary']
        first_lines = text.split('\n')[:10]
        first_text = ' '.join(first_lines).lower()
        
        return any(kw in first_text for kw in stop_keywords)

    def _get_references_from_fulltext_fallback(self) -> str:
        """Fallback: last 20% of document."""
        if not self.full_text:
            return ""
        
        tail_size = max(15000, int(len(self.full_text) * 0.2))
        return self.full_text[-tail_size:].strip()