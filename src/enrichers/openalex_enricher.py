"""
CitePrism OpenAlex Enricher - STABLE VERSION WITH WEB SCRAPING
=============================================================
Updates:
1. Integrated Tier 4 Headless Scraper for ScienceDirect and Springer.
2. Fixed Variable Name Error (best_match vs match).
3. Enhanced User-Agent headers to bypass bot detection.
"""

import json
import logging
import time
import os
import re
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from difflib import SequenceMatcher
from bs4 import BeautifulSoup

# External API Wrapper
try:
    from pyalex import Works, config as pyalex_config
except ImportError:
    raise ImportError("pyalex not installed. Run: pip install pyalex")

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger("src.enrichers.openalex_enricher")

# ============================================================================
# TIER 4: HEADLESS WEB SCRAPER (WINDOWLESS)
# ============================================================================

def scrape_abstract_from_url(url: str) -> Optional[str]:
    """
    Directly scrapes the landing page windowlessly using requests + BeautifulSoup.
    Specifically mapped for ScienceDirect (Elsevier) and Springer.
    """
    if not url:
        return None
        
    # High-quality headers to mimic a real browser session
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/',
        'Connection': 'keep-alive'
    }

    try:
        # allow_redirects=True follows DOIs to the final publisher landing page
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        final_url = response.url
        
        # --- SCIENCEDIRECT (ELSEVIER) MAPPING ---
        if "sciencedirect.com" in final_url:
            # Targets the specific abstract containers found in Elsevier HTML
            sd_abs = (soup.find('div', {'id': 'as005'}) or 
                      soup.find('div', {'id': 'sp0005'}) or 
                      soup.find('div', {'id': 'abspara0010'}) or
                      soup.find('div', class_='abstract author'))
            if sd_abs:
                return sd_abs.get_text(separator=' ', strip=True)

        # --- SPRINGER / NATURE MAPPING ---
        if "springer.com" in final_url or "nature.com" in final_url:
            # Targets Springer's unique abstract IDs
            spr_abs = (soup.find('div', {'id': 'Ab1-content'}) or 
                       soup.find('div', {'id': 'Abs1-content'}) or
                       soup.find('div', class_='c-article-section__content'))
            if spr_abs:
                text = spr_abs.get_text(separator=' ', strip=True)
                return re.sub(r'^Abstract\s*', '', text, flags=re.IGNORECASE).strip()

        # --- GENERIC FALLBACK ---
        meta_desc = soup.find('meta', {'name': 'description'}) or soup.find('meta', {'property': 'og:description'})
        if meta_desc:
            return meta_desc.get('content')

    except Exception as e:
        logger.debug(f"Headless scraping failed for {url}: {e}")
        
    return None

# ============================================================================
# VALIDATION & UTILITIES
# ============================================================================

def check_metadata_consistency(
    parsed_year: Optional[int], 
    enriched_year: Optional[int], 
    parsed_title: str, 
    enriched_title: str,
    similarity_threshold: float = 0.7
) -> str:
    """Validates consistency between the PDF's citation and the API result."""
    try:
        if not parsed_title or not enriched_title:
            return "Incomplete Metadata (Missing Title)"
        
        title_sim = SequenceMatcher(None, parsed_title.lower().strip(), enriched_title.lower().strip()).ratio()
        
        if title_sim < similarity_threshold:
            return f"Mismatch Flagged (Title Similarity: {int(title_sim*100)}%)"

        if parsed_year and enriched_year:
            diff = abs(int(parsed_year) - int(enriched_year))
            if diff == 0:
                return "Match"
            elif diff <= 1:
                return "Acceptable Variance (±1 Year)"
            else:
                return f"Mismatch Flagged (Year diff: {diff})"
        
        return "Partial Match (Year Missing)" if title_sim >= similarity_threshold else "Incomplete Metadata"
    except Exception:
        return "Error (Validation Failed)"

def reconstruct_abstract(inverted_index: Optional[Dict]) -> Optional[str]:
    """Reconstructs plain text from OpenAlex Inverted Index format."""
    if not inverted_index:
        return None
    try:
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join([word for pos, word in word_positions]).strip()
    except Exception:
        return None

# ============================================================================
# TIERED FALLBACK STRATEGY
# ============================================================================

def fetch_fallback_abstract(doi: Optional[str], title: str, url: Optional[str] = None) -> Optional[str]:
    """Tiered recovery logic including the new scraping fallback."""
    clean_doi = doi.replace('https://doi.org/', '') if doi else None

    # TIER 1: SEMANTIC SCHOLAR
    try:
        if clean_doi:
            ss_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{clean_doi}?fields=abstract"
            resp = requests.get(ss_url, timeout=5)
            if resp.status_code == 200:
                abs_text = resp.json().get('abstract')
                if abs_text: return abs_text
    except Exception: pass

    # TIER 2: CROSSREF
    try:
        if clean_doi:
            cr_url = f"https://api.crossref.org/works/{clean_doi}"
            resp = requests.get(cr_url, timeout=5)
            if resp.status_code == 200:
                raw_abs = resp.json().get('message', {}).get('abstract')
                if raw_abs: return re.sub(r'<[^>]*>', '', raw_abs).strip()
    except Exception: pass

    # TIER 3: ARXIV
    try:
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=ti:\"{requests.utils.quote(title[:100])}\"&max_results=1"
        resp = requests.get(arxiv_url, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            if entries:
                summary_elem = entries[0].find('{http://www.w3.org/2005/Atom}summary')
                if summary_elem is not None and summary_elem.text:
                    return summary_elem.text.replace('\n', ' ').strip()
    except Exception: pass

    # TIER 4: SCRAPING (Final Attempt)
    target = doi if (doi and "doi.org" in doi) else url
    if target:
        logger.info(f"  [!] Triggering silent scraping for: {target}")
        return scrape_abstract_from_url(target)

    return None

# ============================================================================
# MAIN ENRICHER CLASS
# ============================================================================

class OpenAlexEnricher:
    def __init__(self, config, db_manager):
        self.config = config
        self.db = db_manager
        pyalex_config.email = getattr(config, 'OPENALEX_EMAIL', 'gowrikamahesh2017@gmail.com')
        self.api_delay = getattr(config, 'OPENALEX_RATE_LIMIT', 0.5) # Increased for stability
        self.title_threshold = getattr(config, 'TITLE_SIMILARITY_THRESHOLD', 0.7)

    def enrich_references(self, parsed_data: Dict, force: bool = False) -> Dict:
        enriched_data = {
            "manuscript_metadata": parsed_data.get("metadata", {}),
            "citations_in_text": parsed_data.get("citations_in_text", []),
            "enriched_references": []
        }
        
        references = parsed_data.get("references_list", [])
        stats = {"success": 0, "fail": 0, "cached": 0, "scraped": 0}
        
        for i, ref in enumerate(references, 1):
            parsed = ref.get('parsed', {})
            title = parsed.get('title', '')
            if not title: continue
            
            cache_key = title.strip().lower()[:200]
            cached_res = None if force else self.db.get_cached_response('openalex', cache_key)
            
            if cached_res:
                api_result = cached_res
                status = "success (cached)"
                stats["cached"] += 1
            else:
                api_result = self._search_openalex_logic(title, parsed)
                if api_result:
                    self.db.cache_api_response('openalex', cache_key, api_result)
                    status = "success"
                    if api_result.get('is_scraped'): stats["scraped"] += 1
                else:
                    status, api_result = "not_found", {}

            consistency = check_metadata_consistency(parsed.get('year'), api_result.get('year'), title, api_result.get('title'), self.title_threshold) if api_result else "Not Checked"
            stats["success" if api_result else "fail"] += 1

            enriched_data["enriched_references"].append({
                "ref_id": ref.get("ref_id", f"ref_{i}"),
                "original_data": ref,
                "enrichment_status": status,
                "consistency_status": consistency,
                "external_metadata": api_result if api_result else {}
            })
            time.sleep(self.api_delay)
        return enriched_data

    def _search_openalex_logic(self, title: str, parsed: Dict) -> Optional[Dict]:
        """Queries OpenAlex with reconstruction and tiered fallback."""
        try:
            results = Works().search(title).get()
            if not results: return None
            
            # Use 'best_match' consistently throughout the function
            best_match = results[0]
            abstract = reconstruct_abstract(best_match.get('abstract_inverted_index'))
            is_scraped = False
            
            if not abstract:
                doi = best_match.get('doi')
                url = (best_match.get('primary_location') or {}).get('landing_page_url')
                abstract = fetch_fallback_abstract(doi, title, url)
                if abstract: is_scraped = True

            primary_loc = best_match.get('primary_location') or {}
            source = primary_loc.get('source') or {}

            return {
                "id": best_match.get('id'), 
                "title": best_match.get('title'),
                "year": best_match.get('publication_year'), 
                "abstract": abstract,
                "is_scraped": is_scraped, 
                "doi": best_match.get('doi'),
                "url": primary_loc.get('landing_page_url'),
                "authors": [{"display_name": a.get("author", {}).get("display_name")} for a in (best_match.get("authorships") or [])],
                "venue": source.get('display_name')
            }
        except Exception as e:
            logger.error(f"OpenAlex logic failed for '{title}': {e}")
            return None