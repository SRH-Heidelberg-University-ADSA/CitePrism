"""
CitePrism OpenAlex Enricher - FIXED VERSION
============================================
Fixes:
1. NoneType errors from OpenAlex API
2. Better fallback logging to show why recovery fails
3. More robust error handling
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
# VALIDATION LOGIC
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
        
        # Title fuzzy matching
        title_sim = SequenceMatcher(
            None, 
            parsed_title.lower().strip(), 
            enriched_title.lower().strip()
        ).ratio()
        
        if title_sim < similarity_threshold:
            return f"Mismatch Flagged (Title Similarity: {int(title_sim*100)}%)"

        # Year verification (±1 year tolerance)
        if parsed_year and enriched_year:
            try:
                diff = abs(int(parsed_year) - int(enriched_year))
                if diff == 0:
                    return "Match"
                elif diff <= 1:
                    return "Acceptable Variance (±1 Year)"
                else:
                    return f"Mismatch Flagged (Year diff: {diff})"
            except (TypeError, ValueError):
                return "Error (Year Comparison Failed)"
        
        if title_sim >= similarity_threshold:
            return "Partial Match (Year Missing)"
            
        return "Incomplete Metadata"
    
    except Exception as e:
        logger.error(f"Unexpected error in validation: {e}")
        return "Error (Validation Failed)"

# ============================================================================
# RECONSTRUCTION & TIERED FALLBACK UTILITIES
# ============================================================================

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
    except Exception as e:
        logger.warning(f"Abstract reconstruction failed: {e}")
        return None

def fetch_fallback_abstract(doi: Optional[str], title: str) -> Optional[str]:
    """
    Tiered Fallback Strategy with verbose logging:
    1. Semantic Scholar
    2. Crossref
    3. arXiv
    """
    # TIER 1: SEMANTIC SCHOLAR
    try:
        base_ss = "https://api.semanticscholar.org/graph/v1/paper"
        if doi:
            clean_doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
            url = f"{base_ss}/DOI:{clean_doi}?fields=abstract"
            logger.debug(f"    Trying Semantic Scholar with DOI: {clean_doi}")
        else:
            search_url = f"{base_ss}/search?query={requests.utils.quote(title)}&limit=1&fields=abstract"
            logger.debug(f"    Trying Semantic Scholar search: {title[:50]}")
            s_data = requests.get(search_url, timeout=5).json()
            if s_data.get('data') and len(s_data['data']) > 0:
                url = f"{base_ss}/{s_data['data'][0]['paperId']}?fields=abstract"
            else:
                logger.debug(f"    Semantic Scholar: No results found")
                url = None
        
        if url:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                abs_text = resp.json().get('abstract')
                if abs_text:
                    logger.info("  [SUCCESS] Recovered abstract from Semantic Scholar")
                    return abs_text
                else:
                    logger.debug("    Semantic Scholar: Response OK but no abstract field")
            else:
                logger.debug(f"    Semantic Scholar: HTTP {resp.status_code}")
    except Exception as e:
        logger.debug(f"    Semantic Scholar failed: {str(e)[:100]}")

    # TIER 2: CROSSREF
    if doi:
        try:
            clean_doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
            cr_url = f"https://api.crossref.org/works/{clean_doi}"
            logger.debug(f"    Trying Crossref with DOI: {clean_doi}")
            resp = requests.get(cr_url, timeout=5)
            if resp.status_code == 200:
                raw_abs = resp.json().get('message', {}).get('abstract')
                if raw_abs:
                    clean_abs = re.sub(r'<[^>]*>', '', raw_abs).strip()
                    logger.info("  [SUCCESS] Recovered abstract from Crossref")
                    return clean_abs
                else:
                    logger.debug("    Crossref: Response OK but no abstract field")
            else:
                logger.debug(f"    Crossref: HTTP {resp.status_code}")
        except Exception as e:
            logger.debug(f"    Crossref failed: {str(e)[:100]}")
    else:
        logger.debug("    Crossref: Skipped (no DOI)")

    # TIER 3: ARXIV
    try:
        arxiv_url = f"http://export.arxiv.org/api/query?search_query=ti:\"{requests.utils.quote(title[:100])}\"&max_results=1"
        logger.debug(f"    Trying arXiv: {title[:50]}")
        resp = requests.get(arxiv_url, timeout=5)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            if entries:
                for entry in entries:
                    summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                    if summary_elem is not None and summary_elem.text:
                        logger.info("  [SUCCESS] Recovered abstract from arXiv")
                        return summary_elem.text.replace('\n', ' ').strip()
                logger.debug("    arXiv: Entry found but no summary")
            else:
                logger.debug("    arXiv: No entries found")
        else:
            logger.debug(f"    arXiv: HTTP {resp.status_code}")
    except Exception as e:
        logger.debug(f"    arXiv failed: {str(e)[:100]}")

    logger.debug("  [FAILED] All fallback methods exhausted - no abstract recovered")
    return None

# ============================================================================
# API SEARCH (FIXED VERSION)
# ============================================================================

def search_openalex(
    title: str, 
    authors: List[str], 
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> Optional[Dict]:
    """Query OpenAlex for a paper match with reconstruction and tiered fallback."""
    if not title or not title.strip():
        return None
    
    safe_title = "".join([c if c.isalnum() else "_" for c in title[:30]])
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Searching OpenAlex (Attempt {attempt+1}): {title[:50]}")
            results = Works().search(title).get()
            
            # FIX 1: Check if results is None or empty
            if results is None:
                logger.warning(f"  OpenAlex returned None for: {title[:40]}")
                return None
            
            if not results:
                logger.debug(f"  No results from OpenAlex for: {title[:40]}")
                return None
            
            best_match = results[0]
            
            # FIX 2: Check if best_match is None
            if best_match is None:
                logger.warning(f"  Best match is None for: {title[:40]}")
                return None

            # Debug save raw response
            try:
                debug_dir = Path("data/debug/api_responses")
                debug_dir.mkdir(parents=True, exist_ok=True)
                with open(debug_dir / f"raw_{safe_title}_att{attempt}.json", "w", encoding='utf-8') as f:
                    json.dump(best_match, f, indent=2, ensure_ascii=False)
            except Exception: pass

            # 1. Primary Recovery (Inverted Index)
            abstract_text = reconstruct_abstract(best_match.get('abstract_inverted_index'))
            
            # 2. Tiered Secondary Recovery
            if not abstract_text:
                logger.info(f"  [!] Missing abstract for '{title[:30]}'. Triggering tiered recovery...")
                abstract_text = fetch_fallback_abstract(best_match.get('doi'), title)

            # FIX 3: Safe access to nested dictionaries
            primary_location = best_match.get('primary_location') or {}
            source = primary_location.get('source') or {}
            
            # Return full canonical record
            return {
                "id": best_match.get('id'),
                "title": best_match.get('title'),
                "display_name": best_match.get('display_name'),
                "year": best_match.get('publication_year'),
                "cited_by_count": best_match.get('cited_by_count', 0),
                "is_retracted": best_match.get('is_retracted', False),
                "abstract": abstract_text,
                "doi": best_match.get('doi'),
                "url": primary_location.get('landing_page_url'),
                "authors": [
                    {"display_name": a.get("author", {}).get("display_name")}
                    for a in (best_match.get("authorships") or [])
                ],
                "venue": source.get('display_name')
            }

        except Exception as e:
            logger.error(f"OpenAlex attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    return None

# ============================================================================
# PIPELINE-COMPATIBLE CLASS INTERFACE
# ============================================================================

class OpenAlexEnricher:
    """CitePrism Enrichment Module."""
    
    def __init__(self, config, db_manager):
        self.config = config
        self.db = db_manager
        
        try:
            pyalex_config.email = getattr(config, 'OPENALEX_EMAIL', 'gowrikamahesh2017@gmail.com')
            logger.info(f"Enricher initialized with email: {pyalex_config.email}")
        except Exception as e:
            logger.warning(f"Config setup failed: {e}")
        
        self.max_retries = getattr(config, 'API_RETRY_COUNT', 3)
        self.retry_delay = getattr(config, 'API_RETRY_DELAY', 2.0)
        self.api_delay = getattr(config, 'OPENALEX_RATE_LIMIT', 0.2)
        self.title_threshold = getattr(config, 'TITLE_SIMILARITY_THRESHOLD', 0.7)

    def enrich_references(self, parsed_data: Dict, force: bool = False) -> Dict:
        """Enrich all references in the parsed manuscript."""
        logger.info("=" * 80)
        logger.info(f"ENRICHMENT PIPELINE START (Force Mode: {force})")
        logger.info("=" * 80)
        
        enriched_data = {
            "manuscript_metadata": parsed_data.get("metadata", {}),
            "citations_in_text": parsed_data.get("citations_in_text", []),
            "enriched_references": []
        }
        
        references = parsed_data.get("references_list", [])
        if not references:
            logger.warning("No references found in input data.")
            return enriched_data
        
        stats = {
            "success": 0, 
            "fail": 0, 
            "cached": 0, 
            "mismatches": 0,
            "fallback_success": 0,
            "fallback_fail": 0
        }
        
        for i, ref in enumerate(references, 1):
            try:
                parsed = ref.get('parsed', {})
                title = parsed.get('title', '')
                parsed_year = parsed.get('year')
                
                if not title:
                    logger.warning(f"[{i}] Skipping reference with no title.")
                    stats["fail"] += 1
                    continue
                
                cache_key = title.strip().lower()[:200]
                
                # Check Cache (Skip if force is True)
                cached_res = None if force else self.db.get_cached_response('openalex', cache_key)
                
                had_abstract_before = False
                if cached_res:
                    logger.info(f"[{i}/{len(references)}] Cache Hit: {title[:40]}...")
                    api_result = cached_res
                    status = "success (cached)"
                    stats["cached"] += 1
                    had_abstract_before = bool(cached_res.get('abstract'))
                else:
                    logger.info(f"[{i}/{len(references)}] API Fetch: {title[:40]}...")
                    api_result = search_openalex(title, parsed.get('authors', []), self.max_retries, self.retry_delay)
                    
                    if api_result:
                        self.db.cache_api_response('openalex', cache_key, api_result)
                        status = "success"
                        
                        # Track fallback success
                        if api_result.get('abstract'):
                            stats["fallback_success"] += 1
                        else:
                            stats["fallback_fail"] += 1
                    else:
                        status = "not_found"
                
                # Consistency Check
                if api_result:
                    consistency = check_metadata_consistency(
                        parsed_year, api_result.get('year'), 
                        title, api_result.get('title'), 
                        self.title_threshold
                    )
                    if "Mismatch" in consistency: 
                        stats["mismatches"] += 1
                    stats["success"] += 1
                else:
                    consistency = "Not Checked"
                    stats["fail"] += 1

                enriched_data["enriched_references"].append({
                    "ref_id": ref.get("ref_id", f"ref_{i}"),
                    "original_data": ref,
                    "enrichment_status": status,
                    "consistency_status": consistency,
                    "external_metadata": api_result if api_result else {}
                })
                
                # Global rate limiting
                time.sleep(self.api_delay)
            
            except Exception as e:
                logger.error(f"Critical error on reference {i}: {e}", exc_info=True)
                stats["fail"] += 1

        # Enhanced Summary
        enriched_data["enrichment_summary"] = {
            "total_references": len(references),
            "successfully_enriched": stats["success"],
            "failed": stats["fail"],
            "cached": stats["cached"],
            "metadata_mismatches": stats["mismatches"],
            "abstracts_recovered_via_fallback": stats["fallback_success"],
            "abstracts_still_missing": stats["fallback_fail"],
            "fallback_success_rate": f"{(stats['fallback_success']/(stats['fallback_success']+stats['fallback_fail'])*100):.1f}%" if (stats['fallback_success']+stats['fallback_fail']) > 0 else "N/A",
            "success_rate": f"{(stats['success']/len(references)*100):.1f}%" if references else "0%"
        }
        
        logger.info(f"ENRICHMENT COMPLETE: {stats['success']}/{len(references)} processed successfully.")
        logger.info(f"Fallback Recovery: {stats['fallback_success']} abstracts recovered, {stats['fallback_fail']} still missing")
        return enriched_data