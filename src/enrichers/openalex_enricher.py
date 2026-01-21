"""
CitePrism OpenAlex Enricher - Pipeline Compatible
==================================================
Enriches parsed citations with abstracts and canonical metadata using OpenAlex API.

Features:
- OpenAlexEnricher class for pipeline integration
- Database caching support (no redundant API calls)
- Comprehensive error handling with retry logic
- Fuzzy validation for bibliographic variance
- Batch processing capability

Author: CitePrism Team
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from difflib import SequenceMatcher

# External API Wrapper
try:
    from pyalex import Works, config as pyalex_config
except ImportError:
    raise ImportError("pyalex not installed. Run: pip install pyalex")

# Configure logging
logger = logging.getLogger(__name__)


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
    """
    Validates consistency between the PDF's citation and the API result.
    Handles 'Bibliographic Variance' (e.g., Conference vs. Proceedings dates).
    
    Args:
        parsed_year: Year from parsed PDF
        enriched_year: Year from OpenAlex API
        parsed_title: Title from parsed PDF
        enriched_title: Title from OpenAlex API
        similarity_threshold: Minimum acceptable title similarity
        
    Returns:
        Consistency status string
    """
    try:
        # Handle empty titles
        if not parsed_title or not enriched_title:
            logger.warning("One or both titles are empty")
            return "Incomplete Metadata (Missing Title)"
        
        # 1. Title Safety Check
        try:
            title_sim = SequenceMatcher(
                None, 
                parsed_title.lower().strip(), 
                enriched_title.lower().strip()
            ).ratio()
        except Exception as e:
            logger.error(f"Title comparison failed: {e}")
            return "Error (Title Comparison Failed)"
        
        if title_sim < similarity_threshold:
            return f"Mismatch Flagged (Title Similarity: {int(title_sim*100)}%)"

        # 2. Year Verification
        if parsed_year and enriched_year:
            try:
                diff = abs(parsed_year - enriched_year)
                
                if diff == 0:
                    return "Match"
                elif diff <= 1:
                    return "Acceptable Variance (±1 Year)"
                else:
                    return f"Mismatch Flagged (Year diff: {diff})"
            except (TypeError, ValueError) as e:
                logger.error(f"Year comparison failed: {e}")
                return "Error (Year Comparison Failed)"
        
        # If we have title match but missing years
        if title_sim >= similarity_threshold:
            return "Partial Match (Year Missing)"
            
        return "Incomplete Metadata"
    
    except Exception as e:
        logger.error(f"Unexpected error in metadata consistency check: {e}")
        return "Error (Validation Failed)"


# ============================================================================
# API SEARCH
# ============================================================================

def search_openalex(
    title: str, 
    authors: List[str], 
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> Optional[Dict]:
    """
    Query OpenAlex for a paper match with retry logic.
    
    Args:
        title: Paper title to search
        authors: List of author names
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Dictionary with enriched metadata or None if not found
    """
    if not title or not title.strip():
        logger.warning("Empty title provided to search_openalex")
        return None
    
    # Retry loop for API resilience
    for attempt in range(max_retries):
        try:
            logger.debug(f"Searching OpenAlex for: {title[:50]}... (attempt {attempt + 1}/{max_retries})")
            
            try:
                results = Works().search(title).get()
            except AttributeError as e:
                logger.error(f"pyalex API error (check installation): {e}")
                return None
            except ConnectionError as e:
                logger.error(f"Network connection error: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return None
            except Exception as e:
                logger.error(f"OpenAlex search failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                return None
            
            if not results:
                logger.debug(f"No results found for: {title[:50]}...")
                return None
            
            # Take the first result (OpenAlex returns sorted by relevance)
            try:
                best_match = results[0]
            except (IndexError, TypeError) as e:
                logger.error(f"Error accessing search results: {e}")
                return None
            
            # Extract Abstract from inverted index
            abstract_text = None
            try:
                if best_match.get('abstract_inverted_index'):
                    index = best_match['abstract_inverted_index']
                    
                    try:
                        max_pos = max([max(pos) for pos in index.values()])
                        word_list = [""] * (max_pos + 1)
                        
                        for word, positions in index.items():
                            for pos in positions:
                                if 0 <= pos < len(word_list):
                                    word_list[pos] = word
                        
                        abstract_text = " ".join(word_list).strip()
                        
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse abstract inverted index: {e}")
                        abstract_text = None
                        
            except Exception as e:
                logger.warning(f"Abstract extraction failed: {e}")
                abstract_text = None
            
            # Build enriched metadata dictionary
            try:
                enriched_data = {
                    "id": best_match.get('id'),
                    "title": best_match.get('title'),
                    "year": best_match.get('publication_year'),
                    "cited_by_count": best_match.get('cited_by_count', 0),
                    "is_retracted": best_match.get('is_retracted', False),
                    "abstract": abstract_text,
                    "doi": best_match.get('doi'),
                    "url": best_match.get('url'),
                    "authors": [
                        {"display_name": a.get("author", {}).get("display_name")}
                        for a in best_match.get("authorships", [])
                    ]
                }
                
                logger.debug(f"Successfully enriched: {enriched_data.get('title', 'Unknown')[:50]}")
                return enriched_data
                
            except Exception as e:
                logger.error(f"Error building enriched metadata: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Unexpected error in search_openalex (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return None
    
    # If all retries exhausted
    logger.warning(f"All retry attempts exhausted for: {title[:50]}...")
    return None


# ============================================================================
# PIPELINE-COMPATIBLE CLASS INTERFACE
# ============================================================================

class OpenAlexEnricher:
    """
    OpenAlex enricher wrapper for CitePrism pipeline.
    
    Provides database-cached enrichment with automatic retry logic
    and comprehensive error handling.
    """
    
    def __init__(self, config, db_manager):
        """
        Initialize enricher with config and database manager.
        
        Args:
            config: Configuration object with OPENALEX_EMAIL, API settings
            db_manager: DatabaseManager instance for caching
        """
        self.config = config
        self.db = db_manager
        
        # Configure pyalex with user email
        try:
            pyalex_config.email = getattr(config, 'OPENALEX_EMAIL', 'your.email@example.com')
            logger.info(f"Configured OpenAlex API with email: {pyalex_config.email}")
        except Exception as e:
            logger.warning(f"Failed to configure OpenAlex email: {e}")
        
        # Set retry parameters
        self.max_retries = getattr(config, 'API_RETRY_COUNT', 3)
        self.retry_delay = getattr(config, 'API_RETRY_DELAY', 2.0)
        self.api_delay = getattr(config, 'OPENALEX_RATE_LIMIT', 0.1)
        self.title_threshold = getattr(config, 'TITLE_SIMILARITY_THRESHOLD', 0.7)
        
        logger.info(f"Enricher initialized: retries={self.max_retries}, delay={self.api_delay}s")
    
    def enrich_references(self, parsed_data: Dict) -> Dict:
        """
        Enrich all references in parsed manuscript data.
        
        Args:
            parsed_data: Dictionary with 'metadata', 'citations_in_text', 'references_list'
            
        Returns:
            Dictionary with enriched references and statistics
        """
        logger.info("=" * 80)
        logger.info("Starting reference enrichment...")
        logger.info("=" * 80)
        
        # Initialize output structure
        enriched_data = {
            "manuscript_metadata": parsed_data.get("metadata", {}),
            "citations_in_text": parsed_data.get("citations_in_text", []),
            "enriched_references": []
        }
        
        references = parsed_data.get("references_list", [])
        
        if not references:
            logger.warning("No references found in parsed data")
            enriched_data["enrichment_summary"] = {
                "total_references": 0,
                "successfully_enriched": 0,
                "failed": 0,
                "cached": 0,
                "metadata_mismatches": 0,
                "success_rate": "0%"
            }
            return enriched_data
        
        logger.info(f"Found {len(references)} references to enrich")
        
        # Process each reference
        success_count = 0
        fail_count = 0
        cached_count = 0
        mismatch_count = 0
        
        for i, ref in enumerate(references, 1):
            try:
                logger.info(f"[{i}/{len(references)}] Processing: {ref.get('ref_id', f'ref_{i}')}")
                
                # Extract reference data
                parsed = ref.get('parsed', {})
                title = parsed.get('title', '')
                parsed_year = parsed.get('year')
                authors = parsed.get('authors', [])
                
                if not title or not title.strip():
                    logger.warning(f"[NO] Skipping (no title)")
                    enriched_data["enriched_references"].append({
                        "ref_id": ref.get("ref_id", f"ref_{i}"),
                        "original_data": ref,
                        "enrichment_status": "failed",
                        "consistency_status": "Not Checked",
                        "external_metadata": {},
                        "error": "Missing title"
                    })
                    fail_count += 1
                    continue
                
                # Check cache first
                cache_key = title.strip().lower()[:200]  # Limit key length
                cached_result = self.db.get_cached_response('openalex', cache_key)
                
                if cached_result:
                    logger.info(f" [OK] Using cached result")
                    
                    # Validate consistency with cached data
                    enriched_title = cached_result.get('title', '')
                    enriched_year = cached_result.get('year')
                    
                    consistency_status = check_metadata_consistency(
                        parsed_year,
                        enriched_year,
                        title,
                        enriched_title,
                        self.title_threshold
                    )
                    
                    enriched_data["enriched_references"].append({
                        "ref_id": ref.get("ref_id", f"ref_{i}"),
                        "original_data": ref,
                        "enrichment_status": "success (cached)",
                        "consistency_status": consistency_status,
                        "external_metadata": cached_result
                    })
                    
                    success_count += 1
                    cached_count += 1
                    
                    if "Mismatch" in consistency_status:
                        mismatch_count += 1
                    
                    continue
                
                # Fetch from API with retry logic
                logger.info(f"  --> Querying OpenAlex API...")
                api_result = search_openalex(
                    title,
                    authors,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay
                )
                
                if api_result:
                    logger.info(f"  [OK] Successfully enriched")
                    
                    # Cache the result
                    try:
                        self.db.cache_api_response('openalex', cache_key, api_result)
                        logger.debug(f"  --> Cached result for future use")
                    except Exception as e:
                        logger.warning(f"  Failed to cache result: {e}")
                    
                    # Validate consistency
                    enriched_title = api_result.get('title', '')
                    enriched_year = api_result.get('year')
                    
                    consistency_status = check_metadata_consistency(
                        parsed_year,
                        enriched_year,
                        title,
                        enriched_title,
                        self.title_threshold
                    )
                    
                    # Log consistency outcome
                    if "Mismatch" in consistency_status:
                        logger.warning(f"  [!] {consistency_status}")
                        mismatch_count += 1
                    elif "Variance" in consistency_status:
                        logger.info(f"  (i) {consistency_status}")
                    elif "Match" in consistency_status:
                        logger.info(f" [OK] {consistency_status}")
                    else:
                        logger.info(f"  . {consistency_status}")
                    
                    # Check for retraction
                    if api_result.get('is_retracted'):
                        logger.warning(f"  [!] CRITICAL: Reference is RETRACTED!")
                    
                    enriched_data["enriched_references"].append({
                        "ref_id": ref.get("ref_id", f"ref_{i}"),
                        "original_data": ref,
                        "enrichment_status": "success",
                        "consistency_status": consistency_status,
                        "external_metadata": api_result
                    })
                    
                    success_count += 1
                else:
                    logger.warning(f"  [NO] No match found in OpenAlex")
                    
                    enriched_data["enriched_references"].append({
                        "ref_id": ref.get("ref_id", f"ref_{i}"),
                        "original_data": ref,
                        "enrichment_status": "not_found",
                        "consistency_status": "Not Checked",
                        "external_metadata": {}
                    })
                    
                    fail_count += 1
                
                # Rate limiting between API calls
                time.sleep(self.api_delay)
            
            except Exception as e:
                logger.error(f"[{i}/{len(references)}] Error processing reference: {e}")
                
                enriched_data["enriched_references"].append({
                    "ref_id": ref.get("ref_id", f"ref_{i}"),
                    "original_data": ref,
                    "enrichment_status": "error",
                    "consistency_status": "Error",
                    "external_metadata": {},
                    "error_message": str(e)
                })
                
                fail_count += 1
                continue
        
        # Add enrichment summary
        total_refs = len(references)
        enriched_data["enrichment_summary"] = {
            "total_references": total_refs,
            "successfully_enriched": success_count,
            "failed": fail_count,
            "cached": cached_count,
            "metadata_mismatches": mismatch_count,
            "success_rate": f"{(success_count/total_refs*100):.1f}%" if total_refs > 0 else "0%"
        }
        
        # Log final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("Enrichment Summary:")
        logger.info(f"  Total references: {total_refs}")
        logger.info(f"  Successfully enriched: {success_count} ({cached_count} from cache)")
        logger.info(f"  Failed/Not found: {fail_count}")
        logger.info(f"  Metadata mismatches: {mismatch_count}")
        logger.info(f"  Success rate: {enriched_data['enrichment_summary']['success_rate']}")
        logger.info("=" * 80)
        
        return enriched_data
