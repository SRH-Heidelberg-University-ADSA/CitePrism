"""
CitePrism Phase 2: Metadata Enrichment (Updated with Enhanced Error Handling)
==============================================================================
Takes parsed citations and enriches them with abstracts and canonical metadata 
using the OpenAlex API. Includes fuzzy validation for bibliographic variance.

Features:
- Automatically processes all *_parsed*.json files in output folder
- Saves enriched files to enriched_output folder
- Comprehensive exception handling throughout
- Progress tracking and detailed logging

Author: CitePrism Team
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from difflib import SequenceMatcher  # Standard library for string comparison

# External API Wrapper
try:
    from pyalex import Works, config
    config.email = "gowrikamahesh2017@gmail.com"  # Good practice for API courtesy
except ImportError:
    raise ImportError("pyalex not installed. Run: pip install pyalex")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for CitePrism Phase 2."""
    
    # Input directory (where parsed JSON files are located)
    INPUT_DIR: Path = Path("output")
    
    # Output directory (where enriched JSON files will be saved)
    OUTPUT_DIR: Path = Path("enriched_output")
    
    # File pattern to match (any file containing "parsed" in the name)
    FILE_PATTERN: str = "*parsed*.json"
    
    # API rate limiting (seconds between requests)
    RATE_LIMIT_DELAY: float = 0.1
    
    # Fuzzy matching threshold for titles (0.0 to 1.0)
    TITLE_SIMILARITY_THRESHOLD: float = 0.7
    
    # Year difference tolerance
    YEAR_DIFFERENCE_TOLERANCE: int = 1


# ============================================================================
# VALIDATION LOGIC
# ============================================================================

def check_metadata_consistency(
    parsed_year: Optional[int],
    enriched_year: Optional[int],
    parsed_title: str,
    enriched_title: str,
    title_threshold: float = 0.7
) -> str:
    """
    Validates consistency between the PDF's citation and the API result.
    Handles 'Bibliographic Variance' (e.g., Conference vs. Proceedings dates).
    
    Args:
        parsed_year: Year from parsed PDF citation
        enriched_year: Year from OpenAlex API
        parsed_title: Title from parsed PDF citation
        enriched_title: Title from OpenAlex API
        title_threshold: Minimum similarity ratio for title matching (0.0-1.0)
    
    Returns:
        Status string indicating the validation result
    """
    try:
        # 1. Title Safety Check (Prevent retrieving completely wrong papers)
        # Calculate similarity ratio (0.0 to 1.0)
        if not parsed_title or not enriched_title:
            return "Incomplete Metadata (Missing Title)"
        
        title_sim = SequenceMatcher(None, parsed_title.lower(), enriched_title.lower()).ratio()
        
        if title_sim < title_threshold:  # If titles are less than threshold% similar
            return f"Mismatch Flagged (Title Similarity: {int(title_sim*100)}%)"

        # 2. Year Verification
        if parsed_year and enriched_year:
            try:
                diff = abs(parsed_year - enriched_year)
                
                if diff == 0:
                    return "Match"
                elif diff <= 1:
                    # This handles the 2006 vs 2007 issue (conference vs. proceedings dates)
                    return "Acceptable Variance (±1 Year)"
                else:
                    return f"Mismatch Flagged (Year Diff: {diff} years)"
            except (TypeError, ValueError) as e:
                logger.warning(f"Error comparing years: {e}")
                return "Incomplete Metadata (Invalid Year Format)"
                
        return "Incomplete Metadata (Missing Year)"
    
    except Exception as e:
        logger.error(f"Error in metadata consistency check: {e}")
        return f"Validation Error: {str(e)}"


# ============================================================================
# API SEARCH
# ============================================================================

def search_openalex(title: str, authors: List[str], retry_count: int = 3) -> Optional[Dict]:
    """
    Query OpenAlex for a paper match with retry logic.
    
    Args:
        title: Paper title to search for
        authors: List of author names (for future filtering)
        retry_count: Number of retry attempts on failure
    
    Returns:
        Dictionary with enriched metadata or None if not found
    """
    if not title or not title.strip():
        logger.warning("Empty title provided to search_openalex")
        return None
    
    # Clean title for better search results
    try:
        cleaned_title = title.strip()
    except Exception as e:
        logger.error(f"Error cleaning title: {e}")
        return None
    
    for attempt in range(retry_count):
        try:
            # 1. Search for the work by title
            results = Works().search(cleaned_title).get()
            
            if not results:
                logger.debug(f"No results found for: {cleaned_title[:30]}...")
                return None
                
            # 2. Naive 'Best Match' - take the first result
            # Note: In production, you might want to score results based on author matching
            best_match = results[0]
            
            if not best_match:
                return None
            
            # 3. Extract Abstract (OpenAlex stores it as an inverted index)
            abstract_text = None
            try:
                if best_match.get('abstract_inverted_index'):
                    index = best_match['abstract_inverted_index']
                    
                    # Find the maximum position to create word list
                    max_pos = max([max(pos) for pos in index.values() if pos])
                    word_list = [""] * (max_pos + 1)
                    
                    # Reconstruct abstract from inverted index
                    for word, positions in index.items():
                        for pos in positions:
                            if pos < len(word_list):
                                word_list[pos] = word
                    
                    abstract_text = " ".join(word_list).strip()
            
            except Exception as e:
                logger.warning(f"Error extracting abstract: {e}")
                abstract_text = None
            
            # 4. Construct enriched metadata
            try:
                enriched_data = {
                    "id": best_match.get('id'),
                    "title": best_match.get('title'),
                    "year": best_match.get('publication_year'),
                    "cited_by_count": best_match.get('cited_by_count', 0),
                    "is_retracted": best_match.get('is_retracted', False),
                    "abstract": abstract_text,
                    "doi": best_match.get('doi'),
                    "type": best_match.get('type'),
                    "open_access": best_match.get('open_access', {})
                }
                
                return enriched_data
            
            except Exception as e:
                logger.error(f"Error constructing enriched data: {e}")
                return None

        except ConnectionError as e:
            logger.warning(f"Connection error on attempt {attempt + 1}/{retry_count}: {e}")
            if attempt < retry_count - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            else:
                logger.error(f"Failed after {retry_count} attempts")
                return None
        
        except Exception as e:
            logger.warning(f"API Error for '{cleaned_title[:30]}...': {e}")
            if attempt < retry_count - 1:
                time.sleep(0.5)
                continue
            else:
                return None
    
    return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def enrich_manuscript(input_path: Path, output_path: Path, config: Config) -> bool:
    """
    Load parsed JSON -> Enrich References -> Validate -> Save.
    
    Args:
        input_path: Path to input parsed JSON file
        output_path: Path to save enriched JSON file
        config: Configuration object
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing: {input_path.name}")
    
    # 1. Validate input file exists
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return False
    
    # 2. Check file is readable
    if not input_path.is_file():
        logger.error(f"Input path is not a file: {input_path}")
        return False
    
    # 3. Check file size (warn if too small or too large)
    try:
        file_size = input_path.stat().st_size
        if file_size == 0:
            logger.error(f"Input file is empty: {input_path}")
            return False
        elif file_size < 100:
            logger.warning(f"Input file is suspiciously small ({file_size} bytes)")
    except Exception as e:
        logger.warning(f"Could not check file size: {e}")

    # 4. Load JSON data
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✓ Successfully loaded JSON from {input_path.name}")
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        return False
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return False
    
    # 5. Validate JSON structure
    if not isinstance(data, dict):
        logger.error(f"Input JSON is not a dictionary")
        return False
    
    # 6. Extract references list
    references_list = data.get('references_list', [])
    
    if not references_list:
        logger.warning(f"No references found in {input_path.name}")
        # Still continue - might be valid to have no references
    
    if not isinstance(references_list, list):
        logger.error(f"'references_list' is not a list")
        return False
    
    logger.info(f"Found {len(references_list)} reference(s) to enrich")
    
    # 7. Initialize enrichment
    enriched_refs = []
    total_refs = len(references_list)
    success_count = 0
    failed_count = 0
    
    # 8. Process each reference
    for idx, ref in enumerate(references_list):
        try:
            # Validate reference structure
            if not isinstance(ref, dict):
                logger.warning(f"[{idx+1}/{total_refs}] Reference is not a dictionary, skipping")
                continue
            
            parsed = ref.get('parsed', {})
            if not isinstance(parsed, dict):
                logger.warning(f"[{idx+1}/{total_refs}] 'parsed' field is not a dictionary, skipping")
                continue
            
            title = parsed.get('title')
            parsed_year = parsed.get('year')
            authors = parsed.get('authors', [])
            
            # Ensure authors is a list
            if not isinstance(authors, list):
                authors = []
            
            # Log current reference
            if title:
                logger.info(f"[{idx+1}/{total_refs}] Enriching: {title[:50]}...")
            else:
                logger.warning(f"[{idx+1}/{total_refs}] Reference has no title, skipping enrichment")
            
            # Initialize enriched entry
            enriched_entry = {
                "ref_id": ref.get("ref_id"),
                "original_data": ref,
                "enrichment_status": "failed",
                "consistency_status": "Not Checked",
                "external_metadata": {}
            }
            
            # Try to enrich if title exists
            if title and title.strip():
                try:
                    # Call API with retry logic
                    api_result = search_openalex(title, authors)
                    
                    if api_result:
                        enriched_entry["enrichment_status"] = "success"
                        enriched_entry["external_metadata"] = api_result
                        success_count += 1
                        
                        # Perform metadata consistency validation
                        try:
                            enriched_year = api_result.get('year')
                            enriched_title = api_result.get('title', '')
                            
                            status = check_metadata_consistency(
                                parsed_year,
                                enriched_year,
                                title,
                                enriched_title,
                                config.TITLE_SIMILARITY_THRESHOLD
                            )
                            enriched_entry["consistency_status"] = status
                            
                            # Log validation outcome
                            if "Mismatch" in status:
                                logger.warning(f"  ⚠ {status}")
                            elif "Variance" in status:
                                logger.info(f"  ℹ {status}")
                            else:
                                logger.info(f"  ✓ {status}")

                            # Check for retraction (CRITICAL)
                            if api_result.get('is_retracted'):
                                logger.warning(f"  ⚠ CRITICAL: Reference {ref.get('ref_id')} is RETRACTED!")
                                enriched_entry["retraction_flag"] = True
                        
                        except Exception as e:
                            logger.warning(f"  Error during validation: {e}")
                            enriched_entry["consistency_status"] = f"Validation Error: {str(e)}"
                    
                    else:
                        logger.warning(f"  ✗ No match found for: {title[:40]}...")
                        failed_count += 1
                
                except Exception as e:
                    logger.error(f"  Error enriching reference: {e}")
                    failed_count += 1
            else:
                logger.warning(f"  ✗ Skipping - no valid title")
                failed_count += 1
            
            enriched_refs.append(enriched_entry)
            
            # Polite rate limiting (respect API guidelines)
            if idx < total_refs - 1:  # Don't sleep after last item
                time.sleep(config.RATE_LIMIT_DELAY)
        
        except Exception as e:
            logger.error(f"[{idx+1}/{total_refs}] Unexpected error processing reference: {e}")
            # Add a placeholder entry to maintain alignment
            enriched_refs.append({
                "ref_id": ref.get("ref_id", f"unknown_{idx}"),
                "original_data": ref,
                "enrichment_status": "error",
                "consistency_status": f"Processing Error: {str(e)}",
                "external_metadata": {}
            })
            failed_count += 1
            continue

    # 9. Construct final output
    try:
        final_output = {
            "manuscript_metadata": data.get("metadata", {}),
            "citations_in_text": data.get("citations_in_text", []),
            "enriched_references": enriched_refs,
            "enrichment_summary": {
                "total_references": total_refs,
                "successfully_enriched": success_count,
                "failed_enrichment": failed_count,
                "success_rate": f"{(success_count/total_refs*100):.1f}%" if total_refs > 0 else "N/A"
            }
        }
    except Exception as e:
        logger.error(f"Error constructing final output: {e}")
        return False
    
    # 10. Save enriched data
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                final_output,
                f,
                indent=2,
                ensure_ascii=False
            )
        
        logger.info(f"✓ Enrichment complete. Saved to {output_path}")
        
        # Verify file was written
        if not output_path.exists():
            logger.error(f"Output file was not created: {output_path}")
            return False
        
        output_size = output_path.stat().st_size
        if output_size == 0:
            logger.error(f"Output file is empty: {output_path}")
            return False
        
        logger.info(f"  - Output file size: {output_size} bytes")
        logger.info(f"  - Success rate: {success_count}/{total_refs} ({(success_count/total_refs*100):.1f}%)" if total_refs > 0 else "  - No references to process")
        
        return True
    
    except PermissionError as e:
        logger.error(f"Permission denied when saving output: {e}")
        return False
    except IOError as e:
        logger.error(f"I/O error when saving output: {e}")
        return False
    except Exception as e:
        logger.error(f"Error saving enriched data: {e}")
        return False


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_all_parsed_files(config: Config) -> Dict[str, any]:
    """
    Automatically find and process all parsed JSON files in the input directory.
    
    Args:
        config: Configuration object
    
    Returns:
        Dictionary with processing statistics
    """
    logger.info("=" * 80)
    logger.info("CitePrism Phase 2 - Metadata Enrichment")
    logger.info("=" * 80)
    
    # 1. Validate input directory exists
    if not config.INPUT_DIR.exists():
        logger.error(f"Input directory does not exist: {config.INPUT_DIR}")
        logger.error(f"Please create the directory or check the path")
        return {"status": "failed", "error": "Input directory not found"}
    
    if not config.INPUT_DIR.is_dir():
        logger.error(f"Input path is not a directory: {config.INPUT_DIR}")
        return {"status": "failed", "error": "Input path is not a directory"}
    
    # 2. Find all matching files
    try:
        parsed_files = list(config.INPUT_DIR.glob(config.FILE_PATTERN))
        parsed_files.sort()  # Process in alphabetical order
    except Exception as e:
        logger.error(f"Error scanning input directory: {e}")
        return {"status": "failed", "error": f"Directory scan error: {str(e)}"}
    
    if not parsed_files:
        logger.warning(f"No files matching pattern '{config.FILE_PATTERN}' found in {config.INPUT_DIR}")
        logger.info(f"Looking for files like: *parsed*.json")
        return {"status": "no_files", "message": "No matching files found"}
    
    logger.info(f"Found {len(parsed_files)} file(s) to process:")
    for f in parsed_files:
        logger.info(f"  - {f.name}")
    logger.info("")
    
    # 3. Create output directory
    try:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Output directory ready: {config.OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return {"status": "failed", "error": f"Cannot create output directory: {str(e)}"}
    
    # 4. Process each file
    results = []
    for i, input_file in enumerate(parsed_files, 1):
        try:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[{i}/{len(parsed_files)}] Processing: {input_file.name}")
            logger.info("=" * 80)
            
            # Generate output filename (replace 'parsed' with 'enriched')
            output_filename = input_file.stem.replace('parsed', 'enriched') + '.json'
            output_file = config.OUTPUT_DIR / output_filename
            
            # Process the file
            success = enrich_manuscript(input_file, output_file, config)
            
            results.append({
                "input_file": input_file.name,
                "output_file": output_filename,
                "status": "success" if success else "failed"
            })
            
            if success:
                logger.info(f"✓ [{i}/{len(parsed_files)}] Successfully processed {input_file.name}")
            else:
                logger.error(f"✗ [{i}/{len(parsed_files)}] Failed to process {input_file.name}")
        
        except Exception as e:
            logger.error(f"✗ [{i}/{len(parsed_files)}] Unexpected error: {e}")
            results.append({
                "input_file": input_file.name,
                "output_file": "N/A",
                "status": "error",
                "error": str(e)
            })
    
    # 5. Generate summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count
    
    logger.info(f"Total files: {len(results)}")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("")
    
    if success_count > 0:
        logger.info("Successfully enriched files:")
        for result in results:
            if result["status"] == "success":
                logger.info(f"  ✓ {result['input_file']} → {result['output_file']}")
    
    if failed_count > 0:
        logger.info("")
        logger.info("Failed files:")
        for result in results:
            if result["status"] != "success":
                logger.info(f"  ✗ {result['input_file']}")
                if "error" in result:
                    logger.info(f"    Error: {result['error']}")
    
    logger.info("=" * 80)
    
    return {
        "status": "completed",
        "total": len(results),
        "successful": success_count,
        "failed": failed_count,
        "results": results
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the enrichment script."""
    try:
        # Initialize configuration
        config = Config()
        
        logger.info(f"Configuration:")
        logger.info(f"  - Input directory: {config.INPUT_DIR}")
        logger.info(f"  - Output directory: {config.OUTPUT_DIR}")
        logger.info(f"  - File pattern: {config.FILE_PATTERN}")
        logger.info(f"  - Title similarity threshold: {config.TITLE_SIMILARITY_THRESHOLD}")
        logger.info(f"  - Year difference tolerance: ±{config.YEAR_DIFFERENCE_TOLERANCE}")
        logger.info("")
        
        # Process all files
        summary = process_all_parsed_files(config)
        
        # Exit with appropriate code
        if summary.get("status") == "completed":
            failed = summary.get("failed", 0)
            sys.exit(0 if failed == 0 else 1)
        else:
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()