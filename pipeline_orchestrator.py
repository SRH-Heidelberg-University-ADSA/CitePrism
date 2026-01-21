"""
CitePrism Pipeline Orchestrator
================================
End-to-end pipeline with intelligent caching and per-stage force reprocess.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


class PipelineOrchestrator:
    """
    Orchestrates the complete CitePrism pipeline with intelligent caching.
    
    Pipeline Stages:
    1. PDF Upload & Registration
    2. Parsing (Gemini LLM)
    3. Enrichment (OpenAlex API)
    4. Scoring (Embeddings + LLM)
    """
    
    def __init__(self, db_manager, config):
        """Initialize pipeline orchestrator."""
        self.db = db_manager
        self.config = config
        
        # Use forward slashes or Path constructor for cross-platform compatibility
        self.data_dirs = {
            'raw_pdfs': Path('data/raw_pdfs'),
            'parsed': Path('data/parsed'),
            'enriched': Path('data/enriched'),
            'scored': Path('data/scored')
        }
        
        for dir_path in self.data_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_document(
        self, 
        pdf_path: Path, 
        force_parse: bool = False,
        force_enrich: bool = False,
        force_score: bool = False
    ) -> Dict:
        """
        Complete end-to-end processing pipeline with per-stage force options.
        
        Args:
            pdf_path: Path to PDF file
            force_parse: Force re-parsing even if already parsed
            force_enrich: Force re-enrichment even if already enriched
            force_score: Force re-scoring even if already scored
        
        Returns:
            Dictionary with processing results and file paths
        """
        results = {
            'success': True,
            'document_id': None,
            'stages_completed': [],
            'stages_skipped': [],
            'file_paths': {},
            'errors': []
        }
        
        try:
            # Stage 0: Register PDF
            results.update(self._stage_register_pdf(pdf_path))
            
            if not results['success']:
                return results
            
            document_id = results['document_id']
            status = self.db.get_document_status(document_id)
            
            # Stage 1: Parsing
            if force_parse or not status['status_parsed']:
                logger.info(f"[Doc {document_id}] Stage 1: Parsing PDF with LLM...")
                parse_result = self._stage_parse(document_id, pdf_path, force_parse)
                results.update(parse_result)
                
                if not parse_result['success']:
                    return results
            else:
                logger.info(f"[Doc {document_id}] Stage 1: Skipped (already parsed)")
                results['stages_skipped'].append('parsing')
                parsed_path = self._normalize_path(status['parsed_path'])
                results['file_paths']['parsed'] = parsed_path
            
            # Refresh status
            status = self.db.get_document_status(document_id)
            
            # Stage 2: Enrichment
            if force_enrich or not status['status_enriched']:
                logger.info(f"[Doc {document_id}] Stage 2: Enriching with OpenAlex...")
                enrich_result = self._stage_enrich(document_id, status['parsed_path'])
                results.update(enrich_result)
                
                if not enrich_result['success']:
                    return results
            else:
                logger.info(f"[Doc {document_id}] Stage 2: Skipped (already enriched)")
                results['stages_skipped'].append('enrichment')
                enriched_path = self._normalize_path(status['enriched_path'])
                results['file_paths']['enriched'] = enriched_path
            
            # Refresh status
            status = self.db.get_document_status(document_id)
            
            # Stage 3: Scoring
            if force_score or not status['status_scored']:
                logger.info(f"[Doc {document_id}] Stage 3: Scoring (Embeddings + LLM)...")
                score_result = self._stage_score(document_id, status['enriched_path'])
                results.update(score_result)
                
                if not score_result['success']:
                    return results
            else:
                logger.info(f"[Doc {document_id}] Stage 3: Skipped (already scored)")
                results['stages_skipped'].append('scoring')
                scored_path = self._normalize_path(status['scored_path'])
                results['file_paths']['scored'] = scored_path
            
            logger.info(f"[Doc {document_id}] [SUCCESS] Pipeline completed!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results['success'] = False
            results['errors'].append(str(e))
        
        return results
    
    def _normalize_path(self, path_str: Optional[str]) -> Optional[str]:
        """Normalize path string to use forward slashes and convert to Path object."""
        if not path_str:
            return None
        # Replace backslashes with forward slashes and create Path object
        normalized = Path(path_str.replace('\\', '/'))
        return str(normalized)
    
    def _stage_register_pdf(self, pdf_path: Path) -> Dict:
        """Stage 0: Register PDF in database."""
        try:
            # Convert both paths to absolute Path objects for comparison
            pdf_path = Path(pdf_path)
            raw_pdfs_dir = self.data_dirs['raw_pdfs'].resolve()
            
            # Check if PDF is already in the raw_pdfs directory
            if pdf_path.parent.resolve() != raw_pdfs_dir:
                dest_path = raw_pdfs_dir / pdf_path.name
                if not dest_path.exists():
                    import shutil
                    shutil.copy2(pdf_path, dest_path)
                    logger.info(f"Copied PDF to: {dest_path}")
                pdf_path = dest_path
            
            document_id, is_new = self.db.register_pdf(pdf_path)
            
            return {
                'success': True,
                'document_id': document_id,
                'is_new_document': is_new,
                'file_paths': {'pdf': str(pdf_path)}
            }
        
        except Exception as e:
            logger.error(f"Failed to register PDF: {e}")
            return {
                'success': False,
                'errors': [f"Registration failed: {str(e)}"]
            }
    
    def _stage_parse(self, document_id: int, pdf_path: Path, force_reprocess: bool = False) -> Dict:
        """Stage 1: Parse PDF with LLM."""
        try:
            from src.extractors.gemini_extractor import GeminiExtractor
            
            extractor = GeminiExtractor(self.config)
            
            output_filename = f"{pdf_path.stem}_parsed.json"
            output_path = self.data_dirs['parsed'] / output_filename
            
            # Check cache unless force_reprocess
            parsed_data = None
            if not force_reprocess:
                cache_key = f"{pdf_path.name}:v1"
                cached = self.db.get_cached_response('parsing', cache_key)
                
                if cached:
                    logger.info("Using cached parsing result")
                    parsed_data = cached
            
            # Extract using LLM if needed
            if parsed_data is None:
                logger.info("Extracting text from PDF and parsing with LLM...")
                parsed_data = extractor.extract(pdf_path)
                
                cache_key = f"{pdf_path.name}:v1"
                self.db.cache_api_response('parsing', cache_key, parsed_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            
            metadata = parsed_data.get('metadata', {})
            self.db.update_parsed_status(
                document_id=document_id,
                parsed_path=str(output_path),
                title=metadata.get('title'),
                authors=json.dumps(metadata.get('authors', [])),
                num_refs=len(parsed_data.get('references_list', []))
            )
            
            logger.info(f"[SUCCESS] Parsing completed: {output_path}")
            
            return {
                'success': True,
                'stages_completed': ['parsing'],
                'file_paths': {'parsed': str(output_path)},
                'metadata': {
                    'title': metadata.get('title'),
                    'num_references': len(parsed_data.get('references_list', []))
                }
            }
        
        except Exception as e:
            logger.error(f"Parsing failed: {e}", exc_info=True)
            self.db.log_error(document_id, 'parsing', str(e))
            return {
                'success': False,
                'errors': [f"Parsing failed: {str(e)}"]
            }
    
    def _stage_enrich(self, document_id: int, parsed_path: str) -> Dict:
        """Stage 2: Enrich with OpenAlex API."""
        try:
            from src.enrichers.openalex_enricher import OpenAlexEnricher
            
            enricher = OpenAlexEnricher(self.config, self.db)
            
            # Normalize the path before using it
            parsed_path_normalized = self._normalize_path(parsed_path)
            parsed_path_obj = Path(parsed_path_normalized)
            
            with open(parsed_path_obj, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)
            
            output_filename = parsed_path_obj.name.replace('_parsed', '_enriched')
            output_path = self.data_dirs['enriched'] / output_filename
            
            enriched_data = enricher.enrich_references(parsed_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, indent=2, ensure_ascii=False)
            
            summary = enriched_data.get('enrichment_summary', {})
            self.db.update_enriched_status(
                document_id=document_id,
                enriched_path=str(output_path),
                success_count=summary.get('successfully_enriched', 0),
                total_count=summary.get('total_references', 0)
            )
            
            logger.info(f"[SUCCESS] Enrichment completed: {output_path}")
            
            return {
                'success': True,
                'stages_completed': ['enrichment'],
                'file_paths': {'enriched': str(output_path)},
                'metadata': {
                    'enriched_count': summary.get('successfully_enriched', 0)
                }
            }
        
        except Exception as e:
            logger.error(f"Enrichment failed: {e}", exc_info=True)
            self.db.log_error(document_id, 'enrichment', str(e))
            return {
                'success': False,
                'errors': [f"Enrichment failed: {str(e)}"]
            }
    
    def _stage_score(self, document_id: int, enriched_path: str) -> Dict:
        """Stage 3: Score references (embeddings + LLM)."""
        try:
            from src.scorers.relevance_scorer import RelevanceScorer
            
            scorer = RelevanceScorer(self.config, self.db)
            
            # Normalize the path before using it
            enriched_path_normalized = self._normalize_path(enriched_path)
            enriched_path_obj = Path(enriched_path_normalized)
            
            with open(enriched_path_obj, 'r', encoding='utf-8') as f:
                enriched_data = json.load(f)
            
            output_filename = enriched_path_obj.name.replace('_enriched', '_scored')
            output_path = self.data_dirs['scored'] / output_filename
            
            scored_data = scorer.score_references(enriched_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scored_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            scoring_meta = scored_data.get('scoring_metadata', {})
            self.db.update_scored_status(
                document_id=document_id,
                scored_path=str(output_path),
                embedding_model=scoring_meta.get('embedding_model'),
                llm_model=scoring_meta.get('llm_model')
            )
            
            logger.info(f"[SUCCESS] Scoring completed: {output_path}")
            
            return {
                'success': True,
                'stages_completed': ['scoring'],
                'file_paths': {'scored': str(output_path)}
            }
        
        except Exception as e:
            logger.error(f"Scoring failed: {e}", exc_info=True)
            self.db.log_error(document_id, 'scoring', str(e))
            return {
                'success': False,
                'errors': [f"Scoring failed: {str(e)}"]
            }
    
    def get_document_files(self, document_id: int) -> Dict[str, Optional[str]]:
        """Get all file paths for a document."""
        status = self.db.get_document_status(document_id)
        
        if not status:
            return {}
        
        # Normalize all paths before returning
        return {
            'pdf': self._normalize_path(status.get('pdf_path')),
            'parsed': self._normalize_path(status.get('parsed_path')),
            'enriched': self._normalize_path(status.get('enriched_path')),
            'scored': self._normalize_path(status.get('scored_path'))
        }
    
    def check_stage_completion(self, document_id: int) -> Dict[str, bool]:
        """Check which stages are completed for a document."""
        status = self.db.get_document_status(document_id)
        
        if not status:
            return {
                'parsed': False,
                'enriched': False,
                'scored': False
            }
        
        return {
            'parsed': bool(status['status_parsed']),
            'enriched': bool(status['status_enriched']),
            'scored': bool(status['status_scored'])
        }