"""
CitePrism Relevance Scorer - Complete Scoring Pipeline
=======================================================
Combines embedding similarity, LLM judgment, and self-citation detection
to produce final relevance scores for all references.

Updated to use HuggingFace Inference API for batch processing.

Author: CitePrism Team
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

# Embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    SentenceTransformer = None
    np = None

# HuggingFace API
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

# Fuzzy matching for self-citations
try:
    from thefuzz import fuzz
except ImportError:
    fuzz = None

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# SELF-CITATION DETECTOR
# ============================================================================

class SelfCitationDetector:
    """Detects author, team, and venue overlaps between manuscript and references."""
    
    def __init__(self, config):
        """Initialize detector."""
        self.author_threshold = getattr(config, 'AUTHOR_SIMILARITY_THRESHOLD', 85)
        logger.info(f"Self-citation detector initialized (threshold: {self.author_threshold}%)")
    
    def detect(self, manuscript_authors: List[str], reference_authors: List[str]) -> Dict:
        """Detect self-citation based on author overlap."""
        if not manuscript_authors or not reference_authors:
            return {
                "is_self_cite": False,
                "overlap_type": "none",
                "matching_authors": [],
                "overlap_percentage": 0.0
            }
        
        # Normalize author names
        ms_authors_norm = [self._normalize_name(a) for a in manuscript_authors]
        ref_authors_norm = [self._normalize_name(a) for a in reference_authors]
        
        # Find matches using fuzzy matching
        matches = []
        for ms_author in ms_authors_norm:
            for ref_author in ref_authors_norm:
                if fuzz:
                    similarity = fuzz.ratio(ms_author, ref_author)
                    if similarity >= self.author_threshold:
                        matches.append((ms_author, ref_author, similarity))
                else:
                    if ms_author == ref_author:
                        matches.append((ms_author, ref_author, 100))
        
        if not matches:
            return {
                "is_self_cite": False,
                "overlap_type": "none",
                "matching_authors": [],
                "overlap_percentage": 0.0
            }
        
        overlap_pct = (len(matches) / len(ms_authors_norm)) * 100
        
        if len(matches) == len(ms_authors_norm):
            overlap_type = "full_team"
        elif len(matches) >= 2:
            overlap_type = "partial_team"
        else:
            overlap_type = "single_author"
        
        return {
            "is_self_cite": True,
            "overlap_type": overlap_type,
            "matching_authors": [m[0] for m in matches],
            "overlap_percentage": round(overlap_pct, 1)
        }
    
    def _normalize_name(self, name: str) -> str:
        """Normalize author name for comparison."""
        if not name:
            return ""
        normalized = " ".join(name.lower().split())
        for char in [',', '.', '-', "'", '"']:
            normalized = normalized.replace(char, ' ')
        return " ".join(normalized.split())


# ============================================================================
# EMBEDDING SCORER
# ============================================================================

class EmbeddingScorer:
    """Computes cosine similarity between manuscript and references using embeddings."""
    
    def __init__(self, config):
        """Initialize embedding model."""
        self.model_name = getattr(config, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed")
        
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Embedding model loaded: {self.model_name}")
    
    def compute_similarity(self, manuscript_profile: str, reference_profile: str) -> float:
        """Compute cosine similarity between manuscript and reference."""
        if not manuscript_profile or not reference_profile:
            return 0.0
        
        try:
            ms_embedding = self.model.encode(manuscript_profile, convert_to_numpy=True)
            ref_embedding = self.model.encode(reference_profile, convert_to_numpy=True)
            
            similarity = np.dot(ms_embedding, ref_embedding) / (
                np.linalg.norm(ms_embedding) * np.linalg.norm(ref_embedding)
            )
            
            score = float(max(0.0, min(100.0, similarity * 100)))
            return round(score, 2)
        
        except Exception as e:
            logger.error(f"Embedding similarity failed: {e}")
            return 0.0


# ============================================================================
# HF LLM BATCH SCORER
# ============================================================================

class HFBatchScorer:
    """Uses HuggingFace Inference API to score references in batches."""
    
    def __init__(self, config):
        """Initialize HF batch scorer."""
        self.api_token = getattr(config, 'HF_API_TOKEN', None)
        self.model = getattr(config, 'HF_MODEL', 'meta-llama/Llama-3.1-8B-Instruct')
        self.batch_size = getattr(config, 'HF_BATCH_SIZE', 5)
        
        if not self.api_token:
            raise ValueError("HF_API_TOKEN not set in configuration")
        
        if InferenceClient is None:
            raise ImportError("huggingface_hub not installed")
        
        self.client = InferenceClient(token=self.api_token)
        logger.info(f"HF Batch Scorer initialized: {self.model} (batch size: {self.batch_size})")
    
    def judge_batch(self, manuscript_context: str, batch_data: List[Dict]) -> List[Dict]:
        """Score a batch of references using HF API."""
        if not batch_data:
            return []
        
        prompt = self._build_batch_prompt(manuscript_context, batch_data)
        
        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert academic reviewer. You MUST return ONLY a valid JSON array. No markdown, no explanation, no preamble. Start with [ and end with ]."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,  # Increased for longer responses
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Aggressive cleaning
            # 1. Remove markdown code blocks
            if "```" in response_text:
                # Extract content between first and last ```
                parts = response_text.split("```")
                if len(parts) >= 3:
                    response_text = parts[1]
                    # Remove 'json' language identifier
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
            
            # 2. Find JSON array boundaries
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            if start_idx == -1 or end_idx == -1:
                logger.error("No JSON array found in response")
                logger.debug(f"Response: {response_text[:200]}")
                return self._default_batch_response(batch_data)
            
            response_text = response_text[start_idx:end_idx + 1]
            
            # 3. Fix common JSON issues
            # Remove trailing commas before ] or }
            import re
            response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
            
            # 4. Try parsing
            try:
                results = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                logger.debug(f"Cleaned text: {response_text[:500]}")
                
                # Last resort: try json-repair
                try:
                    import json_repair
                    results = json_repair.loads(response_text)
                    logger.info("Successfully repaired JSON")
                except:
                    return self._default_batch_response(batch_data)
            
            # 5. Validate it's a list
            if not isinstance(results, list):
                logger.error(f"Response is not a list: {type(results)}")
                return self._default_batch_response(batch_data)
            
            # 6. Normalize results
            normalized_results = []
            for i, result in enumerate(results):
                if i >= len(batch_data):
                    break  # Don't process more than we requested
                
                try:
                    score = result.get('relevance_score', 50)
                    score = max(0, min(100, int(score)))  # Convert to int first
                    
                    normalized_results.append({
                        "ref_id": result.get('ref_id', batch_data[i]['ref_id']),
                        "relevance_score_llm": score,
                        "label": result.get('label', 'borderline'),
                        "rationale": result.get('rationale', 'No rationale')[:500],  # Limit length
                        "evidence": result.get('evidence', [])[:3]  # Limit to 3 pieces
                    })
                except Exception as e:
                    logger.error(f"Error normalizing result {i}: {e}")
                    normalized_results.append({
                        "ref_id": batch_data[i]['ref_id'],
                        "relevance_score_llm": 50,
                        "label": "borderline",
                        "rationale": "Parse error",
                        "evidence": []
                    })
            
            # Fill missing refs with defaults
            while len(normalized_results) < len(batch_data):
                idx = len(normalized_results)
                normalized_results.append({
                    "ref_id": batch_data[idx]['ref_id'],
                    "relevance_score_llm": 50,
                    "label": "borderline",
                    "rationale": "Missing from LLM response",
                    "evidence": []
                })
            
            logger.info(f" --> Successfully parsed {len(normalized_results)}/{len(batch_data)} refs")
            return normalized_results
        
        except Exception as e:
            logger.error(f"HF batch scoring failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._default_batch_response(batch_data)
    
    def _build_batch_prompt(self, manuscript_context: str, batch_data: List[Dict]) -> str:
        """Build prompt for batch scoring."""
        refs_text = []
        for i, ref in enumerate(batch_data, 1):
            citation_text = "\n".join([f"    - {ctx[:200]}" for ctx in ref['contexts'][:3]]) if ref['contexts'] else "    No contexts"
            
            refs_text.append(f"""
Reference {i}:
  ID: {ref['ref_id']}
  Title: {ref['title']}
  Abstract: {ref['abstract'][:400] if ref['abstract'] else 'No abstract'}
  Citations: {citation_text}
""")
        
        return f"""Judge relevance of these references to the manuscript.

Manuscript: {manuscript_context[:800]}

References:
{''.join(refs_text)}

Score 80-100 = relevant, 40-79 = borderline, 0-39 = irrelevant

Return JSON array:
[
  {{"ref_id": "<ID>", "relevance_score": <0-100>, "label": "<relevant|borderline|irrelevant>", "rationale": "<why>", "evidence": ["<quote>"]}},
  ...
]"""
    
    def _default_batch_response(self, batch_data: List[Dict]) -> List[Dict]:
        """Return default responses for failed batch."""
        return [
            {
                "ref_id": ref['ref_id'],
                "relevance_score_llm": 50,
                "label": "borderline",
                "rationale": "LLM unavailable",
                "evidence": []
            }
            for ref in batch_data
        ]


# ============================================================================
# HYBRID RELEVANCE SCORER
# ============================================================================

class RelevanceScorer:
    """Main scorer combining embeddings, LLM judgment, and self-citations."""
    
    def __init__(self, config, db_manager):
        """Initialize relevance scorer."""
        self.config = config
        self.db = db_manager
        
        self.llm_weight = getattr(config, 'LLM_WEIGHT', 0.6)
        self.embedding_weight = getattr(config, 'EMBEDDING_WEIGHT', 0.4)
        self.relevant_threshold = getattr(config, 'RELEVANT_THRESHOLD', 70)
        self.borderline_threshold = getattr(config, 'BORDERLINE_THRESHOLD', 40)
        self.batch_size = getattr(config, 'HF_BATCH_SIZE', 5)
        
        self.self_cite_detector = SelfCitationDetector(config)
        self.embedding_scorer = EmbeddingScorer(config)
        self.llm_scorer = HFBatchScorer(config)
        
        logger.info(f"Scorer initialized: LLM={self.llm_weight}, Embed={self.embedding_weight}, Batch={self.batch_size}")
    
    def score_references(self, enriched_data: Dict) -> Dict:
        """Score all references."""
        logger.info("=" * 80)
        logger.info("Starting relevance scoring...")
        logger.info("=" * 80)
        
        manuscript_meta = enriched_data.get("manuscript_metadata", {})
        manuscript_profile = self._build_manuscript_profile(manuscript_meta)
        manuscript_authors = manuscript_meta.get("authors", [])
        manuscript_context = f"{manuscript_meta.get('title', '')} {manuscript_meta.get('abstract', '')}"[:1500]
        
        references = enriched_data.get("enriched_references", [])
        citations_in_text = enriched_data.get("citations_in_text", [])
        
        logger.info(f"Found {len(references)} references to score")
        
        if not references:
            return self._empty_result(enriched_data)
        
        # Step 1: Embeddings (fast, local)
        logger.info("Step 1: Computing embeddings...")
        ref_embeddings = self._compute_all_embeddings(references, manuscript_profile, citations_in_text)
        
        # Step 2: Batch LLM (5-10 refs per API call)
        logger.info(f"Step 2: Batch LLM scoring ({self.batch_size} refs/call)...")
        llm_results = self._batch_process_llm(references, manuscript_context, citations_in_text)
        
        # Step 3: Combine
        logger.info("Step 3: Computing final scores...")
        scored_refs = []
        self_cite_count = 0
        low_relevance_count = 0
        
        for i, ref in enumerate(references):
            ref_id = ref.get('ref_id', f'ref_{i+1}')
            
            # Get scores
            embed_data = ref_embeddings.get(ref_id, {})
            llm_data = llm_results.get(ref_id, {})
            
            # Self-citation
            original_data = ref.get("original_data", {})
            external_meta = ref.get("external_metadata", {})
            ref_authors = external_meta.get("authors", [])
            ref_author_names = [a.get("display_name", "") for a in ref_authors if isinstance(a, dict)]
            
            self_cite_result = self.self_cite_detector.detect(manuscript_authors, ref_author_names)
            
            # Hybrid score
            rs_embed = embed_data.get('score', 0.0)
            rs_llm = llm_data.get('relevance_score_llm', 50)
            rs_final = round((self.llm_weight * rs_llm) + (self.embedding_weight * rs_embed), 2)
            
            # Label
            if rs_final >= self.relevant_threshold:
                final_label = "relevant"
            elif rs_final >= self.borderline_threshold:
                final_label = "borderline"
            else:
                final_label = "irrelevant"
            
            # Flags
            flags = []
            if rs_final < self.borderline_threshold:
                flags.append("low_relevance")
            if abs(rs_embed - rs_llm) > 30:
                flags.append("score_discrepancy")
            if not external_meta.get("abstract"):
                flags.append("missing_abstract")
            if self_cite_result["is_self_cite"] and rs_final < self.borderline_threshold:
                flags.append("questionable_self_cite")
            if external_meta.get("is_retracted"):
                flags.append("RETRACTED")
            
            scored_ref = {
                **ref,
                "self_citation": self_cite_result,
                "RS_embed": rs_embed,
                "RS_llm": rs_llm,
                "RS_final": rs_final,
                "label": final_label,
                "llm_rationale": llm_data.get('rationale', ''),
                "llm_evidence": llm_data.get('evidence', []),
                "quality_flags": flags
            }
            
            scored_refs.append(scored_ref)
            
            if self_cite_result.get("is_self_cite"):
                self_cite_count += 1
            if rs_final < self.borderline_threshold:
                low_relevance_count += 1
        
        result = {
            "manuscript_metadata": manuscript_meta,
            "citations_in_text": citations_in_text,
            "scored_references": scored_refs,
            "scoring_summary": {
                "total_references": len(references),
                "self_citations": self_cite_count,
                "low_relevance": low_relevance_count,
                "self_citation_rate": f"{(self_cite_count/len(references)*100):.1f}%" if references else "0%"
            }
        }
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("Scoring Summary:")
        logger.info(f"  Total: {len(references)}")
        logger.info(f"  Self-citations: {self_cite_count}")
        logger.info(f"  Low relevance: {low_relevance_count}")
        logger.info("=" * 80)
        
        return result
    
    def _compute_all_embeddings(self, references, manuscript_profile, citations_in_text):
        """Compute embedding scores for all references."""
        results = {}
        
        for i, ref in enumerate(references, 1):
            ref_id = ref.get('ref_id', f'ref_{i}')
            original_data = ref.get("original_data", {})
            parsed = original_data.get("parsed", {})
            external_meta = ref.get("external_metadata", {})
            
            ref_title = parsed.get("title", "")
            ref_abstract = external_meta.get("abstract", "")
            ref_profile = self._build_reference_profile(ref_title, ref_abstract)
            
            score = self.embedding_scorer.compute_similarity(manuscript_profile, ref_profile)
            results[ref_id] = {'score': score}
        
        logger.info(f"  Computed {len(results)} embeddings")
        return results
    
    def _batch_process_llm(self, references, manuscript_context, citations_in_text):
        """Process LLM judgments in batches."""
        results = {}
        total_refs = len(references)
        num_batches = (total_refs + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_refs)
            batch_refs = references[start_idx:end_idx]
            
            logger.info(f"  Batch {batch_idx + 1}/{num_batches} (refs {start_idx + 1}-{end_idx})")
            
            # Prepare batch
            batch_data = []
            for ref in batch_refs:
                ref_id = ref.get('ref_id', '')
                original_data = ref.get("original_data", {})
                parsed = original_data.get("parsed", {})
                external_meta = ref.get("external_metadata", {})
                
                batch_data.append({
                    'ref_id': ref_id,
                    'title': parsed.get("title", ""),
                    'abstract': external_meta.get("abstract", ""),
                    'contexts': self._find_citation_contexts(ref_id, citations_in_text)
                })
            
            # Call HF API
            batch_results = self.llm_scorer.judge_batch(manuscript_context, batch_data)
            
            for result in batch_results:
                results[result['ref_id']] = result
            
            # Rate limiting
            if batch_idx < num_batches - 1:
                time.sleep(1)
        
        logger.info(f"  Completed LLM scoring for {len(results)} references")
        return results
    
    def _build_manuscript_profile(self, metadata):
        """Build manuscript profile."""
        title = metadata.get("title", "")
        abstract = metadata.get("abstract", "")
        return f"{title} {title} {title} {abstract}"
    
    def _build_reference_profile(self, title, abstract):
        """Build reference profile."""
        profile = f"{title} {title}"
        if abstract:
            profile += f" {abstract}"
        return profile
    
    def _find_citation_contexts(self, ref_id, citations):
        """Find citation contexts for reference."""
        contexts = []
        for citation in citations:
            marker = citation.get("marker", "")
            if ref_id in marker or marker in ref_id:
                contexts.append(citation.get("context_window", ""))
        return contexts
    
    def _empty_result(self, enriched_data):
        """Empty result."""
        return {
            "manuscript_metadata": enriched_data.get("manuscript_metadata", {}),
            "citations_in_text": enriched_data.get("citations_in_text", []),
            "scored_references": [],
            "scoring_summary": {
                "total_references": 0,
                "self_citations": 0,
                "low_relevance": 0,
                "self_citation_rate": "0%"
            }
        }