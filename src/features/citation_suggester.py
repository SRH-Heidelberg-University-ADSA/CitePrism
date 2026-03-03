"""
CitePrism Phase 6.2: Missing Citations Suggester
================================================
Reads the manuscript abstract and existing citations, then asks the LLM
to recommend seminal papers the author missed.
"""

import json
import logging
from pathlib import Path
import google.generativeai as genai

logger = logging.getLogger(__name__)

def generate_missing_citations(scored_json_path: Path, config):
    """
    Calls Gemini to suggest missing citations based on the manuscript's abstract.
    """
    logger.info(f"Generating Missing Citation Suggestions for: {scored_json_path}")
    
    if not scored_json_path.exists():
        logger.error("Scored JSON not found.")
        return False

    with open(scored_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If we already generated suggestions, skip to save API calls
    if "missing_citations" in data:
        logger.info("Missing citations already generated. Skipping.")
        return True

    # 1. Gather Context
    meta = data.get("manuscript_metadata", {})
    title = meta.get("title", "Unknown Title")
    abstract = meta.get("abstract", "No abstract available.")
    
    # Get titles of papers the author ALREADY cited (so the LLM doesn't suggest them)
    existing_refs = []
    for ref in data.get("scored_references", []):
        ref_title = ref.get("original_data", {}).get("parsed", {}).get("title")
        if ref_title:
            existing_refs.append(ref_title)
            
    existing_refs_str = "\n".join([f"- {t}" for t in existing_refs[:20]]) # Limit to first 20 to save tokens

    # 2. Setup Gemini Prompt
    genai.configure(api_key=config.GOOGLE_API_KEY)
    
    # We use gemini-2.5-flash as it is fast and excellent at JSON output
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    prompt = f"""
    You are an expert academic peer reviewer. 
    Read the title and abstract of this manuscript:
    
    TITLE: {title}
    ABSTRACT: {abstract}
    
    Here are some papers the author ALREADY cited:
    {existing_refs_str}
    
    TASK:
    Suggest exactly 3 highly relevant, seminal, or state-of-the-art papers that the author SHOULD have cited but missed. 
    Ensure you do not suggest papers already in their list.
    
    Respond STRICTLY in this JSON format, and nothing else:
    [
      {{
        "title": "Full title of the suggested paper",
        "authors": "Main authors",
        "year": "Publication Year",
        "rationale": "Why is this paper crucial for the manuscript?"
      }}
    ]
    """

    # 3. Call the LLM
    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's pure JSON
        response_text = response.text.replace('```json', '').replace('```', '').strip()
        suggestions = json.loads(response_text)
        
        # 4. Save to JSON
        data["missing_citations"] = suggestions
        with open(scored_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info("Successfully added missing citation suggestions.")
        return True

    except Exception as e:
        logger.error(f"Failed to generate missing citations: {e}")
        return False