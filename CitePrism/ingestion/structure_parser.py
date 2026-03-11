import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# CONFIGURATION
# =========================

load_dotenv()

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "ingestion_outputs"
OUTPUT_DIR = PROJECT_ROOT / "ingestion_outputs"


# =========================
# PROMPT (STRICT + SAFE)
# =========================

STRUCTURE_PROMPT = """
You are an academic document STRUCTURE extractor.

CRITICAL RULES (NON-NEGOTIABLE):
- Return ONLY valid JSON.
- DO NOT include explanations, comments, or markdown.
- DO NOT wrap JSON in ``` or ```json blocks.
- The FIRST character of your response must be '{'.
- The LAST character of your response must be '}'.
- DO NOT paraphrase, rewrite, summarize, or clean the text.
- DO NOT remove or alter in-text citations (e.g., [79], [1–3], (Smith et al., 2020)).
- Preserve section text VERBATIM exactly as it appears.

Your task:
- Identify the paper title
- Extract author names
- Extract the abstract
- Extract all main sections
- Preserve original section titles
- Normalize section type ONLY (do not change text)

Return STRICT JSON in the following format:

{
  "title": "...",
  "authors": ["...", "..."],
  "abstract": "...",
  "sections": [
    {
      "section_id": "S1",
      "raw_title": "...",
      "normalized_type": "...",
      "text": "..."
    }
  ]
}
"""


# =========================
# CORE FUNCTION
# =========================

def parse_structure(extracted_text: str) -> dict:
    """
    Sends extracted text to GPT-4.1 and returns structured JSON.
    Uses defensive parsing to handle LLM variability.
    """

    response = CLIENT.responses.create(
        model="gpt-4.1-2025-04-14",
        temperature=0,
        input=[
            {"role": "system", "content": STRUCTURE_PROMPT},
            {"role": "user", "content": extracted_text}
        ]
    )

    raw_output = response.output_text.strip()

    # -------------------------
    # DEFENSIVE JSON EXTRACTION
    # -------------------------
    json_start = raw_output.find("{")
    json_end = raw_output.rfind("}")

    if json_start == -1 or json_end == -1:
        print("----- RAW LLM OUTPUT START -----")
        print(raw_output)
        print("----- RAW LLM OUTPUT END -----")
        raise ValueError("LLM output does not contain JSON")

    json_str = raw_output[json_start:json_end + 1]

    try:
        parsed_json = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("----- RAW LLM OUTPUT START -----")
        print(raw_output)
        print("----- RAW LLM OUTPUT END -----")
        raise ValueError("LLM output is not valid JSON") from e

    return parsed_json


# =========================
# PIPELINE FUNCTION
# =========================

def run_structure_parser(extracted_txt_path: Path):
    """
    Reads extracted text file, parses structure, and saves JSON output.
    """

    if not extracted_txt_path.exists():
        raise FileNotFoundError(f"Extracted text file not found: {extracted_txt_path}")

    print(f"[INFO] Parsing structure for: {extracted_txt_path.name}")

    with open(extracted_txt_path, "r", encoding="utf-8") as f:
        extracted_text = f.read()

    structured_data = parse_structure(extracted_text)

    output_file_name = extracted_txt_path.stem.replace("_extracted", "") + "_structure.json"
    output_path = OUTPUT_DIR / output_file_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Structure JSON saved to: {output_path}")


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    """
    Provide the extracted text file you want to parse.
    """

    EXTRACTED_TEXT_FILE = INPUT_DIR / "paper1_extracted.txt"
    run_structure_parser(EXTRACTED_TEXT_FILE)
