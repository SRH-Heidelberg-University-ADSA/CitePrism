import json
import re
import unicodedata
from pathlib import Path
from openai import OpenAI

# =========================
# CONFIGURATION
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "ingestion_outputs"
OUTPUT_DIR = PROJECT_ROOT / "ingestion_outputs"

CLIENT = OpenAI()
LLM_MODEL = "gpt-4.1"

# =========================
# REGEX PATTERNS (PRIMARY)
# =========================

AUTHOR_YEAR_PATTERN = re.compile(
    r"\(([^()]*?\d{4}[a-z]?(?:\s*;\s*[^()]*?\d{4}[a-z]?)*?)\)"
)

NUMERIC_PATTERN = re.compile(
    r"\[\s*\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*\s*\]"
)

# =========================
# LLM PROMPTS (FALLBACK)
# =========================

SYSTEM_PROMPT = """
You are an academic citation detector.

Task:
- Identify ONLY explicit in-text citation markers.
- Examples: [79], [1–3], (Smith et al., 2020), (Doe, 2019; Lee, 2021)
- Do NOT infer references.
- Do NOT paraphrase text.
- Return exact character offsets.
- If none exist, return an empty list.

Return STRICT JSON only.
JSON schema:
{
  "citations": [
    {
      "citation_text": "...",
      "start_char": 0,
      "end_char": 0
    }
  ]
}
"""

USER_PROMPT_TEMPLATE = """
TEXT:
\"\"\"{text}\"\"\"

Return JSON only.
"""

# =========================
# UTILITY FUNCTIONS
# =========================

def normalize_text(text: str) -> str:
    """
    Normalize Unicode artifacts from PDFs.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\u00a0", " ")
    return text


# =========================
# REGEX-BASED DETECTION
# =========================

def detect_citations_regex(text: str) -> list:
    """
    Detect citations using deterministic regex.
    """
    citations = []

    for match in AUTHOR_YEAR_PATTERN.finditer(text):
        citations.append({
            "citation_text": match.group(0),
            "start_char": match.start(),
            "end_char": match.end()
        })

    for match in NUMERIC_PATTERN.finditer(text):
        citations.append({
            "citation_text": match.group(0),
            "start_char": match.start(),
            "end_char": match.end()
        })

    return citations


# =========================
# LLM FALLBACK DETECTION
# =========================

def detect_citations_llm(text: str) -> list:
    """
    LLM-based citation detection (fallback only).
    """
    response = CLIENT.responses.create(
        model=LLM_MODEL,
        temperature=0,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(text=text)
            }
        ],
    )

    output_text = response.output_text.strip()

    try:
        parsed = json.loads(output_text)
        return parsed.get("citations", [])
    except json.JSONDecodeError:
        print("[WARNING] LLM returned invalid JSON. Skipping.")
        return []


# =========================
# PIPELINE FUNCTION
# =========================

def run_citation_detector(structure_json_path: Path):

    if not structure_json_path.exists():
        raise FileNotFoundError(f"Structure file not found: {structure_json_path}")

    print(f"[INFO] Detecting citations (hybrid) in: {structure_json_path.name}")

    with open(structure_json_path, "r", encoding="utf-8") as f:
        structure = json.load(f)

    all_citations = []
    citation_counter = 1

    for section in structure.get("sections", []):
        raw_text = section.get("text", "")
        if not raw_text:
            continue

        text = normalize_text(raw_text)

        # --- PRIMARY: REGEX ---
        citations = detect_citations_regex(text)

        # --- FALLBACK: LLM ---
        if len(citations) == 0:
            citations = detect_citations_llm(text)

        for c in citations:
            all_citations.append({
                "citation_id": f"C{citation_counter}",
                "citation_text": c["citation_text"],
                "section_id": section["section_id"],
                "section_type": section["normalized_type"],
                "start_char": c["start_char"],
                "end_char": c["end_char"]
            })
            citation_counter += 1

    output_path = OUTPUT_DIR / (
        structure_json_path.stem.replace("_structure", "_citations") + ".json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"citations": all_citations}, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Citations saved to: {output_path}")
    print(f"[INFO] Total citations detected: {len(all_citations)}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    STRUCTURE_FILE = INPUT_DIR / "paper1_structure.json"
    run_citation_detector(STRUCTURE_FILE)
