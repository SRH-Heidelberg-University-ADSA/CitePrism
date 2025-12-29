import pymupdf
import json
import re
import os
from huggingface_hub import InferenceClient

class AcademicExtractor:
    def __init__(self, token: str):
        self.client = InferenceClient(api_key=token)
        self.model_options = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]

    def extract_pdf_sections(self, path: str):
        """Splits the PDF into Header text and Reference text."""
        with pymupdf.open(path) as doc:
            header = " ".join([page.get_text() for page in doc[:3]])
            
            ref_text = ""
            for i in range(len(doc) - 1, max(-1, len(doc) - 8), -1):
                page_text = doc[i].get_text()
                if re.search(r'\b(References|BIBLIOGRAPHY)\b', page_text, re.IGNORECASE):
                    ref_text = " ".join([page.get_text() for page in doc[i:]])
                    break
            
            if not ref_text:
                ref_text = " ".join([page.get_text() for page in doc[-2:]])

            return {
                "header": re.sub(r'\s+', ' ', header)[:5000],
                "refs": re.sub(r'\s+', ' ', ref_text)[:6000]
            }

    def ask_ai(self, prompt: str):
        """Helper to try multiple models for a specific prompt."""
        for model_id in self.model_options:
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "Return ONLY raw JSON. No markdown blocks."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500 # High enough for a single task
                )
                content = response.choices[0].message.content
                clean = re.sub(r'```json\s*|```', '', content).strip()
                return json.loads(clean)
            except Exception as e:
                print(f"     ⚠ {model_id} failed: {str(e)[:50]}")
                continue
        return None

    def process_paper(self, path: str):
        sections = self.extract_pdf_sections(path)
        
        # Call 1: Metadata
        print("    - Extracting Metadata...")
        meta_prompt = f"Extract title, authors (list), abstract, and doi from this text into JSON:\n\n{sections['header']}"
        metadata = self.ask_ai(meta_prompt) or {}

        # Call 2: References
        print("    - Extracting References...")
        ref_prompt = f"Extract a list of full bibliographic strings from this text into a JSON object with one key 'references' (a list of strings):\n\n{sections['refs']}"
        refs_data = self.ask_ai(ref_prompt) or {"references": []}

        # Combine
        metadata["references"] = refs_data.get("references", [])
        return metadata

# --- EXECUTION ---
HF_TOKEN = "" # Add your HuggingFace token here
SRC_FOLDER = "./src/"
OUTPUT_FILE = "extracted_papers.json"

extractor = AcademicExtractor(HF_TOKEN)
all_results = {}

if os.path.exists(SRC_FOLDER):
    pdf_files = [f for f in os.listdir(SRC_FOLDER) if f.lower().endswith('.pdf')]
    for filename in sorted(pdf_files):
        print(f"\n[Processing] {filename}")
        all_results[filename] = extractor.process_paper(os.path.join(SRC_FOLDER, filename))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Finished. Check {OUTPUT_FILE}")