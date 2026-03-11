from pathlib import Path
import pdfplumber


# =========================
# CONFIGURATION
# =========================

# Root project directory (CitePrism)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder to store extracted text
OUTPUT_DIR = PROJECT_ROOT / "ingestion_outputs"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# CORE FUNCTION
# =========================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text from a PDF file using pdfplumber
    and PRESERVES page boundaries explicitly.

    Each page is wrapped with clear markers so downstream
    LLMs can reason about pagination safely.
    """

    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()

            if not page_text:
                page_text = ""

            # Explicit page boundary markers
            page_block = (
                f"\n\n<<<PAGE {page_number} START>>>\n"
                f"{page_text}\n"
                f"<<<PAGE {page_number} END>>>\n"
            )

            pages_text.append(page_block)

    # Join pages but DO NOT flatten boundaries
    return "\n".join(pages_text)


# =========================
# PIPELINE FUNCTION
# =========================

def run_text_extraction(pdf_file_path: Path):
    """
    Extracts text from a single PDF and saves output to ingestion_outputs,
    preserving page boundaries.
    """

    if not pdf_file_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file_path}")

    if pdf_file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    print(f"[INFO] Extracting text from: {pdf_file_path.name}")

    text = extract_text_from_pdf(pdf_file_path)

    output_file_name = pdf_file_path.stem + "_extracted.txt"
    output_path = OUTPUT_DIR / output_file_name

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[SUCCESS] Extracted text saved to: {output_path}")


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    """
    Provide the FULL or RELATIVE path of the PDF you want to process.
    """

    PDF_PATH = PROJECT_ROOT / "papers" / "atom0001 - Arts&Culture.pdf"
    run_text_extraction(PDF_PATH)
