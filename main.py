# main.py
import os
import argparse
import yaml
import json
import time
from pathlib import Path
from dotenv import load_dotenv  

# Load environment variables from .env file immediately
load_dotenv()  

from src.core.pdf_parser import PDFParser
from src.core.extractor import ExtractionEngine
# Import all providers
from src.providers.huggingface import HuggingFaceProvider
from src.providers.gemini import GeminiProvider 
from src.providers.openai import OpenAIProvider  # Add OpenAI import
from src.utils.logger import logger

def load_config(path="config/config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Research Paper Extraction System")
    parser.add_argument("--input", "-i", required=True, help="Path to PDF file or directory")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    args = parser.parse_args()

    # 1. Setup
    config = load_config()
    
    # 2. Initialize Provider
    provider_name = config['llm'].get('provider', 'huggingface')
    
    if provider_name == 'huggingface':
        llm_config = config['llm']['huggingface']
        llm_provider = HuggingFaceProvider(llm_config)
        logger.info("Initialized Hugging Face Provider")
        
    elif provider_name == 'gemini':
        llm_config = config['llm']['gemini']
        llm_provider = GeminiProvider(llm_config)
        logger.info("Initialized Google Gemini Provider")
        
    elif provider_name == 'openai':
        llm_config = config['llm']['openai']
        llm_provider = OpenAIProvider(llm_config)
        logger.info("Initialized OpenAI Provider")
        
    else:
        logger.error(f"Unsupported provider in config: {provider_name}")
        return

    # 3. File Discovery
    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("*.pdf"))

    logger.info(f"Found {len(files)} PDF(s) to process.")
    
    # 4. Processing Loop
    os.makedirs(args.output, exist_ok=True)
    
    for pdf_file in files:
        start_time = time.time()
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Parse PDF
            pdf_parser = PDFParser(str(pdf_file))
            pdf_parser.load()
            
            # Extract
            engine = ExtractionEngine(pdf_parser, llm_provider, config)
            result = engine.process()
            
            # Add processing stats
            result['metadata']['processing_time'] = round(time.time() - start_time, 2)
            result['metadata']['llm_provider'] = provider_name
            
            # Save Output
            output_filename = f"{pdf_file.stem}.json"
            output_path = os.path.join(args.output, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Success! Saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}", exc_info=True)

if __name__ == "__main__":
    main()