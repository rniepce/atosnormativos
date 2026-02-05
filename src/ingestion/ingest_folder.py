import asyncio
import os
import argparse
import logging
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.ingestion.extraction import extract_text_from_pdf, extract_text_from_doc_docx
from src.ingestion.classification import DocumentClassifier
from src.ingestion.chunking import LegalTextSplitter
from src.ingestion.storage import DocumentStorage
from src.utils.vertex import VertexAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pattern to extract basic metadata from filename
# Ex: re00011961.doc -> Tipo: re (Resolucao), Num: 0001, Ano: 1961
# Ex: port_conj_123_2024.pdf -> Tipo: portaria conjunta, Num: 123, Ano: 2024
FILENAME_PATTERNS = [
    (r"^(re)(\d{4})(\d{4})", "Resolução"), # re00011961
    (r"^(port)(\d+)_(\d{4})", "Portaria"), 
    (r"^(port_conj|portaria_conjunta)(\d+)_(\d{4})", "Portaria Conjunta"),
    (r"^(prov)(\d+)_(\d{4})", "Provimento"),
]

def infer_metadata_from_filename(filename: str, parent_dir: str) -> dict:
    """
    Tries to infer basic metadata (Tipo, Numero, Ano) to help the classifier.
    Also uses the parent directory name as a hint for 'Tipo'.
    """
    name_lower = filename.lower()
    
    metadata = {}
    
    # 1. Try regex patterns
    for pattern, tipo in FILENAME_PATTERNS:
        match = re.search(pattern, name_lower)
        if match:
            # Usually groups: 1=prefix, 2=num, 3=ano
            try:
                metadata["tipo"] = tipo
                metadata["numero"] = str(int(match.group(2))) # Remove leading zeros
                metadata["ano"] = int(match.group(3))
                break
            except Exception:
                pass
    
    # 2. Use parent directory as fallback for Tipo
    if "tipo" not in metadata and parent_dir:
        parent_lower = parent_dir.lower()
        if "resolucao" in parent_lower or "resolução" in parent_lower:
            metadata["tipo"] = "Resolução"
        elif "portaria" in parent_lower:
             metadata["tipo"] = "Portaria"
        elif "provimento" in parent_lower:
             metadata["tipo"] = "Provimento"
        elif "aviso" in parent_lower:
             metadata["tipo"] = "Aviso"
        elif "instrucao" in parent_lower or "instrução" in parent_lower:
             metadata["tipo"] = "Instrução"
    
    return metadata

async def process_file(
    file_path: Path, 
    classifier: DocumentClassifier, 
    splitter: LegalTextSplitter, 
    storage: DocumentStorage
):
    filename = file_path.name
    logger.info(f"Processing: {filename}")
    
    # 1. Extract Text
    text = None
    if filename.lower().endswith(".pdf"):
        text = extract_text_from_pdf(str(file_path))
    elif filename.lower().endswith((".doc", ".docx")):
        text = extract_text_from_doc_docx(str(file_path))
    
    if not text:
        logger.warning(f"Skipping {filename}: Text extraction failed or unsupported format.")
        return
    
    # 2. Classify (with hints)
    # We infer hints from filename to improve Gemini's accuracy or serve as fallback
    hints = infer_metadata_from_filename(filename, file_path.parent.name)
    
    # We can pass these hints to the classifier if modified, or just merge after.
    # For now, let's let the classifier do its work, but if it fails to find structured info, we use hints.
    # Actually, the classifier `classify_document` takes text. 
    # Let's trust the classifier mainly, but maybe prepend the filename info to the text context?
    
    context_text = f"Nome do arquivo: {filename}\nDiretório: {file_path.parent.name}\n\n{text[:20000]}" # Truncate for classification to save tokens if huge
    
    try:
        # Note: classifier.classify_document might need the full text for better context, 
        # but let's pass the first chunk + file info to save time/cost.
        # Ideally, we pass the whole text if it fits context window. Gemini 2.0 Flash has big window.
        metadata = classifier.classify_document(text) # Pass full text
    except Exception as e:
        logger.error(f"Classification error for {filename}: {e}")
        metadata = None

    if not metadata:
        metadata = {
            "tipo": hints.get("tipo", "Desconhecido"),
            "numero": hints.get("numero", "0"),
            "ano": hints.get("ano", 0),
            "status": "Indefinido",
            "assunto_resumo": "Falha na classificação automática",
            "tags": []
        }
    else:
        # Merge hints if classifier missed something (optional, trusting classifier for now)
        pass

    logger.info(f"Classified {filename}: {metadata.get('tipo')} {metadata.get('numero')}/{metadata.get('ano')}")

    # 3. Chunk
    chunks = splitter.split_text(text, metadata)
    
    # 4. Store
    await storage.save_document_and_chunks(
        filename=filename,
        gcs_uri=str(file_path), # Storing local path as URI for now
        metadata=metadata,
        chunks=chunks
    )
    logger.info(f"Saved {filename} with {len(chunks)} chunks.")


async def main():
    parser = argparse.ArgumentParser(description="Ingest normative acts from folder")
    parser.add_argument("--dir", required=True, help="Directory to ingest")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files processing (0 for all)")
    args = parser.parse_args()
    
    root_dir = Path(args.dir)
    if not root_dir.exists():
        print(f"Directory not found: {root_dir}")
        return

    # Initialize services
    vertex_client = VertexAIClient() # Ensure env vars are set for project/location if needed
    classifier = DocumentClassifier(vertex_client)
    splitter = LegalTextSplitter()
    storage = DocumentStorage()
    
    # Connect DB pool? DocumentStorage usually handles its own connection or needs one?
    # Checking existing DocumentStorage... it likely creates connection on save, or we passed init?
    # Looking at `src/backend/main.py`: storage = DocumentStorage(); await storage.save...
    # So it seems stateless or connection created in method.
    
    files_to_process = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".pdf", ".doc", ".docx")) and not file.startswith("~$"):
                files_to_process.append(Path(root) / file)
    
    if args.limit > 0:
        files_to_process = files_to_process[:args.limit]
        
    print(f"Found {len(files_to_process)} files to process.")
    
    for file_path in files_to_process:
        try:
            await process_file(file_path, classifier, splitter, storage)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
