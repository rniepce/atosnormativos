import asyncio
import logging
import argparse
from src.ingestion.extraction import extract_text_from_pdf
from src.ingestion.classification import DocumentClassifier
from src.ingestion.chunking import LegalTextSplitter
from src.ingestion.storage import DocumentStorage
from src.utils.vertex import VertexAIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_document(file_path: str, gcs_uri: str = ""):
    logger.info(f"Starting processing for file: {file_path}")
    
    # 1. Extraction
    text = extract_text_from_pdf(file_path)
    if not text:
        logger.error(f"Failed to extract text from {file_path}. Skipping.")
        return

    # 2. Classification
    vertex_client = VertexAIClient()
    classifier = DocumentClassifier(vertex_client)
    logger.info("Classifying document...")
    metadata = classifier.classify_document(text)
    
    if not metadata:
        logger.error("Failed to classify document. Using default metadata.")
        metadata = {
            "tipo": "Desconhecido",
            "numero": "0",
            "ano": 0,
            "status": "Indefinido",
            "assunto_resumo": "Falha na classificação",
            "tags": []
        }
    else:
        logger.info(f"Classification result: {metadata}")

    # 3. Chunking & Vectorization
    splitter = LegalTextSplitter()
    logger.info("Chunking and vectorizing...")
    chunks = splitter.split_text(text, metadata)
    logger.info(f"Generated {len(chunks)} chunks.")

    # 4. Storage
    storage = DocumentStorage()
    logger.info("Saving to database...")
    await storage.save_document_and_chunks(
        filename=file_path.split("/")[-1],
        gcs_uri=gcs_uri,
        metadata=metadata,
        chunks=chunks
    )
    logger.info("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF file.")
    parser.add_argument("file_path", help="Path to the PDF file")
    parser.add_argument("--gcs_uri", help="GCS URI if applicable", default="")
    args = parser.parse_args()

    asyncio.run(process_document(args.file_path, args.gcs_uri))
