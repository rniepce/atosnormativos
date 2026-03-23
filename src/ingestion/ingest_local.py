"""
Simplified ingestion script using local embeddings (no GCP required).
Uses sentence-transformers for embeddings and filename-based classification.
"""
import asyncio
import os
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import asyncpg
from openai import AzureOpenAI

from src.ingestion.common import extract_text, chunk_text, get_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client for embeddings
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT', 'https://assistente-web-resource.cognitiveservices.azure.com/')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION', '2025-01-01-preview')
if not AZURE_API_KEY:
    logger.error("AZURE_API_KEY not found in environment")
    import sys; sys.exit(1)

embedding_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)
logger.info("Azure OpenAI embedding client initialized (text-embedding-3-large, 1024-dim)")

# TODO: considerar migrar classify_from_filename para common.py
# (esta versao faz parsing especifico por tipo de ato com logica adicional de parent_dir,
#  diferente de common.build_metadata_from_path)
def classify_from_filename(filename: str, parent_dir: str) -> Dict[str, Any]:
    """
    Extract metadata from filename and directory.
    Pattern examples:
    - re00011961.doc -> Resolução 1/1961
    - port_123_2024.doc -> Portaria 123/2024
    """
    name_lower = filename.lower()
    
    # Default metadata
    metadata = {
        "tipo": "Desconhecido",
        "numero": "0",
        "ano": 0,
        "status": "VIGENTE",
        "assunto_resumo": f"Extraído de {parent_dir}",
        "tags": []
    }
    
    # Pattern: re00011961.doc -> Resolução 1/1961
    match = re.match(r'^(re|res)(\d{4})(\d{4})', name_lower)
    if match:
        metadata["tipo"] = "Resolução"
        metadata["numero"] = str(int(match.group(2)))  # Remove leading zeros
        metadata["ano"] = int(match.group(3))
        return metadata
    
    # Pattern: port123_2024 or similar
    match = re.match(r'^(port|portaria)[\D]*(\d+)[\D]*(\d{4})', name_lower)
    if match:
        metadata["tipo"] = "Portaria"
        metadata["numero"] = match.group(2)
        metadata["ano"] = int(match.group(3))
        return metadata
    
    # Pattern: prov123_2024 or similar
    match = re.match(r'^(prov|provimento)[\D]*(\d+)[\D]*(\d{4})', name_lower)
    if match:
        metadata["tipo"] = "Provimento"
        metadata["numero"] = match.group(2)
        metadata["ano"] = int(match.group(3))
        return metadata
    
    # Use parent directory as type hint
    parent_lower = parent_dir.lower()
    if "resolucao" in parent_lower or "resolução" in parent_lower:
        metadata["tipo"] = "Resolução"
    elif "portaria" in parent_lower:
        if "conjunta" in parent_lower:
            metadata["tipo"] = "Portaria Conjunta"
        else:
            metadata["tipo"] = "Portaria"
    elif "provimento" in parent_lower:
        metadata["tipo"] = "Provimento"
    elif "aviso" in parent_lower:
        metadata["tipo"] = "Aviso"
    elif "instrucao" in parent_lower or "instrução" in parent_lower:
        metadata["tipo"] = "Instrução"
    elif "emenda" in parent_lower:
        metadata["tipo"] = "Emenda Regimental"
    elif "ordem" in parent_lower:
        metadata["tipo"] = "Ordem de Serviço"
    
    # Try to extract any number and year from filename
    numbers = re.findall(r'\d+', filename)
    if len(numbers) >= 2:
        # Assume last 4-digit number is year if between 1900-2100
        for num in reversed(numbers):
            if len(num) == 4 and 1900 <= int(num) <= 2100:
                metadata["ano"] = int(num)
                break
        # First number is likely the document number
        metadata["numero"] = numbers[0].lstrip('0') or '0'
    
    return metadata

async def get_db_connection():
    """Get database connection using environment variables."""
    return await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER'),
        password=os.getenv('POSTGRES_PASSWORD'),
        database=os.getenv('POSTGRES_DB')
    )

async def process_file(file_path: Path, conn, use_llm: bool = False) -> bool:
    """Process a single file: extract text, classify, chunk, embed, store."""
    filename = file_path.name
    logger.info(f"Processing: {filename}")
    
    # 1. Extract Text
    text = extract_text(file_path)
    if not text or len(text.strip()) < 50:
        logger.warning(f"Skipping {filename}: No text extracted or too short.")
        return False
    
    # 2. Classify - use LLM if enabled, otherwise fallback to filename
    metadata = None
    if use_llm:
        try:
            from src.ingestion.classify_llm import classify_with_llm, chunk_by_articles
            metadata = classify_with_llm(text, filename)
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
    
    if metadata is None:
        # Fallback to filename-based classification
        metadata = classify_from_filename(filename, file_path.parent.name)
        logger.info(f"  Classified (filename): {metadata['tipo']} {metadata['numero']}/{metadata['ano']}")
    else:
        logger.info(f"  Classified (LLM): {metadata['tipo']} {metadata['numero']}/{metadata['ano']} - {metadata['status']}")
    
    # 3. Chunk the text - semantic if LLM available, otherwise simple
    if use_llm:
        try:
            from src.ingestion.classify_llm import chunk_by_articles
            chunks = chunk_by_articles(text, metadata)
        except Exception as e:
            logger.warning(f"Semantic chunking failed: {e}")
            chunks = chunk_text(text)
    else:
        chunks = chunk_text(text)
    logger.info(f"  Created {len(chunks)} chunks")
    
    # 4. Generate embeddings via Azure OpenAI
    embeddings = get_embeddings(chunks, embedding_client)
    
    # 5. Store in database
    try:
        async with conn.transaction():
            # Insert document (includes orgao field)
            doc_id = await conn.fetchval("""
                INSERT INTO documentos (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """, 
                filename,
                str(file_path),
                metadata.get("tipo"),
                metadata.get("numero"),
                metadata.get("ano"),
                metadata.get("orgao"),
                metadata.get("status"),
                metadata.get("assunto_resumo"),
                metadata.get("tags", [])
            )
            
            # Insert chunks with embeddings
            chunk_data = [
                (doc_id, chunk, str(list(emb)))
                for chunk, emb in zip(chunks, embeddings)
            ]
            
            await conn.executemany(
                "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                chunk_data
            )
            
        logger.info(f"  ✓ Saved document ID {doc_id} with {len(chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Database error: {e}")
        return False

async def main():
    parser = argparse.ArgumentParser(description="Ingest normative acts from folder (local embeddings)")
    parser.add_argument("--dir", required=True, help="Directory to ingest")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0 for all)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM (Gemini) for classification and semantic chunking")
    parser.add_argument("--clean", action="store_true", help="Clean database before ingestion (DELETE all existing data)")
    args = parser.parse_args()
    
    root_dir = Path(args.dir)
    if not root_dir.exists():
        print(f"Directory not found: {root_dir}")
        return

    # Get database connection
    try:
        conn = await get_db_connection()
        logger.info("Connected to database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return
    
    # Clean database if requested
    if args.clean:
        logger.warning("Cleaning database - deleting all existing data...")
        try:
            await conn.execute("DELETE FROM chunks")
            await conn.execute("DELETE FROM documentos")
            logger.info("Database cleaned successfully")
        except Exception as e:
            logger.error(f"Failed to clean database: {e}")
            await conn.close()
            return
    
    # Find all files
    files_to_process = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".doc", ".docx")) and not file.startswith("~$"):
                files_to_process.append(Path(root) / file)
    
    if args.limit > 0:
        files_to_process = files_to_process[:args.limit]
        
    logger.info(f"Found {len(files_to_process)} files to process")
    if args.use_llm:
        logger.info("LLM classification ENABLED (using Gemini)")
    else:
        logger.info("LLM classification DISABLED (using filename-based)")
    
    # Process files
    success = 0
    failed = 0
    
    for i, file_path in enumerate(files_to_process, 1):
        logger.info(f"[{i}/{len(files_to_process)}] Processing {file_path.name}")
        try:
            if await process_file(file_path, conn, use_llm=args.use_llm):
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            failed += 1
    
    await conn.close()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Ingestion complete!")
    logger.info(f"  Success: {success}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main())
