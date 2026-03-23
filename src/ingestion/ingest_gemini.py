"""
Ingestion script using Gemini Embeddings API (gemini-embedding-001).
Uses google-generativeai SDK with API Key.
"""
import sys
import os
import asyncio
import re
import time
import logging
from pathlib import Path
from typing import Set, Optional, List
from dotenv import load_dotenv

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from openai import AzureOpenAI

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.db import get_db_connection
from src.ingestion.common import extract_text, chunk_text, get_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/ingestion_gemini.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Source directory
SOURCE_DIR = Path(os.getenv("SOURCE_DIR", str(Path(__file__).parent.parent.parent / "data")))

# Initialize Azure OpenAI client for embeddings
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT', 'https://assistente-web-resource.cognitiveservices.azure.com/')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION', '2025-01-01-preview')
if not AZURE_API_KEY:
    logging.error("AZURE_API_KEY not found in environment")
    sys.exit(1)

embedding_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)
logger.info("Azure OpenAI embedding client initialized (text-embedding-3-large, 1024-dim)")


# TODO: considerar migrar extract_metadata_from_path para common.py
# (esta versao possui mapeamento de orgao_patterns nao presente em common.build_metadata_from_path)
def extract_metadata_from_path(file_path: Path) -> dict:
    """Extract metadata from file path."""
    folder_name = file_path.parent.name
    filename = file_path.stem.lower()
    
    metadata = {
        'tipo': folder_name[:199],
        'numero': '0',
        'ano': 0,
        'orgao': None,
        'status': 'VIGENTE',
        'assunto_resumo': folder_name,
        'tags': []
    }
    
    patterns = [
        r'(\d{3,5})(\d{4})$',
        r'(\d{1,4})[_\-](\d{4})',
        r'(\d{1,4})(\d{4})$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            metadata['numero'] = str(int(match.group(1)))
            year = int(match.group(2))
            if 1900 <= year <= 2030:
                metadata['ano'] = year
            break
    
    orgao_patterns = {
        'TJMG': 'Tribunal de Justiça de MG',
        'CGJ': 'Corregedoria Geral de Justiça',
        'Presidência': 'Presidência do TJMG',
        'Vice-Presidência': 'Vice-Presidência do TJMG',
    }
    
    for key, value in orgao_patterns.items():
        if key.lower() in folder_name.lower():
            metadata['orgao'] = value
            break
    
    if not metadata['orgao']:
        metadata['orgao'] = 'TJMG'
    
    return metadata


async def get_existing_files(conn) -> Set[str]:
    """Get set of already ingested filenames."""
    records = await conn.fetch("SELECT filename FROM documentos")
    return {r['filename'] for r in records}


async def main():
    """Main ingestion loop."""
    try:
        conn = await get_db_connection()
        logger.info("Connected to database")
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        return

    existing_files = await get_existing_files(conn)
    logger.info(f"Found {len(existing_files)} already ingested files")

    all_files = []
    for subdir in SOURCE_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix.lower() in ['.doc', '.docx'] and not f.name.startswith('~$'):
                    all_files.append(f)
    
    files_to_process = [f for f in all_files if f.name not in existing_files]
    
    logger.info(f"Total files found: {len(all_files)}")
    logger.info(f"New files to process: {len(files_to_process)}")

    success, failed = 0, 0
    start_time = time.time()
    
    BATCH_SIZE = 5
    current_batch = []

    async def process_batch(batch_docs):
        nonlocal success, failed
        try:
            all_chunks = []
            chunk_map = []
            
            for i, (_, _, _, chunks) in enumerate(batch_docs):
                for j, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    chunk_map.append((i, j))
            
            if not all_chunks:
                return

            embeddings = get_embeddings(all_chunks, embedding_client)
            
            doc_embeddings = [[] for _ in batch_docs]
            for (doc_idx, _), emb in zip(chunk_map, embeddings):
                doc_embeddings[doc_idx].append(emb)
            
            async with conn.transaction():
                for i, (fpath, meta, _, chunks) in enumerate(batch_docs):
                    doc_embs = doc_embeddings[i]
                    
                    doc_id = await conn.fetchval(
                        '''INSERT INTO documentos 
                           (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags) 
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) 
                           RETURNING id''',
                        fpath.name,
                        str(fpath),
                        meta['tipo'],
                        meta['numero'],
                        meta['ano'],
                        meta['orgao'],
                        meta['status'],
                        meta['assunto_resumo'],
                        meta['tags']
                    )
                    
                    chunk_data = [
                        (doc_id, chunk, str(list(emb))) 
                        for chunk, emb in zip(chunks, doc_embs)
                    ]
                    
                    if chunk_data:
                        await conn.executemany(
                            'INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)',
                            chunk_data
                        )
                    success += 1
                    
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            failed += len(batch_docs)

    for i, file_path in enumerate(files_to_process, 1):
        if i % 10 == 0 or i == 1:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(files_to_process) - i) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {i}/{len(files_to_process)} | "
                f"Success: {success} | Failed: {failed} | "
                f"Rate: {rate:.2f} docs/s | ETA: {eta/60:.1f} min"
            )

        text = extract_text(file_path, timeout=20)
        if not text or len(text.strip()) < 50:
            failed += 1
            continue

        metadata = extract_metadata_from_path(file_path)
        chunks = chunk_text(text, chunk_size=1500)

        current_batch.append((file_path, metadata, text, chunks))
        
        if len(current_batch) >= BATCH_SIZE:
            await process_batch(current_batch)
            current_batch = []
            await asyncio.sleep(0.3)  # Rate limit

    if current_batch:
        await process_batch(current_batch)

    await conn.close()
    
    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info("GEMINI INGESTION COMPLETE")
    logger.info(f"Processed: {len(files_to_process)}")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Time: {elapsed/60:.1f} min")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
