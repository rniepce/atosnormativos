import sys
import os
import asyncio
import re
import time
import logging
from pathlib import Path
from typing import Set

# Load env manually
from dotenv import load_dotenv
load_dotenv()

import asyncpg
from openai import AzureOpenAI

from src.ingestion.common import extract_text, chunk_text, get_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/ingestion_robust.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client for embeddings
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT', 'https://assistente-web-resource.cognitiveservices.azure.com/')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION', '2025-01-01-preview')
if not AZURE_API_KEY:
    logger.error("AZURE_API_KEY not found in environment")
    sys.exit(1)

embedding_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)
logger.info("Azure OpenAI embedding client initialized (text-embedding-3-large, 1024-dim)")

# TODO: considerar migrar classify_from_filename para common.py
# (esta versao faz parsing especifico para Resolucao, diferente de common.build_metadata_from_path)
def classify_from_filename(filename):
    name_lower = filename.lower()
    metadata = {'tipo': 'Resolução', 'numero': '0', 'ano': 0, 'status': 'VIGENTE', 'assunto_resumo': 'Resolução TJMG', 'tags': []}
    match = re.match(r'^(re|res)(\d{4})(\d{4})', name_lower)
    if match:
        metadata['numero'] = str(int(match.group(2)))
        metadata['ano'] = int(match.group(3))
    return metadata

async def get_existing_files(conn) -> Set[str]:
    records = await conn.fetch("SELECT filename FROM documentos")
    return {r['filename'] for r in records}

async def main():
    try:
        conn = await asyncpg.connect(
            host=os.getenv('POSTGRES_HOST'),
            port=int(os.getenv('POSTGRES_PORT')),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            database=os.getenv('POSTGRES_DB')
        )
        logger.info("Connected to database")
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        return

    # 1. Get existing files to skip
    existing_files = await get_existing_files(conn)
    logger.info(f"Checking existing files... Found {len(existing_files)} already ingested.")

    # 2. List files to process
    root_dir = Path(os.getenv("SOURCE_DIR", str(Path(__file__).parent.parent.parent / "data" / "Resolução")))
    all_files = [f for f in root_dir.iterdir() if f.suffix.lower() in ['.doc', '.docx'] and not f.name.startswith('~$')]
    
    files_to_process = [f for f in all_files if f.name not in existing_files]
    logger.info(f"Total files: {len(all_files)}. New to process: {len(files_to_process)}.")

    success, failed = 0, 0
    start_time = time.time()

    for i, file_path in enumerate(files_to_process, 1):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(files_to_process) - i) / rate if rate > 0 else 0
            logger.info(f"Progress: {i}/{len(files_to_process)} | Added: {success} | Failed: {failed} | Rate: {rate:.1f} docs/s | ETA: {eta/60:.1f} min")

        text = extract_text(file_path, timeout=20) # 20s timeout per file
        if not text or len(text.strip()) < 50:
            logger.warning(f"Skipping {file_path.name}: too short or extraction failed")
            failed += 1
            continue
        
        metadata = classify_from_filename(file_path.name)
        chunks = chunk_text(text)
        
        try:
            embeddings = get_embeddings(chunks, embedding_client)
            
            async with conn.transaction():
                doc_id = await conn.fetchval(
                    'INSERT INTO documentos (filename, gcs_uri, tipo, numero, ano, status_vigencia, assunto_resumo, tags) VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id', 
                    file_path.name, str(file_path), metadata['tipo'], metadata['numero'], metadata['ano'], metadata['status'], metadata['assunto_resumo'], []
                )
                chunk_data = [(doc_id, chunk, str(list(emb))) for chunk, emb in zip(chunks, embeddings)]
                await conn.executemany('INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)', chunk_data)
            
            success += 1
            # logger.info(f"Saved {file_path.name}") # Too verbose
        except Exception as e:
            logger.error(f"Error saving {file_path.name}: {e}")
            failed += 1

    await conn.close()
    elapsed = time.time() - start_time
    logger.info("=== INGESTION COMPLETE ===")
    logger.info(f"Processed: {len(files_to_process)}")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Time: {elapsed/60:.1f} min")

if __name__ == "__main__":
    asyncio.run(main())
