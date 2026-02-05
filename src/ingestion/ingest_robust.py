import sys
import os
import asyncio
import subprocess
import re
import time
import logging
from pathlib import Path
from typing import Set

# Load env manually
from dotenv import load_dotenv
load_dotenv('/Users/rafaelpimentel/Downloads/atosnormativos/.env')

os.environ['POSTGRES_HOST'] = 'gondola.proxy.rlwy.net'
os.environ['POSTGRES_PORT'] = '45477'
os.environ['POSTGRES_USER'] = 'postgres'
os.environ['POSTGRES_PASSWORD'] = 'pgvector123secure'
os.environ['POSTGRES_DB'] = 'tjmg_rag'

from sentence_transformers import SentenceTransformer
import asyncpg

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

print("Loading embedding model...", flush=True)
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!", flush=True)

def extract_text(file_path, timeout=30):
    try:
        result = subprocess.run(
            ['textutil', '-convert', 'txt', '-stdout', str(file_path)],
            capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout extracting {file_path}")
        return None
    except Exception as e:
        logger.warning(f"Error extracting {file_path}: {e}")
        return None

def classify_from_filename(filename):
    name_lower = filename.lower()
    metadata = {'tipo': 'Resolução', 'numero': '0', 'ano': 0, 'status': 'VIGENTE', 'assunto_resumo': 'Resolução TJMG', 'tags': []}
    match = re.match(r'^(re|res)(\d{4})(\d{4})', name_lower)
    if match:
        metadata['numero'] = str(int(match.group(2)))
        metadata['ano'] = int(match.group(3))
    return metadata

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        chunks.append(text[start:start+chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c] or [text[:chunk_size]] if text else []

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
    root_dir = Path('/Users/rafaelpimentel/Downloads/word/Resolução')
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
            embeddings = model.encode(chunks, show_progress_bar=False)
            
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
