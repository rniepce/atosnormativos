"""
Ingestion script using BGE-M3 local embeddings.
Model: BAAI/bge-m3 (1024 dimensions)
SOTA multilingual model, optimized for retrieval.
"""
import sys
import os
import asyncio
import subprocess
import re
import time
import logging
from pathlib import Path
from typing import Set, Optional, List
from dotenv import load_dotenv

# Load environment
load_dotenv('/Users/rafaelpimentel/Downloads/atosnormativos/.env')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.db import get_db_connection

from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/ingestion_bge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Failure log
FAILURE_LOG = "/tmp/ingestion_failures.log"

# Source directory
SOURCE_DIR = Path('/Users/rafaelpimentel/Downloads/word')

# Load BGE-large model (1024-dim, compatible with older PyTorch)
print("Loading BGE-large model (this may take a minute)...", flush=True)
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
print(f"✅ Model loaded! Embedding dim: {model.get_sentence_embedding_dimension()}", flush=True)


def log_failure(filename: str, reason: str):
    """Log failed files to separate file."""
    with open(FAILURE_LOG, 'a') as f:
        f.write(f"{filename}|{reason}\n")


def extract_text(file_path: Path, timeout: int = 30) -> Optional[str]:
    """Extract text from .doc/.docx using macOS textutil."""
    try:
        result = subprocess.run(
            ['textutil', '-convert', 'txt', '-stdout', str(file_path)],
            capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        log_failure(file_path.name, "timeout")
        return None
    except Exception as e:
        log_failure(file_path.name, str(e)[:100])
        return None


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


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Text chunking with overlap."""
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        chunks.append(text[start:start+chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c] or [text[:chunk_size]] if text else []


async def get_existing_files(conn) -> Set[str]:
    """Get set of already ingested filenames."""
    records = await conn.fetch("SELECT filename FROM documentos")
    return {r['filename'] for r in records}


async def main():
    """Main ingestion loop."""
    # Clear failure log
    open(FAILURE_LOG, 'w').close()
    
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
    
    BATCH_SIZE = 10
    current_batch = []

    async def process_batch(batch_docs):
        nonlocal success, failed
        try:
            all_chunks = []
            chunk_map = []
            
            for i, (fpath, meta, txt, chunks) in enumerate(batch_docs):
                for j, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    chunk_map.append((i, j))
            
            if not all_chunks:
                return

            # Generate embeddings locally
            embeddings = model.encode(all_chunks, show_progress_bar=False, normalize_embeddings=True)
            
            doc_embeddings = [[] for _ in batch_docs]
            for (doc_idx, _), emb in zip(chunk_map, embeddings):
                doc_embeddings[doc_idx].append(emb)
            
            async with conn.transaction():
                for i, (fpath, meta, txt, chunks) in enumerate(batch_docs):
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
            for fpath, _, _, _ in batch_docs:
                log_failure(fpath.name, str(e)[:100])
            failed += len(batch_docs)

    for i, file_path in enumerate(files_to_process, 1):
        if i % 50 == 0 or i == 1:
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
        chunks = chunk_text(text)
        
        current_batch.append((file_path, metadata, text, chunks))
        
        if len(current_batch) >= BATCH_SIZE:
            await process_batch(current_batch)
            current_batch = []

    if current_batch:
        await process_batch(current_batch)

    await conn.close()
    
    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info("BGE-M3 INGESTION COMPLETE")
    logger.info(f"Processed: {len(files_to_process)}")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Time: {elapsed/60:.1f} min")
    logger.info(f"Failures logged to: {FAILURE_LOG}")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
