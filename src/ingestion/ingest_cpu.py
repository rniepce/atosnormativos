
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
# Force CPU
torch.set_default_device("cpu")

import sys
import asyncio
import asyncpg
import subprocess
import time
import re
import logging
import signal
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment
load_dotenv(".env")
sys.path.insert(0, ".")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ingestion_cpu.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Timeout handler
def handler(signum, frame):
    raise TimeoutError("Operation timed out")

signal.signal(signal.SIGALRM, handler)

# Load model
logger.info("Loading model on CPU (this takes a moment)...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
logger.info(f"Model loaded! Dim: {model.get_sentence_embedding_dimension()}")

SOURCE_DIR = Path("/Users/rafaelpimentel/Downloads/word")

def extract_text(file_path, timeout=30):
    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(file_path)],
            capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout extracting text from {file_path.name}")
        return None
    except Exception as e:
        logger.warning(f"Error extracting {file_path.name}: {e}")
        return None

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        chunks.append(text[start:start+chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c] or [text[:chunk_size]] if text else []

def extract_metadata(file_path):
    folder = file_path.parent.name
    filename = file_path.stem.lower()
    metadata = {
        "tipo": folder[:199], 
        "numero": "0", 
        "ano": 0, 
        "orgao": "TJMG", 
        "status": "VIGENTE", 
        "assunto_resumo": folder, 
        "tags": []
    }
    match = re.search(r"(\d{3,5})(\d{4})$", filename)
    if match:
        metadata["numero"] = str(int(match.group(1)))
        year = int(match.group(2))
        if 1900 <= year <= 2030:
            metadata["ano"] = year
    return metadata

async def main():
    try:
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT")),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB")
        )
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        return

    # Check existing files
    logger.info("Checking database for existing files...")
    existing = await conn.fetch("SELECT filename FROM documentos")
    existing_set = {r["filename"] for r in existing}
    logger.info(f"Found {len(existing_set)} already ingested files.")

    # Find files to process
    all_files = []
    for subdir in SOURCE_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix.lower() in [".doc", ".docx"] and not f.name.startswith("~$"):
                    all_files.append(f)
    
    files_to_process = [f for f in all_files if f.name not in existing_set]
    logger.info(f"Files to process: {len(files_to_process)}")

    success, failed = 0, 0
    total_chunks = 0
    start_time = time.time()
    
    # Process files
    for i, file_path in enumerate(files_to_process, 1):
        try:
            # Stats every 10 files
            if i % 10 == 0 or i == 1:
                elapsed = time.time() - start_time
                rate = (success + failed) / elapsed if elapsed > 0 else 0
                eta = (len(files_to_process) - i) / rate if rate > 0 else 0
                logger.info(f"[{i}/{len(files_to_process)}] Success: {success} | Failed: {failed} | Rate: {rate:.2f} docs/s | ETA: {eta/3600:.1f}h")

            # 1. Extract
            logger.info(f"Extracting {file_path.name}...")
            text = extract_text(file_path, timeout=20)
            if not text or len(text.strip()) < 50:
                logger.info(f"Skipping {file_path.name} (empty or no text)")
                failed += 1
                continue
            
            # 2. Chunk
            meta = extract_metadata(file_path)
            chunks = chunk_text(text)
            logger.info(f"Chunked {file_path.name}: {len(chunks)} chunks")
            
            if not chunks:
                failed += 1
                continue

            # 3. Embed with Timeout
            embeddings = []
            try:
                # Set timeout for embedding generation (2 mins max per doc)
                logger.info(f"Embedding {file_path.name}...")
                signal.alarm(120) 
                embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
                signal.alarm(0) # Disable alarm
                logger.info(f"Embedded {file_path.name}!")
            except TimeoutError:
                logger.error(f"TIMEOUT embedding {file_path.name}")
                failed += 1
                continue
            except Exception as e:
                logger.error(f"Error embedding {file_path.name}: {e}")
                failed += 1
                continue
            finally:
                signal.alarm(0)

            # 4. Save
            async with conn.transaction():
                doc_id = await conn.fetchval(
                    """INSERT INTO documentos (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags) 
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id""",
                    file_path.name, str(file_path), meta["tipo"], meta["numero"], meta["ano"], meta["orgao"], meta["status"], meta["assunto_resumo"], meta["tags"]
                )
                
                chunk_data = [(doc_id, chunk, str(list(emb))) for chunk, emb in zip(chunks, embeddings)]
                await conn.executemany(
                    "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                    chunk_data
                )
            
            success += 1
            total_chunks += len(chunks)

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            failed += 1

    await conn.close()
    elapsed = time.time() - start_time
    logger.info("="*50)
    logger.info(f"INGESTION COMPLETE")
    logger.info(f"Total Processed: {len(files_to_process)}")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Time: {elapsed/3600:.1f}h")
    logger.info("="*50)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Ingestion stopped by user.")
