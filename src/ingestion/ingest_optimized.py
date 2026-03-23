"""
Optimized ingestion script for large document sets.
- Uses BGE-large-en-v1.5 (1024 dimensions) 
- Processes chunks in small batches to avoid OOM
- Limits max chunks per document (takes first N + last N)
- Skips very large documents that would timeout
"""
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import sys
import asyncio
import asyncpg
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

from src.ingestion.common import extract_text, chunk_text, build_metadata_from_path, get_embeddings

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/tmp/ingestion_optimized.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
SOURCE_DIR = Path(os.getenv("SOURCE_DIR", str(Path(__file__).parent.parent.parent / "data")))
MAX_CHUNKS_PER_DOC = 100  # Limit chunks per document
EMBED_BATCH_SIZE = 16     # Small batches to prevent OOM
MAX_DOC_LENGTH = 500000   # Skip docs larger than 500KB of text
# ===================================

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


def smart_sample_chunks(chunks, max_chunks=MAX_CHUNKS_PER_DOC):
    """Keep first half and last half of max_chunks to preserve context."""
    if len(chunks) <= max_chunks:
        return chunks
    
    half = max_chunks // 2
    return chunks[:half] + chunks[-half:]


def embed_in_batches(chunks):
    """Embed chunks in small batches via Azure OpenAI."""
    all_embeddings = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i+EMBED_BATCH_SIZE]
        embs = get_embeddings(batch, embedding_client)
        all_embeddings.extend(embs)
    return all_embeddings


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
    logger.info("Checking for existing files...")
    existing = await conn.fetch("SELECT filename FROM documentos")
    existing_set = {r["filename"] for r in existing}
    logger.info(f"Found {len(existing_set)} already ingested files.")

    # Find files
    all_files = []
    for subdir in SOURCE_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix.lower() in [".doc", ".docx"] and not f.name.startswith("~$"):
                    all_files.append(f)
    
    files_to_process = [f for f in all_files if f.name not in existing_set]
    logger.info(f"Files to process: {len(files_to_process)}")

    success, failed, skipped = 0, 0, 0
    start_time = time.time()
    
    for i, file_path in enumerate(files_to_process, 1):
        try:
            # Progress every 20 files
            if i % 20 == 0 or i == 1:
                elapsed = time.time() - start_time
                rate = success / elapsed if elapsed > 0 else 0
                remaining = len(files_to_process) - i
                eta = remaining / rate if rate > 0 else 0
                logger.info(
                    f"[{i}/{len(files_to_process)}] ✓{success} ✗{failed} ⊘{skipped} | "
                    f"{rate:.2f} docs/s | ETA: {eta/3600:.1f}h"
                )

            # 1. Extract
            text = extract_text(file_path, timeout=20)
            if not text or len(text.strip()) < 50:
                failed += 1
                continue
            
            # Skip very large docs
            if len(text) > MAX_DOC_LENGTH:
                logger.info(f"Skipping {file_path.name}: too large ({len(text)/1000:.0f}KB)")
                skipped += 1
                continue

            # 2. Chunk with smart sampling
            meta = build_metadata_from_path(file_path)
            all_chunks = chunk_text(text)
            chunks = smart_sample_chunks(all_chunks)
            
            if len(all_chunks) > MAX_CHUNKS_PER_DOC:
                logger.debug(f"Sampled {file_path.name}: {len(all_chunks)} -> {len(chunks)} chunks")

            if not chunks:
                failed += 1
                continue

            # 3. Embed in batches
            embeddings = embed_in_batches(chunks)

            # 4. Save
            async with conn.transaction():
                doc_id = await conn.fetchval(
                    """INSERT INTO documentos (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags) 
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id""",
                    file_path.name, str(file_path), meta["tipo"], meta["numero"], meta["ano"], 
                    meta["orgao"], meta["status"], meta["assunto_resumo"], meta["tags"]
                )
                
                chunk_data = [(doc_id, chunk, str(list(emb))) for chunk, emb in zip(chunks, embeddings)]
                await conn.executemany(
                    "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                    chunk_data
                )
            
            success += 1

        except Exception as e:
            logger.error(f"Error {file_path.name}: {e}")
            failed += 1

    await conn.close()
    elapsed = time.time() - start_time
    logger.info("="*50)
    logger.info("INGESTION COMPLETE")
    logger.info(f"Processed: {len(files_to_process)}")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (too large): {skipped}")
    logger.info(f"Time: {elapsed/3600:.1f}h")
    logger.info("="*50)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Ingestion stopped by user.")
