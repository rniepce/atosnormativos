"""
CPU-ONLY Safe Ingestion Script v3
===================================
- Forces CPU (no MPS/GPU) to avoid OOM
- Parallel text extraction (6 workers)
- Strips null bytes to avoid PostgreSQL UTF8 errors
- Auto-reconnects to DB if connection drops
- Processes docs one-by-one with micro-batch embedding
- Head+tail sampling for huge docs
- Resume-safe (skips already ingested files)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_default_device("cpu")

import sys
import asyncio
import asyncpg
import subprocess
import time
import re
import logging
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv("/Users/rafaelpimentel/Downloads/atosnormativos/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/ingestion_cpu_safe.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============= CONFIGURATION =============
SOURCE_DIR = Path("/Users/rafaelpimentel/Downloads/word")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
EMBED_BATCH = 16
MAX_CHUNKS_PER_DOC = 50
MAX_EXTRACT_WORKERS = 6
FAILURE_LOG = "/tmp/ingestion_cpu_failures.log"
# ==========================================


async def get_connection():
    """Create a new DB connection."""
    return await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB"),
        timeout=30
    )


async def ensure_connection(conn):
    """Check if connection is alive, reconnect if needed."""
    try:
        if conn and not conn.is_closed():
            await conn.fetchval("SELECT 1")
            return conn
    except Exception:
        pass

    # Reconnect
    try:
        if conn and not conn.is_closed():
            await conn.close()
    except Exception:
        pass

    logger.info("🔄 Reconnecting to database...")
    new_conn = await get_connection()
    logger.info("✅ Reconnected!")
    return new_conn


def sanitize_text(text: str) -> str:
    """Remove null bytes and other problematic characters for PostgreSQL."""
    if not text:
        return text
    text = text.replace('\x00', '')
    text = ''.join(c for c in text if c in '\n\r\t' or (ord(c) >= 32))
    return text


def extract_single_file(file_path_str: str) -> tuple:
    """Extract text from a single file (runs in process pool)."""
    file_path = Path(file_path_str)
    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(file_path)],
            capture_output=True, text=True, check=True, timeout=30
        )
        text = sanitize_text(result.stdout.strip())
        if len(text) < 50:
            return (file_path_str, None, "too_short")
        return (file_path_str, text, None)
    except subprocess.TimeoutExpired:
        return (file_path_str, None, "timeout")
    except Exception as e:
        return (file_path_str, None, str(e)[:50])


def chunk_text(text: str) -> list:
    """Chunk text with head+tail sampling for large docs."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    if len(chunks) > MAX_CHUNKS_PER_DOC:
        half = MAX_CHUNKS_PER_DOC // 2
        chunks = chunks[:half] + chunks[-half:]

    return chunks


def extract_metadata(file_path: Path) -> dict:
    """Extract metadata from filename and folder."""
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

    patterns = [
        r"(\d{3,5})(\d{4})$",
        r"(\d{1,4})[_\-](\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            metadata["numero"] = str(int(match.group(1)))
            year = int(match.group(2))
            if 1900 <= year <= 2030:
                metadata["ano"] = year
            break

    return metadata


async def main():
    start_total = time.time()

    # Load model on CPU
    logger.info("🚀 Loading BGE-large model on CPU...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    logger.info(f"✅ Model loaded on CPU! Dim: {model.get_sentence_embedding_dimension()}")

    # DB Connection
    try:
        conn = await get_connection()
        logger.info("✅ Connected to database")
    except Exception as e:
        logger.error(f"❌ DB connection failed: {e}")
        return

    # Get existing files
    existing = await conn.fetch("SELECT filename FROM documentos")
    existing_set = {r["filename"] for r in existing}
    logger.info(f"📁 Found {len(existing_set)} already ingested files")

    # Collect files to process
    all_files = []
    for subdir in SOURCE_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix.lower() in [".doc", ".docx"] and not f.name.startswith("~$"):
                    if f.name not in existing_set:
                        all_files.append(f)

    logger.info(f"📄 Files to process: {len(all_files)}")

    if not all_files:
        logger.info("✅ Nothing to process!")
        await conn.close()
        return

    open(FAILURE_LOG, 'w').close()

    # ===== PHASE 1: Parallel Text Extraction =====
    logger.info(f"📖 Phase 1: Extracting text with {MAX_EXTRACT_WORKERS} workers...")
    extract_start = time.time()

    extracted_docs = []
    failed_extraction = 0

    with ProcessPoolExecutor(max_workers=MAX_EXTRACT_WORKERS) as executor:
        futures = {executor.submit(extract_single_file, str(f)): f for f in all_files}

        for i, future in enumerate(as_completed(futures), 1):
            file_path_str, text, error = future.result()

            if i % 200 == 0:
                logger.info(f"   Extracted {i}/{len(all_files)}...")

            if text:
                extracted_docs.append((Path(file_path_str), text))
            else:
                failed_extraction += 1

    extract_time = time.time() - extract_start
    logger.info(f"✅ Extraction complete: {len(extracted_docs)} docs in {extract_time:.0f}s ({len(extracted_docs)/max(extract_time,1):.1f} docs/s)")
    logger.info(f"   Extraction failures: {failed_extraction}")

    # ===== PHASE 2: Process one-by-one (chunk + embed + save) =====
    logger.info(f"🧠 Phase 2: Chunking, embedding & saving {len(extracted_docs)} docs...")
    process_start = time.time()

    success, failed, reconnects = 0, 0, 0
    total_chunks_inserted = 0

    for i, (file_path, text) in enumerate(extracted_docs, 1):
        if i % 25 == 0 or i == 1:
            elapsed = time.time() - process_start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = len(extracted_docs) - i
            eta = remaining / rate if rate > 0 else 0
            logger.info(
                f"   [{i}/{len(extracted_docs)}] ✅{success} ❌{failed} 🔄{reconnects} | "
                f"{rate:.2f} docs/s | ETA: {eta/60:.0f} min"
            )

        try:
            # Chunk
            chunks = chunk_text(text)
            if not chunks:
                failed += 1
                continue

            # Sanitize chunks
            chunks = [sanitize_text(c) for c in chunks]
            chunks = [c for c in chunks if c]

            if not chunks:
                failed += 1
                continue

            # Embed in micro-batches
            all_embeddings = []
            for batch_start in range(0, len(chunks), EMBED_BATCH):
                batch = chunks[batch_start:batch_start + EMBED_BATCH]
                embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
                all_embeddings.extend(embs)

            # Metadata
            meta = extract_metadata(file_path)

            # Ensure connection before DB write
            conn = await ensure_connection(conn)

            # Save to DB with retry
            try:
                async with conn.transaction():
                    doc_id = await conn.fetchval(
                        """INSERT INTO documentos
                           (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags)
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id""",
                        file_path.name, str(file_path),
                        meta["tipo"], meta["numero"], meta["ano"],
                        meta["orgao"], meta["status"],
                        meta["assunto_resumo"], meta["tags"]
                    )

                    chunk_data = [
                        (doc_id, chunk, str(list(emb)))
                        for chunk, emb in zip(chunks, all_embeddings)
                    ]
                    await conn.executemany(
                        "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                        chunk_data
                    )
            except (asyncpg.ConnectionDoesNotExistError, 
                    asyncpg.InterfaceError,
                    OSError) as db_err:
                # Reconnect and retry once
                logger.warning(f"🔄 DB error, reconnecting: {db_err}")
                reconnects += 1
                conn = await get_connection()
                
                async with conn.transaction():
                    doc_id = await conn.fetchval(
                        """INSERT INTO documentos
                           (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags)
                           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id""",
                        file_path.name, str(file_path),
                        meta["tipo"], meta["numero"], meta["ano"],
                        meta["orgao"], meta["status"],
                        meta["assunto_resumo"], meta["tags"]
                    )

                    chunk_data = [
                        (doc_id, chunk, str(list(emb)))
                        for chunk, emb in zip(chunks, all_embeddings)
                    ]
                    await conn.executemany(
                        "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                        chunk_data
                    )

            success += 1
            total_chunks_inserted += len(chunks)

            # GC every 100 docs
            if success % 100 == 0:
                gc.collect()

        except Exception as e:
            logger.error(f"❌ Failed {file_path.name}: {e}")
            with open(FAILURE_LOG, 'a') as f:
                f.write(f"{file_path.name}|{str(e)[:100]}\n")
            failed += 1

    try:
        await conn.close()
    except Exception:
        pass

    total_time = time.time() - start_total
    logger.info("=" * 60)
    logger.info("🎉 CPU INGESTION COMPLETE!")
    logger.info(f"   Docs extracted: {len(extracted_docs)}")
    logger.info(f"   Success: {success}")
    logger.info(f"   Failed: {failed + failed_extraction}")
    logger.info(f"   Reconnects: {reconnects}")
    logger.info(f"   Total chunks: {total_chunks_inserted}")
    logger.info(f"   Extraction time: {extract_time:.0f}s")
    logger.info(f"   Process time: {time.time()-process_start:.0f}s")
    logger.info(f"   TOTAL: {total_time/60:.1f} min ({total_time/3600:.1f} h)")
    if total_time > 0:
        logger.info(f"   Rate: {success/(total_time/60):.1f} docs/min")
    logger.info(f"   Failures: {FAILURE_LOG}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⛔ Stopped by user")
