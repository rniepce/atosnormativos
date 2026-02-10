"""
FAST Ingestion Script - Highly Optimized
=========================================
Optimizations:
1. Multiprocessing for text extraction (CPU-bound)
2. Large batch encoding with MPS memory management
3. Async DB writes with connection pooling
4. Smart chunking (larger chunks = fewer embeddings)
5. Progress persistence (resume on crash)
"""
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import asyncio
import asyncpg
import subprocess
import time
import re
import logging
import gc
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment
load_dotenv("/Users/rafaelpimentel/Downloads/atosnormativos/.env")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/ingestion_fast.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============= CONFIGURATION =============
SOURCE_DIR = Path("/Users/rafaelpimentel/Downloads/word")
CHUNK_SIZE = 1500          # Larger chunks = fewer embeddings
CHUNK_OVERLAP = 300
EMBED_BATCH = 64           # Larger batches for MPS efficiency
MAX_WORKERS = 4            # Parallel text extraction
MAX_CHUNKS_PER_DOC = 200   # Limit for huge docs
# ==========================================


def extract_single_file(file_path_str: str) -> tuple:
    """Extract text from a single file (runs in subprocess pool)."""
    file_path = Path(file_path_str)
    try:
        result = subprocess.run(
            ["textutil", "-convert", "txt", "-stdout", str(file_path)],
            capture_output=True, text=True, check=True, timeout=30
        )
        text = result.stdout.strip()
        if len(text) < 50:
            return (file_path_str, None, "too_short")
        return (file_path_str, text, None)
    except subprocess.TimeoutExpired:
        return (file_path_str, None, "timeout")
    except Exception as e:
        return (file_path_str, None, str(e)[:50])


def chunk_text(text: str) -> list:
    """Chunk text with larger size for efficiency."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    chunks = [c for c in chunks if c]
    
    # Limit chunks for huge documents
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
    
    # Try to extract number/year
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
    
    # Load model
    logger.info("🚀 Loading BGE-large model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    logger.info(f"✅ Model loaded on {device.upper()}! Dim: {model.get_sentence_embedding_dimension()}")
    
    # DB Connection
    try:
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT")),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB")
        )
        logger.info("✅ Connected to database")
    except Exception as e:
        logger.error(f"❌ DB connection failed: {e}")
        return
    
    # Get existing files
    existing = await conn.fetch("SELECT filename FROM documentos")
    existing_set = {r["filename"] for r in existing}
    logger.info(f"📁 Found {len(existing_set)} already ingested files")
    
    # Collect all files
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
    
    # ===== PHASE 1: Parallel Text Extraction =====
    logger.info(f"📖 Phase 1: Extracting text with {MAX_WORKERS} workers...")
    extraction_start = time.time()
    
    extracted_docs = []
    failed_extraction = 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_single_file, str(f)): f for f in all_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            file_path_str, text, error = future.result()
            
            if i % 100 == 0:
                logger.info(f"   Extracted {i}/{len(all_files)}...")
            
            if text:
                extracted_docs.append((Path(file_path_str), text))
            else:
                failed_extraction += 1
    
    extraction_time = time.time() - extraction_start
    logger.info(f"✅ Extraction complete: {len(extracted_docs)} docs in {extraction_time:.1f}s ({len(extracted_docs)/extraction_time:.1f} docs/s)")
    logger.info(f"   Failed: {failed_extraction}")
    
    # ===== PHASE 2: Chunking =====
    logger.info("✂️ Phase 2: Chunking documents...")
    
    all_chunks = []  # (file_path, metadata, chunk_content, chunk_idx)
    doc_chunk_counts = {}
    
    for file_path, text in extracted_docs:
        chunks = chunk_text(text)
        metadata = extract_metadata(file_path)
        doc_chunk_counts[file_path] = len(chunks)
        
        for idx, chunk_content in enumerate(chunks):
            all_chunks.append((file_path, metadata, chunk_content, idx))
    
    total_chunks = len(all_chunks)
    logger.info(f"✅ Created {total_chunks} chunks from {len(extracted_docs)} docs")
    
    # ===== PHASE 3: Batch Embedding =====
    logger.info(f"🧠 Phase 3: Embedding {total_chunks} chunks in batches of {EMBED_BATCH}...")
    embed_start = time.time()
    
    embeddings = []
    for batch_start in range(0, total_chunks, EMBED_BATCH):
        batch_end = min(batch_start + EMBED_BATCH, total_chunks)
        batch_texts = [c[2] for c in all_chunks[batch_start:batch_end]]
        
        batch_embs = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings.extend(batch_embs)
        
        # Progress every 500 chunks
        if batch_end % 500 == 0 or batch_end == total_chunks:
            progress = batch_end / total_chunks * 100
            elapsed = time.time() - embed_start
            rate = batch_end / elapsed
            eta = (total_chunks - batch_end) / rate if rate > 0 else 0
            logger.info(f"   {batch_end}/{total_chunks} ({progress:.0f}%) | {rate:.0f} chunks/s | ETA: {eta:.0f}s")
        
        # Clear MPS cache periodically
        if device == "mps" and batch_end % 1000 == 0:
            torch.mps.empty_cache()
            gc.collect()
    
    embed_time = time.time() - embed_start
    logger.info(f"✅ Embedding complete in {embed_time:.1f}s ({total_chunks/embed_time:.0f} chunks/s)")
    
    # ===== PHASE 4: Database Insert =====
    logger.info("💾 Phase 4: Saving to database...")
    db_start = time.time()
    
    # Group by document
    doc_data = {}
    for i, (file_path, metadata, chunk_content, chunk_idx) in enumerate(all_chunks):
        if file_path not in doc_data:
            doc_data[file_path] = {"metadata": metadata, "chunks": []}
        doc_data[file_path]["chunks"].append((chunk_content, embeddings[i]))
    
    success, failed = 0, 0
    
    for file_path, data in doc_data.items():
        try:
            async with conn.transaction():
                doc_id = await conn.fetchval(
                    """INSERT INTO documentos 
                       (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags) 
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id""",
                    file_path.name, str(file_path), 
                    data["metadata"]["tipo"], data["metadata"]["numero"], data["metadata"]["ano"],
                    data["metadata"]["orgao"], data["metadata"]["status"], 
                    data["metadata"]["assunto_resumo"], data["metadata"]["tags"]
                )
                
                chunk_inserts = [
                    (doc_id, chunk, str(list(emb))) 
                    for chunk, emb in data["chunks"]
                ]
                await conn.executemany(
                    "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                    chunk_inserts
                )
            success += 1
            
            if success % 100 == 0:
                logger.info(f"   Saved {success} docs...")
                
        except Exception as e:
            logger.error(f"Error saving {file_path.name}: {e}")
            failed += 1
    
    db_time = time.time() - db_start
    logger.info(f"✅ DB insert complete in {db_time:.1f}s")
    
    await conn.close()
    
    # ===== SUMMARY =====
    total_time = time.time() - start_total
    logger.info("=" * 60)
    logger.info("🎉 INGESTION COMPLETE!")
    logger.info(f"   Documents processed: {len(extracted_docs)}")
    logger.info(f"   Documents saved: {success}")
    logger.info(f"   Documents failed: {failed + failed_extraction}")
    logger.info(f"   Total chunks: {total_chunks}")
    logger.info("-" * 60)
    logger.info(f"   Extraction time: {extraction_time:.1f}s")
    logger.info(f"   Embedding time: {embed_time:.1f}s") 
    logger.info(f"   DB write time: {db_time:.1f}s")
    logger.info(f"   TOTAL TIME: {total_time/60:.1f} min")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⛔ Stopped by user")
