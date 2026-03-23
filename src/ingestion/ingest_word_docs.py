import sys
import os
import asyncio
import logging
import subprocess
import time
import re
from pathlib import Path
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

import asyncpg
from openai import AsyncAzureOpenAI

from src.ingestion.common import chunk_text, get_embeddings_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/ingest_word_docs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Azure OpenAI setup ─────────────────────────────────────────
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT', 'https://assistente-web-resource.cognitiveservices.azure.com/')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION', '2025-01-01-preview')
if not AZURE_API_KEY:
    logger.error("AZURE_API_KEY not found in environment")
    sys.exit(1)

client = AsyncAzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

EMBEDDING_DIM = 1024  # Match database schema vector(1024)
MODEL = "text-embedding-3-large"

# ── Database connection ─────────────────────────────────────────
def get_dsn() -> str:
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "railway")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

# ── DB Initialization ───────────────────────────────────────────
async def init_db(conn):
    """Ensure the necessary tables exist in the remote database."""
    logger.info("Initializing remote database architecture...")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS documentos (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            gcs_uri VARCHAR(1024),
            tipo VARCHAR(100),
            numero VARCHAR(50),
            ano INTEGER,
            orgao VARCHAR(100),
            status_vigencia VARCHAR(50),
            assunto_resumo TEXT,
            tags TEXT[],
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)

    await conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            documento_id INTEGER REFERENCES documentos(id) ON DELETE CASCADE,
            conteudo_texto TEXT NOT NULL,
            embedding vector(1024),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create HNSW index for fast approximate nearest neighbor search
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
        ON chunks USING hnsw (embedding vector_cosine_ops);
    """)
    logger.info("Database schema initialized (with HNSW index).")

# ── Embeddings (delegated to common.get_embeddings_async) ─────

# ── Text & Metadata Extraction ──────────────────────────────────
# TODO: considerar migrar extract_word_text para common.py
# (esta versao nao possui timeout e usa errors='ignore', diferente de common.extract_text)
def extract_word_text(filepath: Path) -> str:
    """Extract text from .doc or .docx using macOS textutil."""
    try:
        result = subprocess.run(
            ['textutil', '-convert', 'txt', '-stdout', str(filepath)],
            capture_output=True,
            text=True,
            errors='ignore'
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Failed to extract {filepath}: {e}")
        return ""

def determine_status(text: str) -> str:
    """Check if the document header/ementa mentions revocation.

    Only inspects the first 500 characters to avoid false positives from
    documents that merely reference the revocation of *other* acts.
    """
    header = text[:500].lower()
    if "revogad" in header or "revogam-se" in header:
        return "REVOGADO"
    return "VIGENTE"

def parse_metadata(filepath: Path, base_folder: Path) -> dict:
    """Extract metadata like 'tipo' from the folder structure and filename."""
    # The parent folder of the document relative to the base Word folder
    try:
        rel_path = filepath.relative_to(base_folder)
        folder_category = str(rel_path.parent)
        if folder_category == ".":
            folder_category = "Outros Atos Normativos"
    except Exception:
        folder_category = "Atos Normativos"
        
    filename = filepath.name
    
    # Attempt to extract year from filename or text later if needed
    # For now regex 4 digits starting with 19 or 20
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', filename)
    ano = int(year_match.group(1)) if year_match else 0
    
    return {
        "tipo": folder_category[:100],  # trim to 100 chars
        "numero": "0",  # Default if not easily parsed from title
        "ano": ano,
        "orgao": "TJMG",
        "tags": ["word_batch", "ato_normativo"]
    }

# ── Main ingestion ──────────────────────────────────────────────
async def process_file(conn, filepath: Path, base_folder: Path):
    filename = filepath.name
    
    # Ignore Word temporary files
    if filename.startswith('~$') or filename.startswith('~'):
        return
        
    meta = parse_metadata(filepath, base_folder)

    # Check if already ingested
    existing = await conn.fetchval(
        "SELECT id FROM documentos WHERE filename = $1",
        filename
    )
    if existing:
        logger.debug(f"⏭️  {filename} already ingested (id={existing}), skipping")
        return

    # Extract text
    text = extract_word_text(filepath)
    if not text or len(text) < 100:
        logger.warning(f"⚠️  {filename}: text too short or empty ({len(text)} chars). Skipping.")
        return
        
    # PostgreSQL does not accept null bytes (\x00) in text fields
    text = text.replace('\x00', '')

    # Determine status
    status = determine_status(text)

    # Chunk text
    chunks = chunk_text(text, chunk_size=1500)
    
    # Generate embeddings in batches
    BATCH_SIZE = 50
    all_embeddings = []
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        try:
            embs = await get_embeddings_async(batch, client)
            all_embeddings.extend(embs)
            await asyncio.sleep(0.05)  # Be gentle with rate limits
        except Exception as e:
            logger.error(f"Failed embedding chunk for {filename}: {e}")
            return # skip this file if embedding failed
            
    if len(all_embeddings) != len(chunks):
        logger.error(f"❌ {filename}: Embedding count mismatch. Skipping.")
        return

    # Database insertion
    try:
        async with conn.transaction():
            doc_id = await conn.fetchval(
                """INSERT INTO documentos
                   (filename, gcs_uri, tipo, numero, ano, orgao,
                    status_vigencia, assunto_resumo, tags)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                   RETURNING id""",
                filename,
                str(filepath),
                meta["tipo"],
                meta["numero"],
                meta["ano"],
                meta["orgao"],
                status,
                f"Documento categorizado como {meta['tipo']}",
                meta["tags"]
            )

            chunk_data = [
                (doc_id, chunk, str(emb))
                for chunk, emb in zip(chunks, all_embeddings)
            ]

            await conn.executemany(
                "INSERT INTO chunks (documento_id, conteudo_texto, embedding) "
                "VALUES ($1, $2, $3::vector)",
                chunk_data,
            )

        logger.info(f"✅ {filename} ingested. (Status: {status}, Tipo: {meta['tipo']})")
    except Exception as e:
        logger.error(f"Database insertion failed for {filename}: {e}")

async def main():
    dsn = get_dsn()
    logger.info(f"Connecting to Railway database...")
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return

    logger.info("✅ Connected! Initializing tables if needed...")
    await init_db(conn)

    # Gather all word documents
    base_folder = Path(os.getenv("SOURCE_DIR", str(PROJECT_ROOT / "data")))
    word_docs = []
    for root, _, files in os.walk(base_folder):
        for f in files:
            if f.lower().endswith('.doc') or f.lower().endswith('.docx'):
                word_docs.append(Path(root) / f)

    logger.info(f"Found {len(word_docs)} Word documents to process.")

    # Process files sequentially
    count = 0
    for doc_path in word_docs:
        await process_file(conn, doc_path, base_folder)
        count += 1
        if count % 100 == 0:
            logger.info(f"Progress: {count}/{len(word_docs)} files processed.")

    # Summary
    total_docs = await conn.fetchval("SELECT count(*) FROM documentos")
    total_chunks = await conn.fetchval("SELECT count(*) FROM chunks")
    logger.info("=" * 50)
    logger.info("INGESTION COMPLETE")
    logger.info(f"Database now has {total_docs} documents and {total_chunks} chunks")
    logger.info("=" * 50)

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
