"""
Ingest Lei Complementar 59/2001 + Regimento Interno into the vector database.
Uses Gemini Embedding API (gemini-embedding-001) with 1024 dimensions
to match the database schema (vector(1024)).

Usage:
    python3 -m src.ingestion.ingest_lc59_ri
"""
import sys
import os
import asyncio
import logging
import time
from pathlib import Path
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

import asyncpg
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/ingest_lc59_ri.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Gemini setup ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found in environment")
    print("Set it with: export GEMINI_API_KEY='your-key-here'")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

EMBEDDING_DIM = 1024  # Match database schema vector(1024)

# Test embedding
print("Testing Gemini Embedding API...", flush=True)
try:
    test_result = genai.embed_content(
        model="models/gemini-embedding-001",
        content="test",
        task_type="retrieval_document",
        output_dimensionality=EMBEDDING_DIM
    )
    actual_dim = len(test_result['embedding'])
    print(f"✅ Gemini ready! Embedding dim: {actual_dim}", flush=True)
    assert actual_dim == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}, got {actual_dim}"
except Exception as e:
    print(f"❌ Failed to test Gemini: {e}", flush=True)
    sys.exit(1)


# ── Database connection ─────────────────────────────────────────
def get_dsn() -> str:
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "password")
    db = os.getenv("POSTGRES_DB", "tjmg_rag")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


# ── Embeddings ──────────────────────────────────────────────────
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from Gemini API with 1024 dimensions."""
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texts,
        task_type="retrieval_document",
        output_dimensionality=EMBEDDING_DIM
    )
    return result['embedding']


# ── Text chunking ───────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        chunks.append(text[start:start + chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


# ── PDF text extraction ─────────────────────────────────────────
def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using pymupdf (fitz)."""
    import fitz  # pymupdf
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    total_pages = len(doc)
    doc.close()
    full = "\n".join(pages)
    logger.info(f"Extracted {len(full):,} chars from {pdf_path.name} ({total_pages} pages)")
    return full


# ── Document definitions ────────────────────────────────────────
DOCUMENTS = [
    {
        "source": PROJECT_ROOT / "src" / "ingestion" / "data" / "lc59_2001.txt",
        "type": "text",
        "metadata": {
            "filename": "lei_complementar_59_2001.txt",
            "tipo": "Lei Complementar",
            "numero": "59",
            "ano": 2001,
            "orgao": "Assembleia Legislativa de MG",
            "status_vigencia": "VIGENTE",
            "assunto_resumo": "Organização e Divisão Judiciária do Estado de Minas Gerais",
            "tags": [
                "organização judiciária", "divisão judiciária", "comarcas",
                "varas", "TJMG", "magistratura", "lei complementar 59"
            ],
        },
    },
    {
        "source": PROJECT_ROOT / "Regimento Interno.pdf",
        "type": "pdf",
        "metadata": {
            "filename": "regimento_interno_tjmg.pdf",
            "tipo": "Regimento Interno",
            "numero": "0",
            "ano": 2024,
            "orgao": "Tribunal de Justiça de MG",
            "status_vigencia": "VIGENTE",
            "assunto_resumo": "Regimento Interno do Tribunal de Justiça do Estado de Minas Gerais",
            "tags": [
                "regimento interno", "TJMG", "tribunal pleno",
                "órgão especial", "câmaras", "corregedoria"
            ],
        },
    },
]


# ── Main ingestion ──────────────────────────────────────────────
async def main():
    dsn = get_dsn()

    logger.info(f"Connecting to database...")
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error("Make sure Docker is running: docker-compose up -d")
        return

    logger.info("✅ Connected to database")

    for doc_def in DOCUMENTS:
        source = doc_def["source"]
        meta = doc_def["metadata"]
        label = f"{meta['tipo']} {meta['numero']}/{meta['ano']}"

        # Check if already ingested
        existing = await conn.fetchval(
            "SELECT id FROM documentos WHERE filename = $1",
            meta["filename"]
        )
        if existing:
            logger.info(f"⏭️  {label} already ingested (id={existing}), skipping")
            continue

        # Extract text
        if doc_def["type"] == "pdf":
            if not source.exists():
                logger.error(f"❌ PDF not found: {source}")
                continue
            text = extract_pdf_text(source)
        else:
            if not source.exists():
                logger.error(f"❌ Text file not found: {source}")
                continue
            text = source.read_text(encoding="utf-8")

        if len(text.strip()) < 100:
            logger.error(f"❌ {label}: text too short ({len(text)} chars)")
            continue

        logger.info(f"📄 {label}: {len(text):,} chars")

        # Chunk text
        chunks = chunk_text(text)
        logger.info(f"   → {len(chunks)} chunks")

        # Generate embeddings in batches (Gemini API limit)
        BATCH_SIZE = 10
        all_embeddings = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            logger.info(f"   → Embedding batch {i // BATCH_SIZE + 1}/{(len(chunks) - 1) // BATCH_SIZE + 1}")
            embs = get_embeddings(batch)
            all_embeddings.extend(embs)
            time.sleep(0.3)  # Rate limit

        logger.info(f"   → {len(all_embeddings)} embeddings generated")

        # Insert into database
        async with conn.transaction():
            doc_id = await conn.fetchval(
                """INSERT INTO documentos
                   (filename, gcs_uri, tipo, numero, ano, orgao,
                    status_vigencia, assunto_resumo, tags)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                   RETURNING id""",
                meta["filename"],
                str(source),
                meta["tipo"],
                meta["numero"],
                meta["ano"],
                meta["orgao"],
                meta["status_vigencia"],
                meta["assunto_resumo"],
                meta["tags"],
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

        logger.info(f"   ✅ {label} ingested as documento_id={doc_id} ({len(chunks)} chunks)")

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
