"""
Ingest Lei Complementar 59/2001 + Regimento Interno into the vector database.
Uses OpenAI text-embedding-3-large API with 1024 dimensions
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
from openai import AsyncOpenAI

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

# ── OpenAI setup ────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("❌ ERROR: OPENAI_API_KEY not found in environment")
    sys.exit(1)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_DIM = 1024  
MODEL = "text-embedding-3-large"

# ── Database connection ─────────────────────────────────────────
def get_dsn() -> str:
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "railway")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

# ── Embeddings ──────────────────────────────────────────────────
async def get_embeddings(texts: List[str]) -> List[List[float]]:
    try:
        response = await client.embeddings.create(
            input=texts,
            model=MODEL,
            dimensions=EMBEDDING_DIM
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise e

# ── Text chunking ───────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    chunks, start = [], 0
    text = text.strip()
    while start < len(text):
        chunks.append(text[start:start + chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

def extract_pdf_text(pdf_path: Path) -> str:
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

async def main():
    dsn = get_dsn()
    logger.info(f"Connecting to database...")
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return

    logger.info("✅ Connected to database")

    for doc_def in DOCUMENTS:
        source = doc_def["source"]
        meta = doc_def["metadata"]
        label = f"{meta['tipo']} {meta['numero']}/{meta['ano']}"

        existing = await conn.fetchval(
            "SELECT id FROM documentos WHERE filename = $1",
            meta["filename"]
        )
        if existing:
            logger.info(f"⏭️  {label} already ingested (id={existing}), skipping")
            continue

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
        chunks = chunk_text(text)
        logger.info(f"   → {len(chunks)} chunks")

        BATCH_SIZE = 50
        all_embeddings = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            logger.info(f"   → Embedding batch {i // BATCH_SIZE + 1}/{(len(chunks) - 1) // BATCH_SIZE + 1}")
            embs = await get_embeddings(batch)
            all_embeddings.extend(embs)
            await asyncio.sleep(0.05)

        logger.info(f"   → {len(all_embeddings)} embeddings generated")

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

    total_docs = await conn.fetchval("SELECT count(*) FROM documentos")
    logger.info("INGESTION COMPLETE")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
