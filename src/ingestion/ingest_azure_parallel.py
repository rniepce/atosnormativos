"""
Ingestão paralela usando Azure OpenAI para classificação (gpt-5.4-mini)
e embeddings (text-embedding-3-large, 1024-dim).

- Pipeline async com semáforo (concorrência configurável)
- Extração de texto (textutil/python-docx) em thread executor
- Classificação LLM com fallback para regex (classify_from_filename do ingest_local)
- Chunking semântico por artigos (chunk_by_articles)
- Insert idempotente: pula arquivos já presentes em documentos.filename
- Progresso persistido em /tmp/ingestion_azure_progress.json
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()

from src.ingestion.classify_azure import classify_with_llm, chunk_by_articles
from src.ingestion.common import extract_text, get_embeddings_async
from src.ingestion.ingest_local import classify_from_filename
from src.ingestion.rate_limiter import TokenBucket, estimate_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/ingestion_azure.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = Path("/tmp/ingestion_azure_progress.json")

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://assistente-web-resource.cognitiveservices.azure.com/")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
AZURE_EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1024"))


def _load_progress() -> set:
    if PROGRESS_FILE.exists():
        try:
            return set(json.loads(PROGRESS_FILE.read_text()))
        except Exception:
            return set()
    return set()


def _save_progress(done: set) -> None:
    PROGRESS_FILE.write_text(json.dumps(sorted(done)))


def _merge_with_filename(metadata: Dict[str, Any], filename: str, parent_dir: str) -> Dict[str, Any]:
    """Fill missing/zero number/year in LLM metadata using filename regex hints."""
    fallback = classify_from_filename(filename, parent_dir)
    numero = str(metadata.get("numero") or "0")
    ano = int(metadata.get("ano") or 0)
    if numero in ("", "0") and fallback.get("numero") not in ("", "0"):
        metadata["numero"] = fallback["numero"]
    if ano == 0 and fallback.get("ano"):
        metadata["ano"] = fallback["ano"]
    if not metadata.get("tipo") or metadata.get("tipo") == "Desconhecido":
        if fallback.get("tipo") and fallback.get("tipo") != "Desconhecido":
            metadata["tipo"] = fallback["tipo"]
    return metadata


async def _classify_one(
    text: str,
    filename: str,
    parent_dir: str,
    classify_pool: asyncio.Semaphore,
    use_llm: bool,
) -> Dict[str, Any]:
    if use_llm:
        async with classify_pool:
            metadata = await asyncio.to_thread(classify_with_llm, text, filename, parent_dir)
        if metadata is not None:
            return _merge_with_filename(metadata, filename, parent_dir)
        logger.warning(f"LLM classification failed for {filename}, using filename fallback")
    return classify_from_filename(filename, parent_dir)


async def _embed_with_bucket(
    chunks: List[str],
    embed_client: AsyncAzureOpenAI,
    bucket: TokenBucket,
    batch_size: int = 50,
) -> List[List[float]]:
    """Embed chunks in sub-batches, respecting the TokenBucket."""
    out: List[List[float]] = []
    for i in range(0, len(chunks), batch_size):
        sub = chunks[i:i + batch_size]
        await bucket.acquire(estimate_tokens(sub))
        embs = await get_embeddings_async(
            sub,
            embed_client,
            model=AZURE_EMBEDDING_MODEL,
            dimensions=AZURE_EMBEDDING_DIMENSIONS,
        )
        out.extend(embs)
    return out


async def _process_one(
    file_path: Path,
    pool: asyncpg.Pool,
    embed_client: AsyncAzureOpenAI,
    extract_pool: asyncio.Semaphore,
    classify_pool: asyncio.Semaphore,
    embed_pool: asyncio.Semaphore,
    embed_bucket: TokenBucket,
    use_llm: bool,
) -> Tuple[bool, str]:
    filename = file_path.name
    try:
        async with extract_pool:
            text = await asyncio.to_thread(extract_text, file_path)
        if text:
            text = text.replace("\x00", "")
        if not text or len(text.strip()) < 50:
            return False, "skip_empty"

        metadata = await _classify_one(text, filename, file_path.parent.name, classify_pool, use_llm)

        chunks = chunk_by_articles(text, metadata)
        if not chunks:
            return False, "skip_no_chunks"

        async with embed_pool:
            embeddings = await _embed_with_bucket(chunks, embed_client, embed_bucket)

        status_raw = metadata.get("status") or metadata.get("status_vigencia") or "VIGENTE"
        status_norm = str(status_raw).upper()[:20]
        if status_norm not in ("VIGENTE", "REVOGADO"):
            status_norm = "VIGENTE"
        numero_norm = str(metadata.get("numero") or "0")[:50]

        async with pool.acquire() as conn:
            async with conn.transaction():
                doc_id = await conn.fetchval(
                    """
                    INSERT INTO documentos (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                    """,
                    filename,
                    str(file_path),
                    metadata.get("tipo"),
                    numero_norm,
                    int(metadata.get("ano") or 0),
                    metadata.get("orgao"),
                    status_norm,
                    (metadata.get("assunto_resumo") or "")[:1000],
                    list(metadata.get("tags") or []),
                )
                rows = [(doc_id, c, str(list(e))) for c, e in zip(chunks, embeddings)]
                await conn.executemany(
                    "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                    rows,
                )
        return True, f"ok ({len(chunks)} chunks)"
    except Exception as exc:
        logger.error(f"Error processing {filename}: {exc}")
        return False, f"error: {type(exc).__name__}: {str(exc)[:100]}"


async def _existing_filenames(pool: asyncpg.Pool) -> set:
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT filename FROM documentos")
    return {r["filename"] for r in rows}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = sem limite")
    parser.add_argument("--no-llm", action="store_true", help="Pula classificação por LLM (só regex de filename)")
    parser.add_argument("--llm-concurrency", type=int, default=10)
    parser.add_argument("--embed-concurrency", type=int, default=10)
    parser.add_argument("--extract-concurrency", type=int, default=8)
    parser.add_argument("--db-pool-size", type=int, default=8)
    parser.add_argument("--embed-tpm", type=int, default=300_000, help="Token bucket capacity per minute")
    parser.add_argument("--embed-rpm", type=int, default=240, help="Requests per minute cap")
    parser.add_argument("--clean", action="store_true", help="DELETE FROM chunks/documentos antes")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"Directory not found: {root}")
        return

    if not AZURE_API_KEY:
        logger.error("AZURE_API_KEY missing")
        sys.exit(1)

    embed_client = AsyncAzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

    pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB"),
        min_size=2,
        max_size=args.db_pool_size,
    )

    if args.clean:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM chunks")
            await conn.execute("DELETE FROM documentos")
        logger.warning("Banco limpo (chunks + documentos)")

    files: List[Path] = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".doc", ".docx")) and not f.startswith("~$"):
                files.append(Path(r) / f)
    files.sort()

    if args.limit > 0:
        files = files[:args.limit]

    progress = _load_progress()
    existing = await _existing_filenames(pool)
    skip = progress | existing

    pending = [p for p in files if p.name not in skip]
    logger.info(f"Total encontrados: {len(files)} | já processados: {len(files) - len(pending)} | pendentes: {len(pending)}")
    if not pending:
        await pool.close()
        return

    extract_sem = asyncio.Semaphore(args.extract_concurrency)
    classify_sem = asyncio.Semaphore(args.llm_concurrency)
    embed_sem = asyncio.Semaphore(args.embed_concurrency)
    embed_bucket = TokenBucket(args.embed_tpm, args.embed_rpm)
    logger.info(f"Token bucket: {args.embed_tpm} TPM / {args.embed_rpm} RPM")

    use_llm = not args.no_llm
    if use_llm:
        logger.info(f"Classificação LLM ATIVA (Azure {os.getenv('AZURE_LLM_MODEL', 'gpt-5.4-mini')}, conc={args.llm_concurrency})")
    else:
        logger.info("Classificação LLM DESATIVADA (filename only)")

    success = 0
    failed = 0
    skipped = 0
    done_lock = asyncio.Lock()

    async def worker(fp: Path, idx: int, total: int) -> None:
        nonlocal success, failed, skipped
        ok, msg = await _process_one(fp, pool, embed_client, extract_sem, classify_sem, embed_sem, embed_bucket, use_llm)
        async with done_lock:
            if ok:
                success += 1
                progress.add(fp.name)
                if (success + failed + skipped) % 50 == 0:
                    _save_progress(progress)
            elif msg.startswith("skip"):
                skipped += 1
                progress.add(fp.name)
            else:
                failed += 1
        if (idx % 25) == 0 or ok is False:
            logger.info(f"[{idx}/{total}] {fp.name}: {msg}  (ok={success} fail={failed} skip={skipped})")

    tasks = [asyncio.create_task(worker(p, i + 1, len(pending))) for i, p in enumerate(pending)]
    await asyncio.gather(*tasks)
    _save_progress(progress)

    await pool.close()
    logger.info("=" * 60)
    logger.info(f"FIM. ok={success} fail={failed} skip={skipped} (de {len(pending)} pendentes)")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
