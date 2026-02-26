"""
Re-Embedding Script: Azure text-embedding-3-large (v2 - optimized)
==================================================================
Updates all chunk embeddings in the database to use the same model
as the search pipeline (Azure OpenAI text-embedding-3-large, 1024-dim).

Optimizations over v1:
- executemany for batch DB writes (fewer round-trips to Railway)
- Larger API batch size (100 inputs per call)
- Larger DB batch size (500 reads at a time)

Usage:
    python3 reembed_azure.py                  # Full re-embedding
    python3 reembed_azure.py --dry-run        # Preview only (no changes)
    python3 reembed_azure.py --limit 10       # Test with 10 chunks
    python3 reembed_azure.py --fix-duplicates # Also remove duplicate docs
    python3 reembed_azure.py --fix-short      # Also remove short chunks (<50 chars)
"""
import os
import sys
import asyncio
import asyncpg
import json
import time
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment
load_dotenv(Path(__file__).parent / ".env")

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress HTTP request logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/reembed_azure.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
# Set our logger to INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Suppress noisy HTTP logs from openai/httpx
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Configuration ──────────────────────────────────────────────
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://assistente-web-resource.cognitiveservices.azure.com/")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1024"))

API_BATCH_SIZE = 100      # Chunks per API call
DB_BATCH_SIZE = 500       # Chunks per DB read/write batch
PROGRESS_FILE = "/tmp/reembed_progress.json"
# ───────────────────────────────────────────────────────────────


def get_azure_client():
    if not AZURE_API_KEY:
        logger.error("❌ AZURE_API_KEY not set!")
        sys.exit(1)
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )
    logger.info(f"✅ Azure client ready (model={EMBEDDING_MODEL}, dim={EMBEDDING_DIM})")
    return client


def generate_embeddings(client, texts):
    response = client.embeddings.create(
        input=texts, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIM,
    )
    return [data.embedding for data in response.data]


def load_progress():
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f).get("processed_ids", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_progress(processed_ids):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({"processed_ids": list(processed_ids)}, f)


async def fix_duplicates(conn, dry_run=False):
    dupes = await conn.fetch("""
        SELECT filename, array_agg(id ORDER BY id) as ids
        FROM documentos GROUP BY filename HAVING COUNT(*) > 1
    """)
    if not dupes:
        logger.info("✅ No duplicates")
        return 0
    total = 0
    for row in dupes:
        ids_to_remove = list(row['ids'])[1:]
        total += len(ids_to_remove)
        if not dry_run:
            for doc_id in ids_to_remove:
                await conn.execute("DELETE FROM chunks WHERE documento_id = $1", doc_id)
                await conn.execute("DELETE FROM documentos WHERE id = $1", doc_id)
    logger.info(f"{'[DRY-RUN] Would remove' if dry_run else 'Removed'} {total} duplicate docs")
    return total


async def fix_short_chunks(conn, dry_run=False):
    count = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE LENGTH(conteudo_texto) < 50")
    if count == 0:
        return 0
    if not dry_run:
        await conn.execute("DELETE FROM chunks WHERE LENGTH(conteudo_texto) < 50")
    logger.info(f"{'[DRY-RUN] Would remove' if dry_run else 'Removed'} {count} short chunks")
    return count


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fix-duplicates", action="store_true")
    parser.add_argument("--fix-short", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    client = get_azure_client()

    # Test API
    logger.info("🔗 Testing API...")
    try:
        test = generate_embeddings(client, ["teste de conectividade em português"])
        assert len(test[0]) == EMBEDDING_DIM
        logger.info(f"✅ API OK ({EMBEDDING_DIM}-dim)")
    except Exception as e:
        logger.error(f"❌ API failed: {e}")
        sys.exit(1)

    # Connect DB
    logger.info("🔗 Connecting to DB...")
    try:
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT")),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB"),
        )
        logger.info("✅ DB connected")
    except Exception as e:
        logger.error(f"❌ DB failed: {e}")
        sys.exit(1)

    # Cleanup
    if args.fix_duplicates:
        await fix_duplicates(conn, dry_run=args.dry_run)
    if args.fix_short:
        await fix_short_chunks(conn, dry_run=args.dry_run)

    total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
    total_docs = await conn.fetchval("SELECT COUNT(*) FROM documentos")
    logger.info(f"📊 {total_docs} docs, {total_chunks} chunks")

    # Resume
    processed_ids = set() if args.no_resume else load_progress()
    if processed_ids:
        logger.info(f"📂 Resuming from {len(processed_ids)} already done")

    # Get chunks to process
    if args.limit > 0:
        all_ids = await conn.fetch("SELECT id FROM chunks ORDER BY id LIMIT $1", args.limit)
    else:
        all_ids = await conn.fetch("SELECT id FROM chunks ORDER BY id")

    to_process = [r['id'] for r in all_ids if r['id'] not in processed_ids]
    logger.info(f"📝 {len(to_process)} chunks to process")

    if not to_process:
        logger.info("✅ All done!")
        await conn.close()
        return

    # Cost estimate
    avg_chars = float(await conn.fetchval("SELECT AVG(LENGTH(conteudo_texto)) FROM chunks"))
    est_tokens = len(to_process) * (avg_chars / 4)
    logger.info(f"💰 ~${est_tokens / 1000 * 0.00013:.2f} USD ({est_tokens/1e6:.1f}M tokens)")

    if args.dry_run:
        logger.info("🏁 [DRY-RUN] Done.")
        await conn.close()
        return

    # ═══════════════════════════════════════════════════════════
    logger.info(f"🚀 Re-embedding {len(to_process)} chunks...")
    start = time.time()
    success, failed = 0, 0

    for batch_start in range(0, len(to_process), DB_BATCH_SIZE):
        batch_ids = to_process[batch_start:batch_start + DB_BATCH_SIZE]

        # Read texts
        rows = await conn.fetch(
            "SELECT id, conteudo_texto FROM chunks WHERE id = ANY($1::int[]) ORDER BY id",
            batch_ids
        )
        chunk_data = [(r['id'], r['conteudo_texto']) for r in rows]

        # Embed in API batches, accumulate all updates
        updates = []
        for api_start in range(0, len(chunk_data), API_BATCH_SIZE):
            api_batch = chunk_data[api_start:api_start + API_BATCH_SIZE]
            texts = [t for _, t in api_batch]
            ids = [i for i, _ in api_batch]

            embeddings = None
            for attempt in range(3):
                try:
                    embeddings = generate_embeddings(client, texts)
                    break
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        wait = (attempt + 1) * 15
                        logger.warning(f"⏳ Rate limited, wait {wait}s...")
                        await asyncio.sleep(wait)
                    elif attempt < 2:
                        logger.warning(f"⚠️ API retry {attempt+1}: {e}")
                        await asyncio.sleep(3)
                    else:
                        logger.error(f"❌ API failed: {e}")
                        failed += len(api_batch)

            if embeddings:
                for cid, emb in zip(ids, embeddings):
                    updates.append((str(emb), cid))

        # Batch write to DB (single executemany = 1 round-trip) 
        if updates:
            try:
                await conn.executemany(
                    "UPDATE chunks SET embedding = $1::vector WHERE id = $2",
                    updates
                )
                success += len(updates)
                processed_ids.update(uid for _, uid in updates)
            except Exception as e:
                logger.error(f"❌ DB error: {e}")
                failed += len(updates)

        # Progress
        save_progress(processed_ids)
        done = success + failed
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(to_process) - done) / rate if rate > 0 else 0
        logger.info(
            f"  [{done}/{len(to_process)}] "
            f"✅{success} ❌{failed} | "
            f"{rate:.1f}/s | ETA {eta/60:.1f}min"
        )

    save_progress(processed_ids)
    await conn.close()

    total_time = time.time() - start
    logger.info("=" * 50)
    logger.info(f"🎉 DONE! ✅{success} ❌{failed} in {total_time/60:.1f}min ({success/max(total_time,1):.1f}/s)")
    logger.info("=" * 50)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⛔ Stopped. Progress saved — re-run to resume.")
