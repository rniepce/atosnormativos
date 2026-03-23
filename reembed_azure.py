"""
Re-Embedding Script v3 — Máxima Qualidade
==========================================
Atualiza todos os embeddings do banco para Azure text-embedding-3-large,
com paralelismo real (asyncio), re-chunking jurídico opcional e limpeza
de texto antes de embeddar.

Modos de uso:
    python3 reembed_azure.py                    # Re-embeda chunks existentes
    python3 reembed_azure.py --rechunk          # Re-chunkeia + re-embeda (melhor qualidade)
    python3 reembed_azure.py --dry-run          # Simula sem alterar o banco
    python3 reembed_azure.py --limit 50         # Testa com 50 chunks/docs
    python3 reembed_azure.py --concurrency 10   # Controla chamadas paralelas à API
    python3 reembed_azure.py --no-resume        # Ignora progresso salvo e reinicia
    python3 reembed_azure.py --fix-duplicates   # Remove documentos duplicados antes
    python3 reembed_azure.py --fix-short        # Remove chunks curtos (<50 chars) antes

Flags combináveis:
    python3 reembed_azure.py --rechunk --fix-duplicates --fix-short
"""
import os
import re
import sys
import asyncio
import asyncpg
import json
import time
import logging
import argparse
import unicodedata
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv(Path(__file__).parent / ".env")

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/reembed_azure_v3.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
for noisy in ("openai", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ── Configuração ────────────────────────────────────────────────────────────
AZURE_API_KEY      = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT     = os.getenv("AZURE_ENDPOINT", "https://assistente-web-resource.cognitiveservices.azure.com/")
AZURE_API_VERSION  = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")
EMBEDDING_MODEL    = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM      = int(os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1024"))

API_BATCH_SIZE     = 100    # Chunks por chamada de API (máx. suportado: 2048)
DB_BATCH_SIZE      = 500    # Chunks por leitura do banco
DEFAULT_CONCURRENCY = 8     # Chamadas paralelas simultâneas à API
MAX_CHARS_PER_CHUNK = 32000 # ~8000 tokens (4 chars/token) — limite do modelo
PROGRESS_FILE      = "/tmp/reembed_v3_progress.json"
# ───────────────────────────────────────────────────────────────────────────


# ── Pré-processamento de texto ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Limpa o texto antes de embeddar para melhorar a qualidade do embedding."""
    if not text:
        return ""

    # Normaliza unicode (NFC garante composição canônica)
    text = unicodedata.normalize("NFC", text)

    # Remove null bytes e caracteres de controle (exceto whitespace)
    text = "".join(
        c for c in text
        if c in ("\n", "\r", "\t") or (ord(c) >= 32 and not unicodedata.category(c).startswith("C"))
    )

    # Colapsa whitespace excessivo preservando parágrafos
    text = re.sub(r"[ \t]+", " ", text)           # múltiplos espaços → um
    text = re.sub(r"\n{3,}", "\n\n", text)        # mais de 2 quebras → 2
    text = re.sub(r" *\n *", "\n", text)          # espaços em volta de quebras

    return text.strip()


def truncate(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> str:
    """Trunca texto ao limite de tokens do modelo (heurística: 4 chars/token)."""
    if len(text) <= max_chars:
        return text
    logger.debug(f"Truncando chunk de {len(text)} → {max_chars} chars")
    return text[:max_chars]


def prepare_text(text: str) -> str:
    """Pipeline completo de pré-processamento: limpa + trunca."""
    return truncate(clean_text(text))


# ── Cliente Azure ───────────────────────────────────────────────────────────

def get_async_client() -> AsyncAzureOpenAI:
    if not AZURE_API_KEY:
        logger.error("❌ AZURE_API_KEY não definida!")
        sys.exit(1)
    return AsyncAzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )


# ── Geração de embeddings com retry ────────────────────────────────────────

async def embed_batch(client: AsyncAzureOpenAI, texts: List[str], semaphore: asyncio.Semaphore) -> Optional[List[List[float]]]:
    """Embeda um batch de textos com retry e exponential backoff."""
    prepared = [prepare_text(t) for t in texts]

    async with semaphore:
        for attempt in range(5):
            try:
                response = await client.embeddings.create(
                    input=prepared,
                    model=EMBEDDING_MODEL,
                    dimensions=EMBEDDING_DIM,
                )
                return [d.embedding for d in response.data]

            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "rate" in err.lower() or "RateLimitError" in type(e).__name__
                is_transient  = "503" in err or "502" in err or "timeout" in err.lower() or "ConnectionError" in type(e).__name__

                if is_rate_limit:
                    wait = 20 * (2 ** attempt)  # 20s, 40s, 80s, 160s, 320s
                    logger.warning(f"⏳ Rate limit (tentativa {attempt+1}/5) — aguardando {wait}s...")
                    await asyncio.sleep(wait)
                elif is_transient and attempt < 4:
                    wait = 3 * (2 ** attempt)   # 3s, 6s, 12s, 24s
                    logger.warning(f"⚠️  Erro transiente (tentativa {attempt+1}/5): {err[:80]} — retry em {wait}s")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"❌ Falha definitiva no batch após {attempt+1} tentativas: {err[:120]}")
                    return None

    return None


# ── Progresso ───────────────────────────────────────────────────────────────

def load_progress() -> set:
    try:
        with open(PROGRESS_FILE) as f:
            return set(json.load(f).get("done", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_progress(done: set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"done": list(done)}, f)


# ── Manutenção do banco ─────────────────────────────────────────────────────

async def fix_duplicates(conn, dry_run: bool):
    dupes = await conn.fetch("""
        SELECT filename, array_agg(id ORDER BY id) as ids
        FROM documentos GROUP BY filename HAVING COUNT(*) > 1
    """)
    if not dupes:
        logger.info("✅ Sem documentos duplicados")
        return
    total = sum(len(row["ids"]) - 1 for row in dupes)
    if not dry_run:
        for row in dupes:
            for doc_id in list(row["ids"])[1:]:
                await conn.execute("DELETE FROM chunks WHERE documento_id = $1", doc_id)
                await conn.execute("DELETE FROM documentos WHERE id = $1", doc_id)
    logger.info(f"{'[DRY-RUN] Removeria' if dry_run else 'Removidos'} {total} docs duplicados")


async def fix_short_chunks(conn, dry_run: bool):
    count = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE LENGTH(conteudo_texto) < 50")
    if count == 0:
        logger.info("✅ Sem chunks curtos")
        return
    if not dry_run:
        await conn.execute("DELETE FROM chunks WHERE LENGTH(conteudo_texto) < 50")
    logger.info(f"{'[DRY-RUN] Removeria' if dry_run else 'Removidos'} {count} chunks curtos")


# ── Modo 1: Re-embedding dos chunks existentes ──────────────────────────────

async def reembed_existing(conn, client, args, semaphore):
    """Re-embeda os chunks já existentes no banco, sem alterar a divisão."""
    query = "SELECT id FROM chunks ORDER BY documento_id, id"
    if args.limit:
        query += f" LIMIT {args.limit}"

    all_ids = [r["id"] for r in await conn.fetch(query)]

    done = set() if args.no_resume else load_progress()
    to_process = [i for i in all_ids if i not in done]

    logger.info(f"📝 {len(to_process)} chunks para re-embedar ({len(done)} já processados)")
    if not to_process:
        logger.info("✅ Tudo já processado!")
        return

    # Estimativa de custo
    avg_chars = float(await conn.fetchval("SELECT AVG(LENGTH(conteudo_texto)) FROM chunks WHERE id = ANY($1::int[])", to_process[:1000]))
    est_tokens = len(to_process) * avg_chars / 4
    logger.info(f"💰 Estimativa: ~{est_tokens/1e6:.1f}M tokens (~${est_tokens/1000*0.00013:.2f} USD)")

    if args.dry_run:
        logger.info("🏁 [DRY-RUN] Concluído.")
        return

    start = time.time()
    success = failed = 0

    # Processa em batches de DB_BATCH_SIZE, paralelizando as chamadas de API dentro de cada batch
    for db_start in range(0, len(to_process), DB_BATCH_SIZE):
        db_ids = to_process[db_start : db_start + DB_BATCH_SIZE]

        rows = await conn.fetch(
            "SELECT id, conteudo_texto FROM chunks WHERE id = ANY($1::int[]) ORDER BY id",
            db_ids,
        )
        chunk_data = [(r["id"], r["conteudo_texto"]) for r in rows]

        # Divide em sub-batches de API_BATCH_SIZE e dispara em paralelo
        api_batches = [
            chunk_data[i : i + API_BATCH_SIZE]
            for i in range(0, len(chunk_data), API_BATCH_SIZE)
        ]

        tasks = [
            embed_batch(client, [t for _, t in b], semaphore)
            for b in api_batches
        ]
        results = await asyncio.gather(*tasks)

        # Monta updates
        updates = []
        for batch, embeddings in zip(api_batches, results):
            if embeddings is None:
                failed += len(batch)
                continue
            for (cid, _), emb in zip(batch, embeddings):
                updates.append((str(emb), cid))

        # Grava no banco
        if updates and not args.dry_run:
            try:
                await conn.executemany(
                    "UPDATE chunks SET embedding = $1::vector WHERE id = $2",
                    updates,
                )
                success += len(updates)
                done.update(cid for _, cid in updates)
            except Exception as e:
                logger.error(f"❌ Erro no banco: {e}")
                failed += len(updates)

        save_progress(done)

        elapsed = time.time() - start
        total_done = success + failed
        rate = total_done / elapsed if elapsed > 0 else 0
        eta = (len(to_process) - total_done) / rate if rate > 0 else 0
        logger.info(
            f"  [{total_done}/{len(to_process)}] "
            f"✅{success} ❌{failed} | "
            f"{rate:.1f}/s | ETA {eta/60:.1f}min"
        )

    return success, failed


# ── Modo 2: Re-chunking + re-embedding (melhor qualidade) ──────────────────

async def rechunk_and_reembed(pool, client, args, semaphore):
    """
    Lê o texto completo de cada documento (concatenando chunks em ordem),
    re-divide com o chunker jurídico do common.py e grava novos embeddings.
    Adquire uma conexão nova do pool por documento para evitar timeouts.
    """
    from src.ingestion.common import chunk_text

    # Usa conexão temporária só para leitura inicial
    async with pool.acquire() as conn:
        query = "SELECT id FROM documentos ORDER BY id"
        if args.limit:
            query += f" LIMIT {args.limit}"
        all_doc_ids = [r["id"] for r in await conn.fetch(query)]

        done = set() if args.no_resume else load_progress()
        to_process = [i for i in all_doc_ids if i not in done]

        logger.info(f"📝 {len(to_process)} documentos para re-chunkar + re-embedar ({len(done)} já processados)")

        avg_chars = float(await conn.fetchval("SELECT AVG(LENGTH(conteudo_texto)) FROM chunks"))
        avg_chunks = float(await conn.fetchval("SELECT COUNT(*)::float / NULLIF(COUNT(DISTINCT documento_id), 0) FROM chunks"))
        est_tokens = len(to_process) * avg_chunks * avg_chars / 4
        logger.info(f"💰 Estimativa: ~{est_tokens/1e6:.1f}M tokens (~${est_tokens/1000*0.00013:.2f} USD)")

    if args.dry_run:
        logger.info("🏁 [DRY-RUN] Concluído.")
        return

    start = time.time()
    success = failed = 0
    total_new_chunks = 0

    for i, doc_id in enumerate(to_process, 1):
        try:
            # Conexão nova por documento — evita timeout em operações longas
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT conteudo_texto FROM chunks WHERE documento_id = $1 ORDER BY id",
                    doc_id,
                )
                if not rows:
                    done.add(doc_id)
                    continue

                texts = [r["conteudo_texto"] for r in rows]
                full_text = texts[0]
                for t in texts[1:]:
                    overlap_removed = t[200:] if len(t) > 200 else t
                    if overlap_removed.strip():
                        full_text += "\n" + overlap_removed

            # Re-chunkeia (fora do contexto de conexão — CPU only)
            new_chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
            if not new_chunks:
                done.add(doc_id)
                continue

            prepared = [prepare_text(c) for c in new_chunks]
            prepared = [t for t in prepared if t]
            if not prepared:
                done.add(doc_id)
                continue

            # Embeda via API (sem conexão DB aberta — evita timeout)
            api_batches = [
                prepared[j : j + API_BATCH_SIZE]
                for j in range(0, len(prepared), API_BATCH_SIZE)
            ]
            tasks = [embed_batch(client, b, semaphore) for b in api_batches]
            results = await asyncio.gather(*tasks)

            if any(r is None for r in results):
                logger.error(f"❌ Doc {doc_id}: falha em batch de embedding, pulando")
                failed += 1
                continue

            all_embeddings = [emb for batch_embs in results for emb in batch_embs]

            # Nova conexão para gravação no banco
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("DELETE FROM chunks WHERE documento_id = $1", doc_id)
                    chunk_rows = [
                        (doc_id, text, str(emb))
                        for text, emb in zip(prepared, all_embeddings)
                    ]
                    await conn.executemany(
                        "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                        chunk_rows,
                    )

            success += 1
            total_new_chunks += len(prepared)
            done.add(doc_id)

        except Exception as e:
            logger.error(f"❌ Doc {doc_id}: {e}")
            failed += 1

        if i % 10 == 0 or i == len(to_process):
            save_progress(done)
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i) / rate if rate > 0 else 0
            logger.info(
                f"  [{i}/{len(to_process)}] "
                f"✅{success} ❌{failed} | "
                f"~{total_new_chunks} novos chunks | "
                f"{rate:.2f} docs/s | ETA {eta/60:.1f}min"
            )

    return success, failed


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Re-embeda o banco com Azure text-embedding-3-large")
    parser.add_argument("--rechunk",         action="store_true", help="Re-chunkeia com chunker jurídico antes de embeddar (melhor qualidade)")
    parser.add_argument("--dry-run",         action="store_true", help="Simula sem alterar o banco")
    parser.add_argument("--limit",           type=int, default=0, help="Processa apenas N chunks/docs (para testes)")
    parser.add_argument("--concurrency",     type=int, default=DEFAULT_CONCURRENCY, help="Chamadas paralelas simultâneas à API")
    parser.add_argument("--no-resume",       action="store_true", help="Ignora progresso salvo e reinicia do zero")
    parser.add_argument("--fix-duplicates",  action="store_true", help="Remove documentos duplicados antes de embeddar")
    parser.add_argument("--fix-short",       action="store_true", help="Remove chunks curtos (<50 chars) antes de embeddar")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🚀 RE-EMBEDDING v3 — Azure text-embedding-3-large")
    logger.info(f"   Modo: {'--rechunk (re-chunking jurídico)' if args.rechunk else 'embedding dos chunks existentes'}")
    logger.info(f"   Concorrência: {args.concurrency} chamadas paralelas")
    logger.info(f"   Dry-run: {args.dry_run}")
    logger.info("=" * 60)

    # Cliente Azure (async)
    client = get_async_client()

    # Testa API
    logger.info("🔗 Testando API...")
    try:
        resp = await client.embeddings.create(
            input=["teste de conectividade em português jurídico"],
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIM,
        )
        assert len(resp.data[0].embedding) == EMBEDDING_DIM
        logger.info(f"✅ API OK ({EMBEDDING_DIM}-dim)")
    except Exception as e:
        logger.error(f"❌ Falha na API: {e}")
        sys.exit(1)

    # Conecta ao banco
    logger.info("🔗 Conectando ao banco...")
    try:
        pool = await asyncpg.create_pool(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            database=os.getenv("POSTGRES_DB"),
            min_size=2,
            max_size=5,
            command_timeout=60,
        )
        logger.info("✅ Banco conectado")
    except Exception as e:
        logger.error(f"❌ Falha na conexão: {e}")
        sys.exit(1)

    # Manutenção prévia
    async with pool.acquire() as conn:
        if args.fix_duplicates:
            await fix_duplicates(conn, dry_run=args.dry_run)
        if args.fix_short:
            await fix_short_chunks(conn, dry_run=args.dry_run)

        total_docs   = await conn.fetchval("SELECT COUNT(*) FROM documentos")
        total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")
        logger.info(f"📊 Estado atual: {total_docs} documentos, {total_chunks} chunks")

    # Semáforo para controlar paralelismo
    semaphore = asyncio.Semaphore(args.concurrency)

    # Executa modo escolhido
    start = time.time()
    if args.rechunk:
        result = await rechunk_and_reembed(pool, client, args, semaphore)
    else:
        async with pool.acquire() as conn:
            result = await reembed_existing(conn, client, args, semaphore)

    await pool.close()
    await client.close()

    if result:
        success, failed = result
        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"🎉 CONCLUÍDO em {elapsed/60:.1f}min")
        logger.info(f"   ✅ Sucesso: {success}")
        logger.info(f"   ❌ Falhas:  {failed}")
        logger.info(f"   ⚡ Taxa:    {(success+failed)/elapsed:.1f}/s")
        logger.info("=" * 60)

        if failed > 0:
            logger.warning(f"⚠️  {failed} itens falharam. Re-execute sem --no-resume para tentar novamente.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⛔ Interrompido. Progresso salvo — re-execute para continuar.")
