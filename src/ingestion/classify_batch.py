"""
Batch LLM Classification Script
=================================
Classifies all documents using Gemini 2.0 Flash.
- Reconstructs text from chunks
- Extracts: tipo, numero, ano, orgao, status, assunto_resumo, tags
- Resume-safe (skips docs that already have tags)
- Rate-limited with retry logic
- Auto-reconnects to DB on failure
"""
import os
import sys
import json
import asyncio
import asyncpg
import logging
import time
import re
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv("/Users/rafaelpimentel/Downloads/atosnormativos/.env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/classification.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============= CONFIGURATION =============
MODEL_NAME = "gemini-2.0-flash"
MAX_RPM = 500               # Higher rate for paid tier
BATCH_LOG_EVERY = 25
MAX_TEXT_CHARS = 15000       # Max chars to send to LLM per doc
FAILURE_LOG = "/tmp/classification_failures.log"
# ==========================================


# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("❌ GEMINI_API_KEY not set!")
    sys.exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)
logger.info(f"✅ Gemini model initialized: {MODEL_NAME}")


CLASSIFICATION_PROMPT = """Você é um classificador especializado em atos normativos do Tribunal de Justiça de Minas Gerais (TJMG).

Analise o ato normativo abaixo e extraia os metadados.

INSTRUÇÕES:
1. **TIPO**: Identifique o tipo exato (Resolução, Portaria, Portaria Conjunta, Provimento, Aviso, Instrução, Ordem de Serviço, Emenda Regimental, Deliberação, Ato Normativo, etc.)
2. **NÚMERO e ANO**: Extraia do cabeçalho. Se não encontrar, use "0" e 0.
3. **ÓRGÃO**: Identifique o órgão emissor (Presidência, Corregedoria, Vice-Presidência, Secretaria, Escola Judicial, etc.)
4. **STATUS**: 
   - "REVOGADO" se contiver menção de revogação, perda de efeito, etc.
   - "VIGENTE" caso contrário
5. **ASSUNTO**: Resumo conciso do tema (máx 2 frases)
6. **TAGS**: 3-5 palavras-chave para busca

ARQUIVO: {filename}

Responda APENAS com JSON válido:
{{"tipo": "string", "numero": "string", "ano": 0, "orgao": "string", "status": "VIGENTE", "assunto_resumo": "string", "tags": ["tag1", "tag2", "tag3"]}}

TEXTO:
{text}"""


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
    try:
        if conn and not conn.is_closed():
            await conn.close()
    except Exception:
        pass
    logger.info("🔄 Reconnecting to database...")
    new_conn = await get_connection()
    logger.info("✅ Reconnected!")
    return new_conn


def detect_revocation_patterns(text: str) -> bool:
    """Detect patterns that indicate revoked text."""
    patterns = [
        r'revogad[oa]', r'sem efeito', r'perde\s*efeito',
        r'perdeu\s*eficácia', r'revoga-se', r'fica\s*revogad[oa]',
        r'torna\s*sem\s*efeito', r'deixa\s*de\s*vigorar',
        r'\(revogad[oa]\)', r'REVOGAD[OA]',
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


async def classify_document(text: str, filename: str) -> dict:
    """Classify a single document using Gemini."""
    # Truncate text for API
    truncated = text[:MAX_TEXT_CHARS]
    
    prompt = CLASSIFICATION_PROMPT.format(
        filename=filename,
        text=truncated
    )
    
    # Add revocation hint
    if detect_revocation_patterns(text):
        prompt = prompt.replace(
            "TEXTO:",
            "⚠️ ATENÇÃO: Detectados possíveis indicadores de revogação.\n\nTEXTO:"
        )
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        
        response_text = response.text.strip()
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(response_text)
        
        # Validate required fields
        if "tipo" not in data or "status" not in data:
            return None
        
        # Normalize
        data["status"] = data.get("status", "VIGENTE").upper()
        if data["status"] not in ("VIGENTE", "REVOGADO", "ALTERADO"):
            data["status"] = "VIGENTE"
        
        data["ano"] = int(data.get("ano", 0) or 0)
        data["numero"] = str(data.get("numero", "0") or "0")
        data["tags"] = data.get("tags", []) or []
        if isinstance(data["tags"], str):
            data["tags"] = [data["tags"]]
        
        return data
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for {filename}: {e}")
        return None
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RATE_LIMIT" in error_str.upper():
            raise  # Re-raise rate limits for retry
        logger.error(f"Gemini error for {filename}: {e}")
        return None


async def main():
    start_time = time.time()
    
    conn = await get_connection()
    logger.info("✅ Connected to database")
    
    # Get docs that need classification (no tags = not classified yet)
    docs = await conn.fetch("""
        SELECT id, filename 
        FROM documentos 
        WHERE tags IS NULL OR array_length(tags, 1) IS NULL OR array_length(tags, 1) = 0
        ORDER BY id
    """)
    
    logger.info(f"📄 Documents to classify: {len(docs)}")
    
    if not docs:
        logger.info("✅ All documents already classified!")
        await conn.close()
        return
    
    open(FAILURE_LOG, 'w').close()
    
    success, failed, skipped = 0, 0, 0
    request_times = []  # For rate limiting
    
    for i, doc in enumerate(docs, 1):
        doc_id = doc["id"]
        filename = doc["filename"]
        
        # Progress log
        if i % BATCH_LOG_EVERY == 0 or i == 1:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = len(docs) - i
            eta = remaining / rate if rate > 0 else 0
            logger.info(
                f"[{i}/{len(docs)}] ✅{success} ❌{failed} ⏭{skipped} | "
                f"{rate:.2f} docs/s | ETA: {eta/60:.0f} min"
            )
        
        try:
            # Rate limiting
            now = time.time()
            request_times = [t for t in request_times if now - t < 60]
            if len(request_times) >= MAX_RPM:
                wait_time = 60 - (now - request_times[0]) + 0.5
                logger.info(f"⏳ Rate limit, waiting {wait_time:.0f}s...")
                await asyncio.sleep(wait_time)
            
            # Get document text from chunks
            conn = await ensure_connection(conn)
            rows = await conn.fetch(
                "SELECT conteudo_texto FROM chunks WHERE documento_id = $1 ORDER BY id",
                doc_id
            )
            
            if not rows:
                skipped += 1
                continue
            
            text = "\n".join([r["conteudo_texto"] for r in rows])
            
            # Classify with retry for rate limits
            metadata = None
            for attempt in range(3):
                try:
                    metadata = await classify_document(text, filename)
                    request_times.append(time.time())
                    break
                except Exception as e:
                    if "429" in str(e) or "RATE_LIMIT" in str(e).upper():
                        wait = (attempt + 1) * 15
                        logger.warning(f"⏳ Rate limited, retry in {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        raise
            
            if not metadata:
                failed += 1
                with open(FAILURE_LOG, 'a') as f:
                    f.write(f"{filename}|classification_failed\n")
                continue
            
            # Update DB — truncate to fit column limits
            conn = await ensure_connection(conn)
            await conn.execute("""
                UPDATE documentos 
                SET tipo = $1, numero = $2, ano = $3, orgao = $4,
                    status_vigencia = $5, assunto_resumo = $6, tags = $7
                WHERE id = $8
            """,
                str(metadata.get("tipo", "Desconhecido") or "Desconhecido")[:199],
                str(metadata.get("numero", "0") or "0")[:19],
                metadata.get("ano", 0),
                str(metadata.get("orgao", "TJMG") or "TJMG")[:99],
                str(metadata.get("status", "VIGENTE") or "VIGENTE")[:19],
                str(metadata.get("assunto_resumo", "") or ""),
                metadata.get("tags", []) or [],
                doc_id
            )
            
            success += 1
            
        except Exception as e:
            logger.error(f"❌ Failed {filename}: {e}")
            with open(FAILURE_LOG, 'a') as f:
                f.write(f"{filename}|{str(e)[:100]}\n")
            failed += 1
    
    try:
        await conn.close()
    except Exception:
        pass
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🎉 CLASSIFICATION COMPLETE!")
    logger.info(f"   Success: {success}")
    logger.info(f"   Failed: {failed}")
    logger.info(f"   Skipped: {skipped}")
    logger.info(f"   Time: {total_time/60:.1f} min")
    logger.info(f"   Rate: {success/(total_time/60):.1f} docs/min")
    logger.info(f"   Failures: {FAILURE_LOG}")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⛔ Stopped by user")
