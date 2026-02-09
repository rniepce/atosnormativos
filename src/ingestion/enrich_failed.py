"""
Enrich Failed Docs
==================
Targets the ~125 docs that failed Phase 2 classification.
These have very short text (55-341 chars), so we:
1. Try Gemini with a more lenient prompt
2. Fall back to filename-based heuristic classification
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/enrichment.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")


# Filename-based type mapping
TYPE_PREFIXES = {
    "pu": "Provimento",
    "pc": "Portaria Conjunta",
    "po": "Portaria",
    "ri": "Resolução Interna",
    "rs": "Resolução",
    "os": "Ordem de Serviço",
    "av": "Aviso",
    "vc": "Vice-Corregedoria",
    "in": "Instrução",
    "er": "Emenda Regimental",
    "dl": "Deliberação",
}


def classify_from_filename(filename: str) -> dict:
    """Extract metadata heuristically from filename."""
    prefix = filename[:2].lower()
    tipo = TYPE_PREFIXES.get(prefix, "Ato Normativo")
    
    # Extract number and year from filename pattern like "pu00122017.doc"
    match = re.match(r'[a-z]{2}(\d{4})(\d{4})', filename.lower())
    numero = "0"
    ano = 0
    if match:
        numero = str(int(match.group(1)))  # Remove leading zeros
        ano = int(match.group(2))
    
    return {
        "tipo": tipo,
        "numero": numero,
        "ano": ano,
        "orgao": "TJMG",
        "status": "VIGENTE",
        "assunto_resumo": f"{tipo} nº {numero}/{ano}" if ano else f"{tipo} (sem dados)",
        "tags": [tipo.lower(), "tjmg", "ato normativo"]
    }


async def try_llm_classify(text: str, filename: str) -> dict:
    """Try LLM classification with a lenient prompt for short texts."""
    prompt = f"""Classifique este ato normativo do TJMG. O texto pode estar muito curto ou incompleto.
Use o nome do arquivo como pista adicional.

ARQUIVO: {filename}
TEXTO: {text}

Responda APENAS JSON válido:
{{"tipo": "string", "numero": "string", "ano": 0, "orgao": "string", "status": "VIGENTE", "assunto_resumo": "string", "tags": ["tag1", "tag2"]}}"""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        text_resp = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(text_resp)
        if "tipo" in data and "status" in data:
            return data
    except Exception as e:
        logger.warning(f"LLM failed for {filename}: {e}")
    return None


async def main():
    start = time.time()
    
    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB"),
        timeout=30
    )
    logger.info("✅ Connected to database")
    
    docs = await conn.fetch("""
        SELECT d.id, d.filename, string_agg(c.conteudo_texto, E'\n' ORDER BY c.id) as text
        FROM documentos d
        LEFT JOIN chunks c ON c.documento_id = d.id
        WHERE d.tags IS NULL OR array_length(d.tags, 1) IS NULL OR array_length(d.tags, 1) = 0
        GROUP BY d.id, d.filename
        ORDER BY d.id
    """)
    
    logger.info(f"📄 Docs to enrich: {len(docs)}")
    
    llm_ok, fallback_ok, failed = 0, 0, 0
    
    for i, doc in enumerate(docs, 1):
        doc_id = doc["id"]
        filename = doc["filename"]
        text = doc["text"] or ""
        
        logger.info(f"[{i}/{len(docs)}] {filename} ({len(text)} chars)")
        
        # Try LLM first
        metadata = await try_llm_classify(text, filename)
        
        if metadata:
            source = "LLM"
            llm_ok += 1
        else:
            # Fallback to filename heuristic
            metadata = classify_from_filename(filename)
            source = "heuristic"
            fallback_ok += 1
        
        try:
            await conn.execute("""
                UPDATE documentos 
                SET tipo = $1, numero = $2, ano = $3, orgao = $4,
                    status_vigencia = $5, assunto_resumo = $6, tags = $7
                WHERE id = $8
            """,
                str(metadata.get("tipo", "Ato Normativo") or "Ato Normativo")[:199],
                str(metadata.get("numero", "0") or "0")[:19],
                int(metadata.get("ano", 0) or 0),
                str(metadata.get("orgao", "TJMG") or "TJMG")[:99],
                str(metadata.get("status", "VIGENTE") or "VIGENTE")[:19],
                str(metadata.get("assunto_resumo", "") or ""),
                metadata.get("tags", []) or ["ato normativo"],
                doc_id
            )
            logger.info(f"  ✅ [{source}] {metadata.get('tipo')} {metadata.get('numero')}/{metadata.get('ano')}")
        except Exception as e:
            logger.error(f"  ❌ DB error: {e}")
            failed += 1
    
    await conn.close()
    
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("🎉 ENRICHMENT COMPLETE!")
    logger.info(f"   LLM classified: {llm_ok}")
    logger.info(f"   Fallback (filename): {fallback_ok}")
    logger.info(f"   Failed: {failed}")
    logger.info(f"   Time: {elapsed:.0f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
