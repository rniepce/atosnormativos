"""
Post-ingestion script to enrich document metadata using Gemini 2.0 Flash.
Extracts: Status (Vigente/Revogado), Resumo, Órgão, Tags.
"""
import sys
import os
import asyncio
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Add src to pythonpath
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.vertex import VertexAIClient
from src.utils.db import get_db_connection

# Load environment
load_dotenv('/Users/rafaelpimentel/Downloads/atosnormativos/.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/enrichment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Vertex AI
print("Initializing Vertex AI Client...", flush=True)
vertex_client = VertexAIClient()

async def get_document_text(conn, doc_id: int) -> str:
    """Reconstruct document text from chunks."""
    rows = await conn.fetch(
        'SELECT conteudo_texto FROM chunks WHERE documento_id = $1 ORDER BY id', 
        doc_id
    )
    return "\n".join([r['conteudo_texto'] for r in rows])

async def generate_metadata(text: str) -> dict:
    """Use LLM to generate metadata."""
    prompt = f"""
    Analise o seguinte ato normativo e extraia as informações abaixo em formato JSON.
    Se a informação não estiver clara, use null.
    
    1. status: "VIGENTE", "REVOGADO" ou "ALTERADO". Se houver menção expressa de revogação ou texto riscado, marque como REVOGADO.
    2. resumo: Um resumo conciso do que trata o ato (máximo 2 frases).
    3. orgao: Órgão emissor (ex: Presidência, Corregedoria, Secretaria de Fazenda).
    4. tags: Lista de 3-5 palavras-chave relevantes.

    Documento:
    {text[:10000]}  # Limit context window just in case
    
    Responda APENAS o JSON.
    """
    
    try:
        response = vertex_client.generate_text(
            prompt, 
            model_name="gemini-2.0-flash-exp",
            response_mime_type="application/json"
        )
        return response
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return None

async def main():
    conn = await get_db_connection()
    
    # Get documents that need enrichment (e.g., where status is default or summary is just title)
    # For now, let's process all documents that haven't been enriched (custom flag or null check)
    # We'll use a simple check: assume 'Tags' is empty for new docs
    docs = await conn.fetch('''
        SELECT id, filename, tipo FROM documentos 
        WHERE tags IS NULL OR array_length(tags, 1) IS NULL
        ORDER BY id DESC
    ''')
    
    logger.info(f"Docs to enrich: {len(docs)}")
    
    import json
    
    for i, doc in enumerate(docs):
        doc_id = doc['id']
        filename = doc['filename']
        
        logger.info(f"[{i+1}/{len(docs)}] Enriching {filename}...")
        
        full_text = await get_document_text(conn, doc_id)
        if not full_text:
            logger.warning(f"No text found for {filename}")
            continue
            
        json_str = await generate_metadata(full_text)
        
        if json_str:
            try:
                data = json.loads(json_str)
                
                status = data.get('status', 'VIGENTE')
                resumo = data.get('resumo')
                orgao = data.get('orgao')
                tags = data.get('tags', [])
                
                # Update DB
                await conn.execute('''
                    UPDATE documentos 
                    SET status_vigencia = $1, 
                        assunto_resumo = $2, 
                        orgao = COALESCE($3, orgao), -- Keep existing if LLM returns null
                        tags = $4
                    WHERE id = $5
                ''', status, resumo, orgao, tags, doc_id)
                
                logger.info(f"Updated {filename}: {status} | {resumo[:50]}...")
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON for {filename}: {json_str}")
            except Exception as e:
                logger.error(f"Error updating DB for {filename}: {e}")
        
        # Rate limit friendly
        await asyncio.sleep(0.5)

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
