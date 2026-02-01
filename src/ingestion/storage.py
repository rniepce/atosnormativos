import logging
import asyncpg
import os
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

from src.utils.db import StorageParams

class DocumentStorage:
    def __init__(self):
        self.params = StorageParams()

    async def save_document_and_chunks(self, filename: str, gcs_uri: str, metadata: Dict[str, Any], chunks: List[Dict[str, Any]]):
        conn = await asyncpg.connect(self.params.dsn)
        try:
            async with conn.transaction():
                # 1. Insert Document
                # "tags" is a list, needs valid postgres array literal or let asyncpg handle it
                
                doc_query = """
                    INSERT INTO documentos (filename, gcs_uri, tipo, numero, ano, status_vigencia, assunto_resumo, tags)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """
                doc_id = await conn.fetchval(
                    doc_query,
                    filename,
                    gcs_uri,
                    metadata.get("tipo"),
                    metadata.get("numero"),
                    metadata.get("ano"),
                    metadata.get("status"),
                    metadata.get("assunto_resumo"),
                    metadata.get("tags", [])
                )
                
                logger.info(f"Inserted document ID: {doc_id}")

                # 2. Insert Chunks
                # Prepare data for executemany
                chunk_data = [
                    (doc_id, chunk["conteudo_texto"], str(chunk["embedding"])) # pgvector expects string representation for vector? or list? asyncpg+pgvector usually handles list if type map is set. 
                    # Assuming standard vector string format handling or raw list if driver supports it.
                    # With asyncpg, it often requires registering the type or passing as string. 
                    # Let's try passing as native list, if fails we fix. Or safer: string "[1.0, 2.0, ...]"
                ]
                
                # Safer: convert list to string for vector type if not registered
                formatted_chunks = []
                for chunk in chunks:
                   formatted_chunks.append((doc_id, chunk["conteudo_texto"], str(chunk["embedding"])))

                await conn.executemany(
                    "INSERT INTO chunks (documento_id, conteudo_texto, embedding) VALUES ($1, $2, $3::vector)",
                    formatted_chunks
                )
                
                logger.info(f"Inserted {len(chunks)} chunks for document {doc_id}")
                
        except Exception as e:
            logger.error(f"Error saving to DB: {e}")
            raise e
        finally:
            await conn.close()
