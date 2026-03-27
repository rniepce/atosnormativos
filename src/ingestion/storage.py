import logging
import asyncpg
import os
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
                # Coerce types to avoid asyncpg type mismatch errors
                try:
                    ano = int(metadata.get("ano") or 0)
                except (ValueError, TypeError):
                    ano = 0
                numero = str(metadata.get("numero") or "0")
                tags = [str(t) for t in (metadata.get("tags") or [])]

                doc_query = """
                    INSERT INTO documentos (filename, gcs_uri, tipo, numero, ano, orgao, status_vigencia, assunto_resumo, tags)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """
                doc_id = await conn.fetchval(
                    doc_query,
                    filename,
                    gcs_uri,
                    metadata.get("tipo"),
                    numero,
                    ano,
                    metadata.get("orgao"),
                    metadata.get("status"),
                    metadata.get("assunto_resumo"),
                    tags,
                )
                
                logger.info(f"Inserted document ID: {doc_id}")

                # 2. Insert Chunks
                
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
