import logging
import json
import torch
from typing import List, Optional
# from src.utils.vertex import VertexAIClient  # REMOVED
from sentence_transformers import SentenceTransformer
from src.utils.db import get_db_connection
from src.backend.models import SearchRequest, SearchResultItem

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading on every request if service is re-instantiated
_EMBEDDING_MODEL = None

def get_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info("Loading usage-time embedding model (all-MiniLM-L6-v2)...")
        try:
            # Try offline first/fallback
            _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
        except Exception as e:
            logger.warning(f"Could not load local model, trying online: {e}")
            _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBEDDING_MODEL

class SearchService:
    def __init__(self):
        # self.vertex_client = VertexAIClient() # REMOVED
        self.model = get_model() # Load local model

    async def rewrite_query(self, original_query: str) -> str:
        """
        Mock rewrite: just return original query since we have no LLM.
        """
        # rewritten = self.vertex_client.generate_text(...)
        return original_query.strip()

    async def search(self, request: SearchRequest) -> List[SearchResultItem]:
        conn = await get_db_connection()
        try:
            # 1. Rewrite Query (Mocked)
            rewritten_query = await self.rewrite_query(request.query)
            logger.info(f"Query: {rewritten_query}")

            # 2. Generate Embedding (Local)
            # embedding = self.vertex_client.get_query_embedding(rewritten_query)
            embedding = self.model.encode([rewritten_query])[0]
            embedding_list = embedding.tolist()
            
            # 3. Build Query
            where_clauses = []
            params = [str(embedding_list)] # $1 is embedding (passed as string for pgvector)
            param_idx = 2
            
            if request.filter_status:
                where_clauses.append(f"d.status_vigencia = ${param_idx}")
                params.append(request.filter_status)
                param_idx += 1
            
            if request.filter_tipo:
                where_clauses.append(f"d.tipo = ${param_idx}")
                params.append(request.filter_tipo)
                param_idx += 1
                
            if request.filter_ano:
                where_clauses.append(f"d.ano = ${param_idx}")
                params.append(request.filter_ano)
                param_idx += 1

            where_sql = " AND ".join(where_clauses)
            if where_sql:
                where_sql = f"AND {where_sql}"
            
            # Use 384-dim vector casting if needed, or just vector
            query_sql = f"""
                SELECT 
                    c.documento_id, 
                    d.filename, 
                    d.tipo, 
                    d.numero, 
                    d.ano, 
                    d.status_vigencia, 
                    c.conteudo_texto,
                    1 - (c.embedding <=> $1::vector) as similarity
                FROM chunks c
                JOIN documentos d ON c.documento_id = d.id
                WHERE 1 - (c.embedding <=> $1::vector) > 0.3 -- Threshold lowered for local model
                {where_sql}
                ORDER BY c.embedding <=> $1::vector ASC
                LIMIT 10
            """
            
            rows = await conn.fetch(query_sql, *params)
            
            results = []
            for row in rows:
                results.append(SearchResultItem(
                    document_id=row["documento_id"],
                    filename=row["filename"],
                    tipo=row["tipo"],
                    numero=row["numero"],
                    ano=row["ano"],
                    status=row["status_vigencia"],
                    chunk_text=row["conteudo_texto"],
                    score=row["similarity"]
                ))
                
            return results
            
        finally:
            await conn.close()

    async def generate_answer(self, query: str, context: List[SearchResultItem]) -> str:
        """
        Generates a static answer concatenating context since we have no LLM.
        """
        if not context:
            return "Não encontrei normas relevantes para sua pergunta nos critérios selecionados."
            
        # Mock Answer Generation
        answer = f"**Resultados encontrados para:** '{query}'\n\n"
        answer += "Como não temos conexão com LLM (GCP), exibo abaixo os trechos relevantes encontrados:\n\n"
        
        for i, item in enumerate(context, 1):
            answer += f"**{i}. {item.tipo} {item.numero}/{item.ano}** (Semelhança: {item.score:.2f})\n"
            answer += f"_{item.chunk_text[:300]}..._\n"
            answer += f"[Ver documento completo: {item.filename}]\n\n"
            
        return answer
