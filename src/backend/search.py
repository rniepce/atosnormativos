import logging
import json
from typing import List, Optional
from src.utils.vertex import VertexAIClient
from src.utils.db import get_db_connection
from src.backend.models import SearchRequest, SearchResultItem

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.vertex_client = VertexAIClient()

    async def rewrite_query(self, original_query: str) -> str:
        """
        Uses Gemini 1.5 Flash to optimize the query for search.
        """
        prompt = f"""
        Você é um assistente especializada em busca jurídica. Rescreva a pergunta do usuário para otimizar a recuperação de documentos em uma base vetorial de atos normativos.
        Mantenha os termos jurídicos chave e remova conversas desnecessárias.
        Retorne APENAS a query reescrita.
        
        Pergunta original: "{original_query}"
        """
        rewritten = self.vertex_client.generate_text(prompt, model_name="gemini-2.0-flash-exp")
        return rewritten.strip()

    async def search(self, request: SearchRequest) -> List[SearchResultItem]:
        conn = await get_db_connection()
        try:
            # 1. Provide embedding for query
            # We use the rewritten query for embedding or original? Usually rewritten is better context, 
            # but sometimes simple keywords are better. Let's use rewritten if it's not too divergent, 
            # currently just using original + filters for simplicity or let's use the rewritten one.
            # Let's keep it simple: Embedding of the original query often captures intent well enough, 
            # but let's try to embed the original for now to avoid drift, 
            # use rewrite for keyword augmentation if we were doing valid hybrid full-text.
            
            # Actually, per requirements: "1. Rewrite Query -> 2. Busca Híbrida"
            rewritten_query = await self.rewrite_query(request.query)
            logger.info(f"Rewritten query: {rewritten_query}")

            embedding = self.vertex_client.get_query_embedding(rewritten_query)
            
            # 2. Build Query
            # Distance operator <=> is cosine distance
            # Using simple string formatting for filters (be careful with SQL injection if not using params, 
            # but here values are bound)
            
            where_clauses = []
            params = [str(embedding)] # $1 is embedding
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
                WHERE 1 - (c.embedding <=> $1::vector) > 0.5 -- Threshold
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
        Generates an answer using Gemini 1.5 Pro based on retrieved context.
        """
        if not context:
            return "Não encontrei normas relevantes para sua pergunta nos critérios selecionados."
            
        context_str = ""
        for item in context:
             context_str += f"""
             ---
             Norma: {item.tipo} {item.numero}/{item.ano} (Status: {item.status})
             Conteúdo: {item.chunk_text}
             ---
             """
             
        prompt = f"""
        Você é um assistente jurídico sênior do TJMG.
        Com base APENAS nos contextos abaixo, responda à pergunta do magistrado/servidor.
        
        Pergunta: "{query}"
        
        Contextos Recuperados:
        {context_str}
        
        Instruções:
        1. Cite a norma (Portaria/Resolução) e o artigo/parágrafo específico que fundamenta sua resposta.
        2. Se o contexto indicar que a norma está revogada, ALERTE o usuário de forma destacada.
        3. Se a resposta não estiver nos contextos, diga que não encontrou a informação.
        4. Seja direto, formal e fundamentado.
        """
        
        return self.vertex_client.generate_text(prompt, model_name="gemini-2.5-pro")
