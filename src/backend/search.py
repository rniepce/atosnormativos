import logging
import os
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from src.utils.db import get_db_connection
from src.backend.models import SearchRequest, SearchResultItem

logger = logging.getLogger(__name__)

# Global model cache
_EMBEDDING_MODEL = None
_GEMINI_MODEL = None

def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info("Loading embedding model (BAAI/bge-large-en-v1.5, 1024-dim)...")
        try:
            _EMBEDDING_MODEL = SentenceTransformer('BAAI/bge-large-en-v1.5', local_files_only=True)
        except Exception as e:
            logger.warning(f"Could not load local model, trying online: {e}")
            _EMBEDDING_MODEL = SentenceTransformer('BAAI/bge-large-en-v1.5')
    return _EMBEDDING_MODEL

def get_gemini_model():
    global _GEMINI_MODEL
    if _GEMINI_MODEL is None:
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                # Try different model names (Gemini 2.5 Pro, then 2.0 Flash as fallback)
                model_names = ['gemini-3-flash-preview', 'gemini-2.5-pro-preview-06-05', 'gemini-2.0-flash', 'gemini-1.5-flash']
                for model_name in model_names:
                    try:
                        _GEMINI_MODEL = genai.GenerativeModel(model_name)
                        logger.info(f"Gemini model '{model_name}' initialized successfully")
                        break
                    except Exception as e:
                        logger.warning(f"Could not load {model_name}: {e}")
                        continue
                if _GEMINI_MODEL is None:
                    logger.error("No Gemini model could be loaded")
            else:
                logger.warning("GEMINI_API_KEY not set, LLM answers disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
    return _GEMINI_MODEL

class SearchService:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.gemini_model = get_gemini_model()

    async def rewrite_query(self, original_query: str) -> str:
        """Optionally rewrite query using Gemini for better legal search."""
        if not self.gemini_model:
            return original_query.strip()
        
        try:
            prompt = f"""Reescreva a seguinte pergunta do usuário para otimizar a busca em um sistema de atos normativos jurídicos do TJMG.
Mantenha os termos técnicos jurídicos e adicione sinônimos relevantes.
Responda APENAS com a query reescrita, sem explicações.

Query original: {original_query}

Query otimizada:"""
            response = self.gemini_model.generate_content(prompt)
            rewritten = response.text.strip()
            logger.info(f"Query rewritten: '{original_query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return original_query.strip()

    async def _rerank_with_llm(self, query: str, results: List[SearchResultItem]) -> List[SearchResultItem]:
        """Use LLM to rerank results by relevance to the query."""
        if not self.gemini_model or len(results) <= 3:
            return results
        
        try:
            # Build context for reranking
            docs_text = ""
            for i, item in enumerate(results):
                status_marker = "✓ VIGENTE" if item.status == "VIGENTE" else "✗ REVOGADO"
                docs_text += f"\n[{i}] {item.tipo} {item.numero}/{item.ano} ({status_marker})\n{item.chunk_text[:300]}...\n"
            
            prompt = f"""Analise a relevância dos seguintes trechos de atos normativos para a pergunta do usuário.
Retorne APENAS os números dos documentos mais relevantes, ordenados do mais ao menos relevante, separados por vírgula.
Considere: (1) relevância semântica, (2) status de vigência (prefira VIGENTE), (3) especificidade.

PERGUNTA: {query}

DOCUMENTOS:
{docs_text}

ORDEM DE RELEVÂNCIA (números separados por vírgula):"""

            response = self.gemini_model.generate_content(prompt)
            order_text = response.text.strip()
            
            # Parse the order
            order = []
            for num in order_text.replace(" ", "").split(","):
                try:
                    idx = int(num.strip("[]"))
                    if 0 <= idx < len(results):
                        order.append(idx)
                except ValueError:
                    continue
            
            # Reorder results
            if order:
                reranked = [results[i] for i in order if i < len(results)]
                # Add any missing results at the end
                remaining = [r for i, r in enumerate(results) if i not in order]
                reranked.extend(remaining)
                logger.info(f"Reranked {len(results)} results")
                return reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
        
        return results

    async def search(self, request: SearchRequest) -> List[SearchResultItem]:
        conn = await get_db_connection()
        try:
            # Optionally rewrite query for better search
            rewritten_query = await self.rewrite_query(request.query)
            logger.info(f"Query: {rewritten_query}")

            # Generate embedding
            embedding = self.embedding_model.encode([rewritten_query])[0]
            embedding_list = embedding.tolist()
            
            # Build query with filters
            where_clauses = []
            params = [str(embedding_list), rewritten_query]  # $1 = vector, $2 = text query
            param_idx = 3
            
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
            
            # Hybrid search: vector + BM25 keyword
            # vigente_boost: +0.15 for VIGENTE documents
            if request.use_hybrid_search:
                query_sql = f"""
                    WITH vector_search AS (
                        SELECT 
                            c.id,
                            c.documento_id, 
                            d.filename, 
                            d.tipo, 
                            d.numero, 
                            d.ano,
                            d.orgao,
                            d.status_vigencia, 
                            c.conteudo_texto,
                            1 - (c.embedding <=> $1::vector) as vector_score
                        FROM chunks c
                        JOIN documentos d ON c.documento_id = d.id
                        WHERE 1 - (c.embedding <=> $1::vector) > 0.25
                        {where_sql}
                        ORDER BY c.embedding <=> $1::vector ASC
                        LIMIT 30
                    ),
                    keyword_search AS (
                        SELECT 
                            c.id,
                            ts_rank_cd(to_tsvector('portuguese', c.conteudo_texto), plainto_tsquery('portuguese', $2)) as keyword_score
                        FROM chunks c
                        WHERE to_tsvector('portuguese', c.conteudo_texto) @@ plainto_tsquery('portuguese', $2)
                    )
                    SELECT 
                        v.*,
                        COALESCE(k.keyword_score, 0) as keyword_score,
                        (0.7 * v.vector_score + 0.3 * COALESCE(k.keyword_score, 0) + 
                         CASE WHEN v.status_vigencia = 'VIGENTE' THEN 0.15 ELSE 0 END) as combined_score
                    FROM vector_search v
                    LEFT JOIN keyword_search k ON v.id = k.id
                    ORDER BY combined_score DESC
                    LIMIT 20
                """
            else:
                # Simple vector search with vigente boost
                vigente_boost = "CASE WHEN d.status_vigencia = 'VIGENTE' THEN 0.15 ELSE 0 END" if request.prioritize_vigente else "0"
                query_sql = f"""
                    SELECT 
                        c.documento_id, 
                        d.filename, 
                        d.tipo, 
                        d.numero, 
                        d.ano,
                        d.orgao,
                        d.status_vigencia, 
                        c.conteudo_texto,
                        (1 - (c.embedding <=> $1::vector) + {vigente_boost}) as combined_score
                    FROM chunks c
                    JOIN documentos d ON c.documento_id = d.id
                    WHERE 1 - (c.embedding <=> $1::vector) > 0.25
                    {where_sql}
                    ORDER BY combined_score DESC
                    LIMIT 20
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
                    orgao=row.get("orgao"),
                    status=row["status_vigencia"],
                    chunk_text=row["conteudo_texto"],
                    score=float(row["combined_score"])
                ))
            
            # Rerank with LLM if enabled
            if request.use_reranking and len(results) > 3:
                results = await self._rerank_with_llm(request.query, results)
            
            # Return top 10 after reranking
            return results[:10]
            
        finally:
            await conn.close()

    async def generate_answer(self, query: str, context: List[SearchResultItem]) -> str:
        """Generate answer using Gemini based on retrieved context."""
        if not context:
            return "Não encontrei normas relevantes para sua pergunta nos critérios selecionados."
        
        # Build context for LLM
        context_text = ""
        for i, item in enumerate(context, 1):
            context_text += f"\n--- Documento {i}: {item.tipo} {item.numero}/{item.ano} ({item.filename}) ---\n"
            context_text += f"{item.chunk_text}\n"
        
        # If Gemini is available, use it
        if self.gemini_model:
            try:
                prompt = f"""Você é um assistente jurídico especializado em atos normativos do TJMG (Tribunal de Justiça de Minas Gerais).

Com base nos documentos abaixo, responda à pergunta do usuário de forma clara, objetiva e fundamentada, citando os atos normativos relevantes.

DOCUMENTOS ENCONTRADOS:
{context_text}

PERGUNTA DO USUÁRIO: {query}

INSTRUÇÕES:
- Responda em português brasileiro
- Cite os números e anos das portarias/resoluções quando relevante
- Se não houver informação suficiente, indique isso claramente
- Seja conciso mas completo

RESPOSTA:"""

                response = self.gemini_model.generate_content(prompt)
                answer = response.text
                
                # Add sources footer
                answer += "\n\n---\n**📚 Fontes consultadas:**\n"
                for item in context[:5]:
                    answer += f"- {item.tipo} {item.numero}/{item.ano} ({item.filename})\n"
                
                return answer
                
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                return self._fallback_answer(query, context)
        else:
            return self._fallback_answer(query, context)
    
    def _fallback_answer(self, query: str, context: List[SearchResultItem]) -> str:
        """Fallback when Gemini is not available."""
        answer = f"**Resultados encontrados para:** '{query}'\n\n"
        answer += "_(LLM não configurado - exibindo trechos relevantes)_\n\n"
        
        for i, item in enumerate(context, 1):
            answer += f"**{i}. {item.tipo} {item.numero}/{item.ano}** (Relevância: {item.score:.2f})\n"
            answer += f"_{item.chunk_text[:400]}..._\n\n"
            
        return answer
