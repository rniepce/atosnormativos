import logging
import os
from typing import List
from openai import AsyncOpenAI
from src.utils.db import get_db_connection
from src.backend.models import SearchRequest, SearchResultItem

logger = logging.getLogger(__name__)

# Global model/client caches
_OPENAI_CLIENT = None
_ANTHROPIC_CLIENT = None
_AMAZONIA_CLIENT = None

def get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                _OPENAI_CLIENT = AsyncOpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OPENAI_API_KEY not set, OpenAI embeddings disabled")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    return _OPENAI_CLIENT

def get_anthropic_client():
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                _ANTHROPIC_CLIENT = anthropic.AsyncAnthropic(api_key=api_key)
                logger.info("Anthropic client initialized successfully")
            else:
                logger.warning("ANTHROPIC_API_KEY not set, Anthropic answers disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
    return _ANTHROPIC_CLIENT

def get_amazonia_client():
    """Initialize the Amazônia IA client (OpenAI-compatible)."""
    global _AMAZONIA_CLIENT
    if _AMAZONIA_CLIENT is None:
        api_key = os.getenv("AMAZONIA_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                _AMAZONIA_CLIENT = OpenAI(
                    api_key=api_key,
                    base_url="https://amazonia-a.amazoniaia.com.br/v1"
                )
                logger.info("Amazônia IA client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Amazônia IA client: {e}")
        else:
            logger.warning("AMAZONIA_API_KEY not set, Amazônia IA answers disabled")
    return _AMAZONIA_CLIENT


async def _llm_generate(prompt: str, provider: str = "anthropic") -> str:
    """
    Abstraction layer: generate text from either Anthropic or Amazônia IA.
    Returns the generated text, or raises on failure.
    """
    if provider == "amazonia":
        client = get_amazonia_client()
        if client is None:
            raise RuntimeError("Amazônia IA client not configured (missing AMAZONIA_API_KEY)")
        resp = client.chat.completions.create(
            model="rodrigomalossi/amazonia-a",
            messages=[
                {"role": "system", "content": "Você é Amazônia-a, um assistente jurídico conciso e prestativo."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            top_p=0.9,
        )
        return resp.choices[0].message.content.strip()
    else:
        # Default: Anthropic
        client = get_anthropic_client()
        if client is None:
            raise RuntimeError("Anthropic client not configured (missing ANTHROPIC_API_KEY)")
        
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            temperature=0.4,
            system="Você é um assistente jurídico inteligente especializado em atos normativos do Tribunal de Justiça de Minas Gerais (TJMG). Mantenha as respostas concisas e altamente baseadas nos dados fornecidos.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()


class SearchService:
    def __init__(self):
        self.openai_client = get_openai_client()
        # Lazy-init both LLM backends (they cache globally)
        get_anthropic_client()
        get_amazonia_client()

    async def rewrite_query(self, original_query: str, provider: str = "anthropic") -> str:
        """Optionally rewrite query using LLM for better legal search."""
        try:
            prompt = f"""Reescreva a seguinte pergunta do usuário para otimizar a busca em um sistema de atos normativos jurídicos do TJMG.
Mantenha os termos técnicos jurídicos e adicione sinônimos relevantes.
Responda APENAS com a query reescrita, sem explicações.

Query original: {original_query}

Query otimizada:"""
            rewritten = await _llm_generate(prompt, provider)
            logger.info(f"Query rewritten ({provider}): '{original_query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed ({provider}): {e}")
            return original_query.strip()

    async def _rerank_with_llm(self, query: str, results: List[SearchResultItem], provider: str = "anthropic") -> List[SearchResultItem]:
        """Use LLM to rerank results by relevance to the query."""
        if len(results) <= 3:
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

            order_text = await _llm_generate(prompt, provider)
            
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
                logger.info(f"Reranked {len(results)} results ({provider})")
                return reranked
            
        except Exception as e:
            logger.warning(f"Reranking failed ({provider}): {e}")
        
        return results

    def _detect_recency_intent(self, query: str) -> bool:
        """Check if query implies a need for recent documents."""
        keywords = ["recente", "novo", "atual", "último", "2024", "2025", "hoje", "agora"]
        query_lower = query.lower()
        return any(k in query_lower for k in keywords)

    async def search(self, request: SearchRequest) -> List[SearchResultItem]:
        conn = await get_db_connection()
        provider = request.llm_provider
        
        # Detect recency intent if not explicitly set
        prioritize_recency = request.prioritize_recency
        if not prioritize_recency and self._detect_recency_intent(request.query):
            prioritize_recency = True
            logger.info(f"Recency intent detected for query: '{request.query}'")

        try:
            # Optionally rewrite query for better search
            rewritten_query = await self.rewrite_query(request.query, provider)
            logger.info(f"Query: {rewritten_query}")

            # Generate embedding
            if self.openai_client is None:
                raise RuntimeError("OpenAI client not configured (missing OPENAI_API_KEY)")
            
            response = await self.openai_client.embeddings.create(
                input=[rewritten_query],
                model="text-embedding-3-large",
                dimensions=1024
            )
            embedding = response.data[0].embedding
            embedding_str = str(embedding)
            
            # Initialize params with vector ($1)
            params = [embedding_str]
            
            # If hybrid search, $2 is the text query. Otherwise $2 is next param.
            if request.use_hybrid_search:
                params.append(rewritten_query)

            # Build query with filters
            where_clauses = []
            where_clauses = []
            
            # Helper to add param and get its placeholder index
            def add_param(value):
                params.append(value)
                return len(params)

            if request.filter_status:
                idx = add_param(request.filter_status)
                where_clauses.append(f"d.status_vigencia = ${idx}")
            
            if request.filter_tipo:
                idx = add_param(request.filter_tipo)
                where_clauses.append(f"d.tipo = ${idx}")
                
            if request.filter_ano:
                idx = add_param(request.filter_ano)
                where_clauses.append(f"d.ano = ${idx}")

            where_sql = " AND ".join(where_clauses)
            if where_sql:
                where_sql = f"AND {where_sql}"
            
            # Hybrid search: vector + BM25 keyword
            # vigente_boost: +0.15 for VIGENTE documents
            if request.use_hybrid_search:
                # Recency boost logic
                recency_boost_sql = "0"
                if prioritize_recency:
                    # +0.25 for 2025, +0.15 for 2024
                    recency_boost_sql = """
                        CASE 
                            WHEN d.ano = 2025 THEN 0.25 
                            WHEN d.ano = 2024 THEN 0.15 
                            ELSE 0 
                        END
                    """

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
                            1 - (c.embedding <=> $1::vector) as vector_score,
                            {recency_boost_sql} as recency_score
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
                         CASE WHEN v.status_vigencia = 'VIGENTE' THEN 0.15 ELSE 0 END +
                         v.recency_score) as combined_score
                    FROM vector_search v
                    LEFT JOIN keyword_search k ON v.id = k.id
                    ORDER BY combined_score DESC
                    LIMIT 20
                """
            else:
                # Simple vector search with vigente boost
                # Simple vector search with similar logic
                recency_boost_sql = "0"
                if prioritize_recency:
                    recency_boost_sql = "CASE WHEN d.ano = 2025 THEN 0.25 WHEN d.ano = 2024 THEN 0.15 ELSE 0 END"

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
                        (1 - (c.embedding <=> $1::vector) + {vigente_boost} + {recency_boost_sql}) as combined_score
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
                results = await self._rerank_with_llm(request.query, results, provider)
            
            # Return top 10 after reranking
            return results[:10]
            
        finally:
            await conn.close()

    async def generate_answer(self, query: str, context: List[SearchResultItem], provider: str = "anthropic") -> str:
        """Generate answer using the selected LLM based on retrieved context."""
        if not context:
            return "Não encontrei normas relevantes para sua pergunta nos critérios selecionados."
        
        # Build context for LLM
        context_text = ""
        for i, item in enumerate(context, 1):
            context_text += f"\n--- Documento {i}: {item.tipo} {item.numero}/{item.ano} ({item.filename}) ---\n"
            context_text += f"{item.chunk_text}\n"
        
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

            answer = await _llm_generate(prompt, provider)
            
            # Add sources footer
            answer += "\n\n---\n**📚 Fontes consultadas:**\n"
            for item in context[:5]:
                answer += f"- {item.tipo} {item.numero}/{item.ano} ({item.filename})\n"
            
            return answer
            
        except Exception as e:
            logger.error(f"LLM error ({provider}): {e}", exc_info=True)
            return self._fallback_answer(query, context)
    
    def _fallback_answer(self, query: str, context: List[SearchResultItem]) -> str:
        """Fallback when no LLM is available."""
        answer = f"**Resultados encontrados para:** '{query}'\n\n"
        answer += "_(LLM não configurado - exibindo trechos relevantes)_\n\n"
        
        for i, item in enumerate(context, 1):
            answer += f"**{i}. {item.tipo} {item.numero}/{item.ano}** (Relevância: {item.score:.2f})\n"
            answer += f"_{item.chunk_text[:400]}..._\n\n"
            
        return answer
