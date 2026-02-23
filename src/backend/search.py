import logging
import os
from typing import List
from openai import AzureOpenAI
from src.utils.db import get_db_connection
from src.backend.models import SearchRequest, SearchResultItem

logger = logging.getLogger(__name__)

# ── Azure OpenAI Configuration ──────────────────────────────────
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://assistente-web-resource.cognitiveservices.azure.com/")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_LLM_MODEL = os.getenv("AZURE_LLM_MODEL", "gpt-4.1-mini")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
AZURE_EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1024"))

# Single Azure OpenAI client for both embeddings + chat
_AZURE_CLIENT = None


def get_azure_client(api_key: str = None):
    """Initialize an Azure OpenAI client. Uses per-request key if provided, otherwise env default."""
    if api_key:
        # Per-request client (not cached)
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )

    global _AZURE_CLIENT
    if _AZURE_CLIENT is None:
        env_key = AZURE_API_KEY
        if not env_key:
            logger.warning("AZURE_API_KEY not set, Azure AI disabled")
            return None
        try:
            _AZURE_CLIENT = AzureOpenAI(
                api_key=env_key,
                azure_endpoint=AZURE_ENDPOINT,
                api_version=AZURE_API_VERSION,
            )
            logger.info(f"Azure OpenAI client initialized (LLM: {AZURE_LLM_MODEL}, Embeddings: {AZURE_EMBEDDING_MODEL})")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    return _AZURE_CLIENT


def _llm_generate(prompt: str, model: str = None, api_key: str = None) -> str:
    """Generate text using Azure OpenAI (model selectable per request)."""
    client = get_azure_client(api_key)
    if client is None:
        raise RuntimeError("Azure OpenAI client not configured (missing API Key)")

    use_model = model or AZURE_LLM_MODEL
    response = client.chat.completions.create(
        model=use_model,
        messages=[
            {
                "role": "system",
                "content": "Você é um assistente jurídico inteligente especializado em atos normativos do Tribunal de Justiça de Minas Gerais (TJMG). Mantenha as respostas concisas e altamente baseadas nos dados fornecidos."
            },
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=4096,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


class SearchService:
    def __init__(self):
        self.client = get_azure_client()

    def rewrite_query(self, original_query: str, model: str = None, api_key: str = None) -> str:
        """Optionally rewrite query using LLM for better legal search."""
        try:
            prompt = f"""Reescreva a seguinte pergunta do usuário para otimizar a busca em um sistema de atos normativos jurídicos do TJMG.
Mantenha os termos técnicos jurídicos e adicione sinônimos relevantes.
Responda APENAS com a query reescrita, sem explicações.

Query original: {original_query}

Query otimizada:"""
            rewritten = _llm_generate(prompt, model=model, api_key=api_key)
            logger.info(f"Query rewritten: '{original_query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return original_query.strip()

    def _rerank_with_llm(self, query: str, results: List[SearchResultItem], model: str = None, api_key: str = None) -> List[SearchResultItem]:
        """Use LLM to rerank results by relevance to the query."""
        if len(results) <= 3:
            return results

        try:
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

            order_text = _llm_generate(prompt, model=model, api_key=api_key)

            order = []
            for num in order_text.replace(" ", "").split(","):
                try:
                    idx = int(num.strip("[]"))
                    if 0 <= idx < len(results):
                        order.append(idx)
                except ValueError:
                    continue

            if order:
                reranked = [results[i] for i in order if i < len(results)]
                remaining = [r for i, r in enumerate(results) if i not in order]
                reranked.extend(remaining)
                logger.info(f"Reranked {len(results)} results")
                return reranked

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")

        return results

    def _detect_recency_intent(self, query: str) -> bool:
        """Check if query implies a need for recent documents."""
        keywords = ["recente", "novo", "atual", "último", "2024", "2025", "hoje", "agora"]
        query_lower = query.lower()
        return any(k in query_lower for k in keywords)

    async def search(self, request: SearchRequest) -> List[SearchResultItem]:
        conn = await get_db_connection()

        prioritize_recency = request.prioritize_recency
        if not prioritize_recency and self._detect_recency_intent(request.query):
            prioritize_recency = True
            logger.info(f"Recency intent detected for query: '{request.query}'")

        try:
            # Optionally rewrite query for better search
            rewritten_query = self.rewrite_query(request.query, model=request.model, api_key=request.api_key)
            logger.info(f"Query: {rewritten_query}")

            # Generate embedding via Azure OpenAI
            embed_client = get_azure_client(request.api_key) if request.api_key else self.client
            if embed_client is None:
                raise RuntimeError("Azure OpenAI client not configured (missing API Key)")

            response = embed_client.embeddings.create(
                input=[rewritten_query],
                model=AZURE_EMBEDDING_MODEL,
                dimensions=AZURE_EMBEDDING_DIMENSIONS,
            )
            embedding = response.data[0].embedding
            embedding_str = str(embedding)

            # Initialize params with vector ($1)
            params = [embedding_str]

            # If hybrid search, $2 is the text query
            if request.use_hybrid_search:
                params.append(rewritten_query)

            # Build query with filters
            where_clauses = []

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

            if request.use_hybrid_search:
                recency_boost_sql = "0"
                if prioritize_recency:
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
                results = self._rerank_with_llm(request.query, results, model=request.model, api_key=request.api_key)

            return results[:10]

        finally:
            await conn.close()

    async def generate_answer(self, query: str, context: List[SearchResultItem], model: str = None, api_key: str = None) -> str:
        """Generate answer using Azure OpenAI (model selectable)."""
        if not context:
            return "Não encontrei normas relevantes para sua pergunta nos critérios selecionados."

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

            answer = _llm_generate(prompt, model=model, api_key=api_key)

            answer += "\n\n---\n**📚 Fontes consultadas:**\n"
            for item in context[:5]:
                answer += f"- {item.tipo} {item.numero}/{item.ano} ({item.filename})\n"

            return answer

        except Exception as e:
            logger.error(f"LLM error: {e}", exc_info=True)
            return self._fallback_answer(query, context)

    def _fallback_answer(self, query: str, context: List[SearchResultItem]) -> str:
        """Fallback when no LLM is available."""
        answer = f"**Resultados encontrados para:** '{query}'\n\n"
        answer += "_(LLM não configurado - exibindo trechos relevantes)_\n\n"

        for i, item in enumerate(context, 1):
            answer += f"**{i}. {item.tipo} {item.numero}/{item.ano}** (Relevância: {item.score:.2f})\n"
            answer += f"_{item.chunk_text[:400]}..._\n\n"

        return answer
