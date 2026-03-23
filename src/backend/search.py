import logging
import os
import time
import contextvars
from typing import List
import httpx
from openai import AzureOpenAI, OpenAI
from src.utils.db import get_db_connection, release_db_connection
from src.backend.models import SearchRequest, SearchResultItem

logger = logging.getLogger(__name__)

# ── LLM Provider Configuration ──────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")  # "google", "ollama", or "azure"

# ── Ollama (Local) Configuration ────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")

# ── Google AI Studio (Gemini API) Configuration ─────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemma-3-12b-it")
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# ── Azure AI Foundry Configuration ──────────────────────────────
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "https://assistente-web-resource.cognitiveservices.azure.com/")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_LLM_MODEL = os.getenv("AZURE_LLM_MODEL", "gpt-5.4-mini")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-large")
AZURE_EMBEDDING_DIMENSIONS = int(os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1024"))

# Cached clients
_AZURE_CLIENT = None
_GOOGLE_CLIENT = None
_OLLAMA_AVAILABLE = None
_OLLAMA_CHECK_TIME = 0.0  # Timestamp of last Ollama check
_OLLAMA_TTL = 300  # Re-check Ollama availability every 5 minutes

# Per-request API key using ContextVar (safe for concurrent async requests)
_REQUEST_GOOGLE_KEY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    '_REQUEST_GOOGLE_KEY', default=None
)


def set_request_google_key(key: str):
    """Set a per-request Google API key (from frontend)."""
    _REQUEST_GOOGLE_KEY.set(key)


def clear_request_google_key():
    """Clear the per-request Google API key."""
    _REQUEST_GOOGLE_KEY.set(None)


def get_azure_client():
    """Initialize a single Azure OpenAI client from environment variables."""
    global _AZURE_CLIENT
    if _AZURE_CLIENT is None:
        if not AZURE_API_KEY:
            logger.warning("AZURE_API_KEY not set, Azure AI disabled")
            return None
        try:
            _AZURE_CLIENT = AzureOpenAI(
                api_key=AZURE_API_KEY,
                azure_endpoint=AZURE_ENDPOINT,
                api_version=AZURE_API_VERSION,
            )
            logger.info(f"Azure OpenAI client initialized (LLM: {AZURE_LLM_MODEL}, Embeddings: {AZURE_EMBEDDING_MODEL})")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    return _AZURE_CLIENT


def get_google_client():
    """Initialize Google AI Studio client. Prefers per-request key from frontend."""
    global _GOOGLE_CLIENT
    # If a per-request key is provided, create a temporary client
    per_request_key = _REQUEST_GOOGLE_KEY.get()
    if per_request_key:
        return OpenAI(
            api_key=per_request_key,
            base_url=GOOGLE_BASE_URL,
        )
    # Otherwise use the cached env-var client
    if _GOOGLE_CLIENT is None:
        if not GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not set, Google AI Studio disabled")
            return None
        try:
            _GOOGLE_CLIENT = OpenAI(
                api_key=GOOGLE_API_KEY,
                base_url=GOOGLE_BASE_URL,
            )
            logger.info(f"✅ Google AI Studio client initialized (Model: {GOOGLE_MODEL})")
        except Exception as e:
            logger.error(f"Failed to initialize Google AI Studio client: {e}")
    return _GOOGLE_CLIENT


def check_ollama_available() -> bool:
    """Check if Ollama is running and the model is available (re-checks every 5 min)."""
    global _OLLAMA_AVAILABLE, _OLLAMA_CHECK_TIME
    now = time.time()
    if _OLLAMA_AVAILABLE is not None and (now - _OLLAMA_CHECK_TIME) < _OLLAMA_TTL:
        return _OLLAMA_AVAILABLE
    _OLLAMA_CHECK_TIME = now
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            model_base = OLLAMA_MODEL.split(":")[0]
            _OLLAMA_AVAILABLE = any(model_base in m for m in models)
            if _OLLAMA_AVAILABLE:
                logger.info(f"✅ Ollama available with model: {OLLAMA_MODEL}")
            else:
                logger.warning(f"⚠️ Ollama running but model '{OLLAMA_MODEL}' not found. Available: {models}")
            return _OLLAMA_AVAILABLE
    except Exception as e:
        logger.warning(f"⚠️ Ollama not reachable at {OLLAMA_BASE_URL}: {e}")
    _OLLAMA_AVAILABLE = False
    return False


def get_active_llm_info() -> dict:
    """Return info about the currently active LLM provider and model."""
    if LLM_PROVIDER == "google" and (GOOGLE_API_KEY or _REQUEST_GOOGLE_KEY.get()):
        return {"provider": "google", "model": GOOGLE_MODEL, "label": f"Gemma 3 12B (Google AI)"}
    elif LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
        return {"provider": "google", "model": GOOGLE_MODEL, "label": f"Gemma 3 12B (chave necessária)", "needs_key": True}
    elif LLM_PROVIDER == "ollama" and check_ollama_available():
        return {"provider": "ollama", "model": OLLAMA_MODEL, "label": f"Gemma 3 12B (Local)"}
    elif get_azure_client() is not None:
        return {"provider": "azure", "model": AZURE_LLM_MODEL, "label": f"{AZURE_LLM_MODEL} (Azure AI)"}
    else:
        return {"provider": "none", "model": "none", "label": "Nenhum LLM configurado"}


def _ollama_generate(prompt: str, system_prompt: str = None) -> str:
    """Generate text using Ollama (local Gemma 3 12B)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 2048,
                }
            },
            timeout=120.0,  # Generous timeout for local inference
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except httpx.TimeoutException:
        raise RuntimeError(f"Ollama timeout — model '{OLLAMA_MODEL}' may be loading or too slow")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def _google_generate(prompt: str, system_prompt: str = None) -> str:
    """Generate text using Google AI Studio (Gemini API with OpenAI compatibility)."""
    client = get_google_client()
    if client is None:
        raise RuntimeError("Google AI Studio client not configured (missing GOOGLE_API_KEY)")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=GOOGLE_MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def _azure_generate(prompt: str, system_prompt: str = None) -> str:
    """Generate text using Azure OpenAI."""
    client = get_azure_client()
    if client is None:
        raise RuntimeError("Azure OpenAI client not configured (missing AZURE_API_KEY)")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=AZURE_LLM_MODEL,
        messages=messages,
        max_completion_tokens=2048,
    )
    return response.choices[0].message.content.strip()


SYSTEM_PROMPT = "Você é um assistente jurídico inteligente especializado em atos normativos do Tribunal de Justiça de Minas Gerais (TJMG). Mantenha as respostas concisas e altamente baseadas nos dados fornecidos."


def _llm_generate(prompt: str) -> str:
    """Route LLM generation to the configured provider with automatic fallback.
    
    Fallback chain: google → ollama → azure (or your primary first)
    """
    providers = {
        "google": (lambda: get_google_client() is not None, _google_generate, "Google AI Studio"),
        "ollama": (lambda: check_ollama_available(), _ollama_generate, "Ollama"),
        "azure": (lambda: get_azure_client() is not None, _azure_generate, "Azure"),
    }

    if LLM_PROVIDER not in providers:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    # Try primary provider first
    check_fn, gen_fn, name = providers[LLM_PROVIDER]
    if check_fn():
        try:
            return gen_fn(prompt, system_prompt=SYSTEM_PROMPT)
        except Exception as e:
            logger.warning(f"{name} failed: {e}")

    # Try fallbacks in order
    for fallback_key, (check_fn, gen_fn, name) in providers.items():
        if fallback_key == LLM_PROVIDER:
            continue
        if check_fn():
            try:
                logger.info(f"Falling back to {name}")
                return gen_fn(prompt, system_prompt=SYSTEM_PROMPT)
            except Exception as e:
                logger.warning(f"{name} fallback also failed: {e}")

    raise RuntimeError("No LLM available (all providers failed)")


class SearchService:
    def __init__(self):
        self.client = get_azure_client()

    def rewrite_query(self, original_query: str, use_enriched: bool = True) -> str:
        """Rewrite query with TJMG domain vocabulary for better vector search."""
        if not use_enriched:
            return original_query.strip()
        try:
            prompt = f"""Você receberá uma pergunta de um servidor ou magistrado do TJMG que quer encontrar atos normativos relevantes.

Sua tarefa é EXPANDIR e REFORMULAR a query para maximizar os resultados de busca vetorial no banco de atos normativos do TJMG.

REGRAS:
1. Mantenha a intenção original da pergunta
2. Adicione sinônimos e termos técnicos jurídicos do TJMG que ampliem a busca
3. Expanda siglas (ex: "CEJUSC" → "CEJUSC Centro Judiciário de Solução de Conflitos")
4. Inclua termos correlatos (ex: "férias" → "férias licença afastamento recesso servidor magistrado")
5. Inclua os tipos de ato normativo mais prováveis (ex: "Resolução Portaria Provimento")
6. NÃO adicione explicações — responda APENAS com a query expandida
7. A query expandida deve ter entre 15 e 50 palavras

EXEMPLOS:
- "férias" → "férias de magistrado juiz desembargador servidor licença afastamento recesso forense Resolução Portaria"
- "teletrabalho" → "teletrabalho trabalho remoto home office regime híbrido servidor magistrado produtividade Resolução Portaria"
- "plantão" → "plantão judiciário noturno recesso habeas corpus urgência Portaria Conjunta Resolução escala"
- "concurso" → "concurso público remoção promoção seleção vaga magistrado servidor cargo provimento Resolução Edital"

Query original:
<user_query>{original_query}</user_query>

Query expandida:"""
            rewritten = _llm_generate(prompt)
            logger.info(f"Query rewritten: '{original_query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return original_query.strip()

    def _rerank_with_llm(self, query: str, results: List[SearchResultItem]) -> List[SearchResultItem]:
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

PERGUNTA:
<user_query>{query}</user_query>

DOCUMENTOS:
{docs_text}

ORDEM DE RELEVÂNCIA (números separados por vírgula):"""

            order_text = _llm_generate(prompt)

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
            rewritten_query = self.rewrite_query(request.query)
            logger.info(f"Query (enriched=True): {rewritten_query}")

            # Generate embedding via Azure OpenAI (embeddings always use Azure)
            if self.client is None:
                raise RuntimeError("Azure OpenAI client not configured (missing AZURE_API_KEY) — needed for embeddings")

            response = self.client.embeddings.create(
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
                results = self._rerank_with_llm(request.query, results)

            return results[:10]

        finally:
            await release_db_connection(conn)

    async def generate_answer(self, query: str, context: List[SearchResultItem]) -> str:
        """Generate answer using the active LLM provider."""
        if not context:
            return "Não encontrei normas relevantes para sua pergunta nos critérios selecionados."

        context_text = ""
        for i, item in enumerate(context, 1):
            context_text += f"\n--- Documento {i}: {item.tipo} {item.numero}/{item.ano} ({item.filename}) ---\n"
            context_text += f"{item.chunk_text}\n"

        # Always use enriched TJMG prompt (model selection is handled by _llm_generate)

        try:
            prompt = f"""Com base EXCLUSIVAMENTE nos documentos abaixo, responda à pergunta do usuário.

DOCUMENTOS ENCONTRADOS:
{context_text}

PERGUNTA DO USUÁRIO:
<user_query>{query}</user_query>

INSTRUÇÕES PARA A RESPOSTA:
1. Cite SEMPRE o tipo, número e ano dos atos normativos relevantes (ex: "Resolução nº 973/2021")
2. Se um ato revogou ou alterou outro, explique a cadeia normativa
3. Diferencie entre atos VIGENTES e REVOGADOS — priorize os vigentes
4. Indique o órgão emissor quando relevante (Presidência, Corregedoria, Órgão Especial)
5. Se a pergunta não puder ser respondida com os documentos encontrados, diga claramente e sugira termos de busca alternativos
6. Responda em português brasileiro, de forma clara e objetiva
7. Use formatação Markdown para melhor legibilidade (negritos, listas)
8. Ao final, se houver atos que complementem o tema, sugira ao usuário buscar por eles

RESPOSTA:"""

            answer = _llm_generate(prompt)

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
        answer += "_(LLM não disponível — exibindo trechos relevantes)_\n\n"

        for i, item in enumerate(context, 1):
            answer += f"**{i}. {item.tipo} {item.numero}/{item.ano}** (Relevância: {item.score:.2f})\n"
            answer += f"_{item.chunk_text[:400]}..._\n\n"

        return answer
