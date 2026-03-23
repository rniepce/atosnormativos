from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse
from src.backend.models import SearchRequest, SearchResponse
from src.backend.search import SearchService, get_active_llm_info, set_request_google_key, clear_request_google_key, _llm_generate
from src.ingestion.extraction import extract_text_from_pdf, extract_text_from_doc_docx
from src.ingestion.storage import DocumentStorage
from src.utils.db import init_pool, close_pool
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import os
import json
import tempfile

# ── Structured Logging (CESEC §3) ───────────────────────────────
LOG_FORMAT = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt='%Y-%m-%dT%H:%M:%S%z',
)
logger = logging.getLogger(__name__)

# ── Rate Limiting (CESEC §2.2) ──────────────────────────────────
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

# ── Security Configuration ──────────────────────────────────────
UPLOAD_API_KEY = os.getenv("UPLOAD_API_KEY", "")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(",")]
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

MAX_BODY_SIZE = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # bytes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB pool. Shutdown: close it."""
    await init_pool(min_size=2, max_size=10)
    logger.info("DB connection pool initialized")
    yield
    await close_pool()
    logger.info("DB connection pool closed")


app = FastAPI(title="TJMG Normativos RAG API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter

# Rate limit error handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    client_ip = request.client.host if request.client else "unknown"
    logger.warning(f"Rate limit exceeded | ip={client_ip} | path={request.url.path}")
    return JSONResponse(
        status_code=429,
        content={"detail": "Muitas requisições. Tente novamente em breve."},
    )

app.add_middleware(SlowAPIMiddleware)

# ── CORS — restricted origins (CESEC §5.2) ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ── Request Logging Middleware (CESEC §3) ────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"request_start | ip={client_ip} | method={request.method} | path={request.url.path}")
    response = await call_next(request)
    logger.info(f"request_end | ip={client_ip} | method={request.method} | path={request.url.path} | status={response.status_code}")
    return response

# Service instance
search_service = SearchService()


# ── Auth dependency for protected endpoints (CESEC §2) ──────────
async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key for protected endpoints. Skips if no key is configured."""
    if not UPLOAD_API_KEY:
        # No key configured — allow (dev mode)
        return
    if x_api_key != UPLOAD_API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API inválida ou ausente. Envie o header X-API-Key.")


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/model-info")
async def model_info():
    """Return the active LLM provider and model name."""
    return get_active_llm_info()

@app.get("/api/config-check")
async def config_check():
    """Return which env vars are configured (not their values). For debugging."""
    from src.backend.search import AZURE_API_KEY, AZURE_ENDPOINT, AZURE_EMBEDDING_MODEL, GOOGLE_API_KEY, LLM_PROVIDER
    return {
        "AZURE_API_KEY": bool(AZURE_API_KEY),
        "AZURE_ENDPOINT": bool(AZURE_ENDPOINT),
        "AZURE_EMBEDDING_MODEL": AZURE_EMBEDDING_MODEL,
        "AZURE_EMBEDDING_DIMENSIONS": int(os.getenv("AZURE_EMBEDDING_DIMENSIONS", "1024")),
        "GOOGLE_API_KEY": bool(GOOGLE_API_KEY),
        "LLM_PROVIDER": LLM_PROVIDER,
        "POSTGRES_HOST": bool(os.getenv("POSTGRES_HOST")),
        "DB_POOL": bool(True),  # if we reach here, pool is up
    }

@app.post("/upload")
@limiter.limit("10/minute")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    _auth=Depends(verify_api_key),
):
    """Upload and process a PDF/DOCX file for RAG ingestion using Azure."""
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"upload_start | ip={client_ip} | filename={file.filename}")

    allowed_ext = (".pdf", ".doc", ".docx")
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(status_code=400, detail=f"Formatos suportados: {', '.join(allowed_ext)}")

    try:
        # Read file content
        content = await file.read()

        # ── File size validation (CESEC §6) ──────────────────────
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(status_code=400, detail=f"Arquivo excede o limite de {MAX_UPLOAD_SIZE_MB}MB.")

        # ── MIME type validation via magic bytes + extensão ──────────
        ext = Path(file.filename).suffix.lower() if file.filename else ""
        magic_bytes = content[:8]
        is_pdf  = magic_bytes[:4] == b"%PDF"
        is_docx = magic_bytes[:4] == b"PK\x03\x04"  # ZIP (DOCX/XLSX)
        is_doc  = magic_bytes[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # OLE2 (DOC)
        if not (is_pdf or is_docx or is_doc) or ext not in (".pdf", ".docx", ".doc"):
            logger.warning(f"upload_rejected | ip={client_ip} | filename={file.filename}")
            raise HTTPException(status_code=400, detail="Tipo de arquivo inválido. Envie PDF, DOC ou DOCX.")

        # Save to temp file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # 1. Extract text — choose extractor based on file extension
            lower_suffix = suffix.lower()
            if lower_suffix == ".pdf":
                text = extract_text_from_pdf(tmp_path)
            elif lower_suffix in (".doc", ".docx"):
                text = extract_text_from_doc_docx(tmp_path)
            else:
                text = None

            if not text:
                raise HTTPException(status_code=400, detail="Não foi possível extrair texto do arquivo.")

            # 2. Classify using Azure LLM
            classify_prompt = f"""Você é um classificador jurídico especializado em atos normativos do TJMG.
Analise o texto abaixo e extraia os metadados em formato JSON:
{{
  "tipo": "Portaria|Resolução|Provimento|Portaria Conjunta|Aviso|Ordem de Serviço|Outro",
  "numero": "string (ex: 1234)",
  "ano": int (ex: 2023),
  "orgao": "string (ex: Corregedoria, Presidência)",
  "status": "VIGENTE|REVOGADO",
  "assunto_resumo": "Resumo conciso",
  "tags": ["tag1", "tag2"]
}}

Texto (primeiros 10.000 caracteres):
<user_query>{text[:10000]}</user_query>

Responda APENAS com JSON válido:"""

            try:
                metadata_text = _llm_generate(classify_prompt)
                metadata_text = metadata_text.replace("```json", "").replace("```", "").strip()
                metadata = json.loads(metadata_text)
            except Exception as e:
                logger.warning(f"classification_failed | filename={file.filename} | error={e}")
                metadata = {
                    "tipo": "Desconhecido", "numero": "0", "ano": 0,
                    "orgao": "Desconhecido", "status": "VIGENTE",
                    "assunto_resumo": "Classificação pendente", "tags": []
                }

            # 3. Chunk text with legal separators
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200,
                separators=["\nCAPÍTULO", "\nSeção", "\nArt.", "\nParágrafo", "\n§", "\n\n", "\n", " "]
            )
            raw_chunks = splitter.split_text(text)

            # 4. Embed with Azure (same model as search)
            from src.backend.search import get_azure_client, AZURE_EMBEDDING_MODEL, AZURE_EMBEDDING_DIMENSIONS
            azure_client = get_azure_client()
            if azure_client is None:
                raise HTTPException(status_code=500, detail="Serviço de embeddings indisponível.")

            chunks = []
            for i in range(0, len(raw_chunks), 100):
                batch = raw_chunks[i:i+100]
                response = azure_client.embeddings.create(
                    input=batch, model=AZURE_EMBEDDING_MODEL, dimensions=AZURE_EMBEDDING_DIMENSIONS
                )
                for j, emb_data in enumerate(response.data):
                    chunks.append({
                        "conteudo_texto": batch[j],
                        "embedding": emb_data.embedding
                    })

            # 5. Store in database
            storage = DocumentStorage()
            await storage.save_document_and_chunks(
                filename=file.filename,
                gcs_uri="",
                metadata=metadata,
                chunks=chunks
            )

            logger.info(f"upload_success | ip={client_ip} | filename={file.filename} | chunks={len(chunks)}")

            return {
                "status": "success",
                "filename": file.filename,
                "metadata": metadata,
                "chunks_created": len(chunks)
            }
        finally:
            # Always clean up temp file, even on exception
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        # ── Sanitized error response (CESEC §6) ─────────────────
        logger.error(f"upload_error | ip={client_ip} | filename={file.filename} | error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno ao processar o arquivo. Tente novamente.")

@app.post("/search", response_model=SearchResponse)
@limiter.limit(RATE_LIMIT)
async def search_endpoint(request: Request, body: SearchRequest):
    client_ip = request.client.host if request.client else "unknown"
    try:
        logger.info(f"search_start | ip={client_ip} | query={body.query[:80]}")

        # Set per-request Google API key if provided by frontend
        if body.google_api_key:
            set_request_google_key(body.google_api_key)

        try:
            # 1. Search Logic
            results = await search_service.search(body)

            if not results:
                return SearchResponse(answer="Nenhum ato normativo encontrado com os critérios fornecidos.", sources=[])

            # 2. Answer Generation
            answer = await search_service.generate_answer(body.query, results)

            return SearchResponse(answer=answer, sources=results)
        finally:
            # Always clear the per-request key
            clear_request_google_key()

    except Exception as e:
        # ── Sanitized error response (CESEC §6) ─────────────────
        logger.error(f"search_error | ip={client_ip} | query={body.query[:80]} | error={e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno na busca. Tente novamente.")

# IMPORTANT: Mount the React frontend AFTER all API routes
frontend_dist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")
else:
    logger.warning(f"Frontend dist folder not found at {frontend_dist_path}. Run 'npm run build' inside src/frontend.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
