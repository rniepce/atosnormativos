from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.backend.models import SearchRequest, SearchResponse
from src.backend.search import SearchService, get_azure_client, AZURE_EMBEDDING_MODEL, AZURE_EMBEDDING_DIMENSIONS, _llm_generate
from src.ingestion.extraction import extract_text_from_pdf
from src.ingestion.storage import DocumentStorage
import logging
import os
import json
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TJMG Normativos RAG API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service instance (singleton-ish for simple app)
search_service = SearchService()

# API routes start with /api (except upload and search for legacy reasons)
@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF/DOCX file for RAG ingestion using Azure."""
    logger.info(f"Received upload: {file.filename}")
    
    allowed_ext = (".pdf", ".doc", ".docx")
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(status_code=400, detail=f"Supported formats: {', '.join(allowed_ext)}")
    
    try:
        # Save to temp file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 1. Extract text
        text = extract_text_from_pdf(tmp_path)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from file.")
        
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
{text[:10000]}

Responda APENAS com JSON válido:"""
        
        try:
            metadata_text = _llm_generate(classify_prompt)
            metadata_text = metadata_text.replace("```json", "").replace("```", "").strip()
            metadata = json.loads(metadata_text)
        except Exception as e:
            logger.warning(f"Classification failed: {e}, using defaults")
            metadata = {
                "tipo": "Desconhecido", "numero": "0", "ano": 0,
                "orgao": "Desconhecido", "status": "VIGENTE",
                "assunto_resumo": "Classificação pendente", "tags": []
            }
        
        # 3. Chunk text with legal separators
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\nCAPÍTULO", "\nSeção", "\nArt.", "\nParágrafo", "\n§", "\n\n", "\n", " "]
        )
        raw_chunks = splitter.split_text(text)
        
        # 4. Embed with Azure (same model as search)
        azure_client = get_azure_client()
        if azure_client is None:
            raise HTTPException(status_code=500, detail="Azure OpenAI client not configured")
        
        chunks = []
        # Process in batches of 100
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
        
        # Cleanup temp file
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "metadata": metadata,
            "chunks_created": len(chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    try:
        logger.info(f"Received search request: {request.query}")
        
        # 1. Search Logic
        results = await search_service.search(request)
        
        if not results:
            return SearchResponse(answer="Nenhum ato normativo encontrado com os critérios fornecidos.", sources=[])

        # 2. Answer Generation
        # (Could be parallelized or streamed in future)
        answer = await search_service.generate_answer(
            request.query, results, model=request.model,
            use_enriched=request.use_enriched_prompt
        )
        
        return SearchResponse(answer=answer, sources=results)

    except Exception as e:
        logger.error(f"Error processing search request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# IMPORTANT: Mount the React frontend AFTER all API routes
frontend_dist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")
else:
    logger.warning(f"Frontend dist folder not found at {frontend_dist_path}. Run 'npm run build' inside src/frontend.")

if __name__ == "__main__":
    import uvicorn
    # Use environment port or default 8000
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
