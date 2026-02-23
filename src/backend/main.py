from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.backend.models import SearchRequest, SearchResponse
from src.backend.search import SearchService
from src.ingestion.extraction import extract_text_from_pdf
from src.ingestion.classification import DocumentClassifier
from src.ingestion.chunking import LegalTextSplitter
from src.ingestion.storage import DocumentStorage
from src.utils.vertex import VertexAIClient
import logging
import os
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
    """Upload and process a PDF file for RAG ingestion."""
    logger.info(f"Received upload: {file.filename}")
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 1. Extract text
        text = extract_text_from_pdf(tmp_path)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        
        # 2. Classify
        vertex_client = VertexAIClient()
        classifier = DocumentClassifier(vertex_client)
        metadata = classifier.classify_document(text)
        
        if not metadata:
            metadata = {
                "tipo": "Desconhecido", "numero": "0", "ano": 0,
                "status": "Indefinido", "assunto_resumo": "Falha na classificação", "tags": []
            }
        
        # 3. Chunk & Embed
        splitter = LegalTextSplitter()
        chunks = splitter.split_text(text, metadata)
        
        # 4. Store
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
        
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
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
        answer = await search_service.generate_answer(request.query, results, model=request.model, api_key=request.api_key)
        
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
