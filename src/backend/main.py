from fastapi import FastAPI, HTTPException
from src.backend.models import SearchRequest, SearchResponse
from src.backend.search import SearchService
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TJMG Normativos RAG API", version="1.0.0")

# Service instance (singleton-ish for simple app)
search_service = SearchService()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    try:
        logger.info(f"Received search request: {request.query}")
        
        # 1. Search Logic
        results = await search_service.search(request)
        
        if not results:
            return SearchResponse(answer="Nenhum atorm normativo encontrado com os critérios fornecidos.", sources=[])

        # 2. Answer Generation
        # (Could be parallelized or streamed in future)
        answer = await search_service.generate_answer(request.query, results)
        
        return SearchResponse(answer=answer, sources=results)

    except Exception as e:
        logger.error(f"Error processing search request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use environment port or default 8000
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
