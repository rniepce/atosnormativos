import sys
print("DEBUG: Script started", flush=True)
import asyncio
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Fix python path
sys.path.append(os.getcwd())

# Load env
load_dotenv(Path(__file__).parent / ".env")

from src.backend.models import SearchRequest
import src.backend.search as search_module
from src.backend.search import SearchService

print(f"DEBUG: Loaded search module from: {search_module.__file__}")

async def main():
    print("Initializing Search Service...")
    service = SearchService()
    
    queries = [
        "qual o ato mais recente?",
        "atos de 2024",
        "atos de 2025"
    ]

    for q in queries:
        print(f"\n\n>>> Testing Query: '{q}'")
        req = SearchRequest(
            query=q,
            use_hybrid_search=True,
            use_reranking=False # Disable reranking to see raw retrieval first
        )
        
        try:
            results = await service.search(req)
            print(f"Found {len(results)} results:")
            for i, res in enumerate(results):
                print(f"{i+1}. [{res.ano}] {res.tipo} {res.numero} (Score: {res.score:.3f}) - File: {res.filename}")
                if i >= 9: break 
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
