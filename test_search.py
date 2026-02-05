import asyncio
import os
from src.backend.search import SearchService
from src.backend.models import SearchRequest
from dotenv import load_dotenv

# Load env variables (for DB connection)
load_dotenv('/Users/rafaelpimentel/Downloads/atosnormativos/.env')

async def test_search():
    print("Initializing Search Service...")
    service = SearchService()
    
    query = "comite de gestao"
    print(f"\nSearching for: '{query}'")
    
    request = SearchRequest(query=query)
    results = await service.search(request)
    
    print(f"Found {len(results)} results.")
    
    if results:
        print("\nTop Result:")
        top = results[0]
        print(f"File: {top.filename}")
        print(f"Score: {top.score}")
        print(f"Text: {top.chunk_text[:200]}...")
        
        print("\nGenerated Answer:")
        answer = await service.generate_answer(query, results)
        print(answer[:500])
    
if __name__ == "__main__":
    asyncio.run(test_search())
