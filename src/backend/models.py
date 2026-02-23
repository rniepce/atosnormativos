from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    filter_status: Optional[str] = Field(None, description="Filter by status (e.g., VIGENTE)")
    filter_tipo: Optional[str] = Field(None, description="Filter by document type")
    filter_ano: Optional[int] = Field(None, description="Filter by year")
    prioritize_vigente: bool = Field(True, description="Boost VIGENTE documents in ranking")
    use_hybrid_search: bool = Field(True, description="Combine vector + keyword search")
    use_reranking: bool = Field(True, description="Use LLM to rerank results")
    prioritize_recency: bool = Field(False, description="Boost recent documents (2024-2025) in ranking")
    model: Optional[str] = Field(None, description="Azure LLM deployment name (e.g., gpt-4.1-mini, gpt-5.2-chat)")
    api_key: Optional[str] = Field(None, description="Azure OpenAI API key")

class SearchResultItem(BaseModel):
    document_id: int
    filename: str
    tipo: Optional[str]
    numero: Optional[str]
    ano: Optional[int]
    orgao: Optional[str]
    status: Optional[str]
    chunk_text: str
    score: float

class SearchResponse(BaseModel):
    answer: str
    sources: List[SearchResultItem]
