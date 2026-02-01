from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    filter_status: Optional[str] = Field(None, description="Filter by status (e.g., VIGENTE)")
    filter_tipo: Optional[str] = Field(None, description="Filter by document type")
    filter_ano: Optional[int] = Field(None, description="Filter by year")

class SearchResultItem(BaseModel):
    document_id: int
    filename: str
    tipo: Optional[str]
    numero: Optional[str]
    ano: Optional[int]
    status: Optional[str]
    chunk_text: str
    score: float

class SearchResponse(BaseModel):
    answer: str
    sources: List[SearchResultItem]
