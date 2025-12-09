from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    
class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    
class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    
class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_created: int