# mcp_server/models.py
from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    id: str
    title: str
    score: float
    abstract: str
    url: str
    cluster: Optional[int] = None

class RagRequest(BaseModel):
    question: str
    temperature: float = 0.3
    max_context: int = 2000

class RagResponse(BaseModel):
    answer: str
    sources: List[str]
    context_length: int

class ClusterInfo(BaseModel):
    cluster_id: int
    paper_count: int
    sample_titles: List[str]

class StatsResponse(BaseModel):
    nodes: int
    edges: int
    papers: int
    authors: int
    topics: int