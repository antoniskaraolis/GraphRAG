# mcp_server/endpoints/graph.py
from fastmcp import APIRouter
from ..models import SearchRequest, SearchResult, StatsResponse
from ..utils import get_graph
from graphrag.query import semantic_search
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger("graph-endpoint")

@router.endpoint("/stats", response_model=StatsResponse)
def get_stats():
    """Get graph statistics"""
    G = get_graph()
    papers = sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'paper')
    authors = sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'author')
    topics = sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'topic')
    
    return StatsResponse(
        nodes=len(G.nodes),
        edges=len(G.edges),
        papers=papers,
        authors=authors,
        topics=topics
    )

@router.endpoint("/search", response_model=list[SearchResult])
def search_entities(req: SearchRequest):
    """Semantic search for papers"""
    G = get_graph()
    results = []
    for node, score, title, cluster in semantic_search(G, req.query, req.top_k):
        results.append(SearchResult(
            id=node,
            title=title,
            score=score,
            abstract=G.nodes[node].get('abstract', '')[:200],
            url=G.nodes[node].get('url', ''),
            cluster=cluster
        ))
    return results

@router.endpoint("/neighbors")
def get_neighbors(node_id: str, relationship: str = None):
    """Get neighbors of a node"""
    G = get_graph()
    neighbors = []
    for neighbor in G.neighbors(node_id):
        if relationship:
            edge_data = G.get_edge_data(node_id, neighbor)
            if edge_data and edge_data.get('relationship') == relationship:
                neighbors.append({
                    "id": neighbor,
                    "type": G.nodes[neighbor].get('type'),
                    "name": G.nodes[neighbor].get('name') or G.nodes[neighbor].get('title') or G.nodes[neighbor].get('category')
                })
        else:
            neighbors.append({
                "id": neighbor,
                "type": G.nodes[neighbor].get('type'),
                "name": G.nodes[neighbor].get('name') or G.nodes[neighbor].get('title') or G.nodes[neighbor].get('category')
            })
    return {"node": node_id, "neighbors": neighbors}