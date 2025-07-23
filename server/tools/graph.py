import networkx as nx
from functools import lru_cache
from typing import List, Dict, Any
from server import mcp
from server.graph_utils import get_graph, get_neighbors
from graphrag.query import semantic_search

@lru_cache(maxsize=1)
def _load_graph() -> nx.Graph:
    return get_graph()

@mcp.tool(name="graph/stats", description="Get total node and edge counts.")
def graph_stats() -> Dict[str, int]:
    G = _load_graph()
    return {"num_nodes": G.number_of_nodes(), "num_edges": G.number_of_edges()}

@mcp.tool(name="graph/search", description="Semantic search over paper embeddings.")
def graph_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    G = _load_graph()
    results = semantic_search(G, query, top_k)
    return [
        {"id": node, "score": score, "title": title, "cluster": cluster}
        for node, score, title, cluster in results
    ]

@mcp.tool(name="graph/neighbors", description="List neighbors of a given node.")
def graph_neighbors(node_id: str, relationship: str | None = None) -> List[str]:
    return get_neighbors(node_id, relationship)
