import networkx as nx
from functools import lru_cache
from typing import Any, Dict, List
from server import mcp
from server.graph_utils import get_graph
from graphrag.clustering import cluster_papers, analyze_clusters

@lru_cache(maxsize=1)
def _load_graph() -> nx.Graph:
    """Load graph once for clustering endpoints."""
    return get_graph()

@mcp.tool(name="clusters/list", description="Cluster papers and return groupings.")
def clusters_list(n_clusters: int = 10) -> List[List[Any]]:
    G = _load_graph()
    G = cluster_papers(G, n_clusters=n_clusters)
    clusters: Dict[int, List[Any]] = {}
    for node, data in G.nodes(data=True):
        if data.get('type') == 'paper' and 'cluster' in data:
            clusters.setdefault(data['cluster'], []).append(node)
    return list(clusters.values())

@mcp.tool(name="clusters/analyze", description="Analyze paper clustering quality.")
def clusters_analyze(n_clusters: int = 10) -> Dict[str, Any]:
    G = _load_graph()
    G = cluster_papers(G, n_clusters=n_clusters)
    return analyze_clusters(G)