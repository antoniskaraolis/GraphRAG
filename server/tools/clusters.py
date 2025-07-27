import networkx as nx
from functools import lru_cache
from typing import Any, Dict, List
from server import mcp
from server.graph_utils import get_graph
from graphrag.clustering import cluster_papers, cluster_authors, analyze_clusters

@lru_cache(maxsize=1)
def _load_graph() -> nx.DiGraph:
    return get_graph()

@mcp.tool(
    name="clusters/list",
    description="Cluster entities (papers or authors) and return groupings."
)
def clusters_list(
    n_clusters: int = 10,
    entity_type: str = "paper"
) -> List[List[Any]]:
    G = _load_graph()
    if entity_type == "author":
        # hybrid method for author clustering
        G = cluster_authors(G, method="hybrid", n_clusters=n_clusters)
        clusters: Dict[Any, List[Any]] = {}
        for node, data in G.nodes(data=True):
            if data.get("type") == "author":
                cluster_id = data.get("author_subcluster")
                if cluster_id is None:
                    cluster_id = data.get("author_community")
                if cluster_id is not None:
                    clusters.setdefault(cluster_id, []).append(node)
        return list(clusters.values())
    else:
        # K-Means for papers clustering 
        G = cluster_papers(G, n_clusters=n_clusters)
        clusters: Dict[int, List[Any]] = {}
        for node, data in G.nodes(data=True):
            if data.get('type') == 'paper' and 'cluster' in data:
                clusters.setdefault(data['cluster'], []).append(node)
        return [members for members in clusters.values()]

@mcp.tool(
    name="clusters/analyze",
    description="Analyze clustering quality for papers or authors."
)
def clusters_analyze(
    n_clusters: int = 10,
    entity_type: str = "paper"
) -> Dict[str, Any]:
    G = _load_graph()
    if entity_type == "author":
        G = cluster_authors(G, method="hybrid", n_clusters=n_clusters)
        clusters: Dict[Any, List[Any]] = {}
        for node, data in G.nodes(data=True):
            if data.get("type") == "author":
                cluster_id = data.get("author_subcluster")
                if cluster_id is None:
                    cluster_id = data.get("author_community")
                if cluster_id is not None:
                    clusters.setdefault(cluster_id, []).append(node)
        num_clusters = len(clusters)
        cluster_sizes = {str(cid): len(members) for cid, members in clusters.items()}
        return {
            "num_clusters": num_clusters,
            "cluster_sizes": cluster_sizes,
            "method": "hybrid"
        }
    else:
        G = cluster_papers(G, n_clusters=n_clusters)
        return analyze_clusters(G)
