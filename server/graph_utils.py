import networkx as nx
from functools import lru_cache
from server.config import settings

@lru_cache(maxsize=1)
def load_graph() -> nx.Graph:
    """Load and cache the GraphML file once."""
    return nx.read_graphml(settings.GRAPH_PATH)


def get_graph() -> nx.Graph:
    """Return the cached graph instance."""
    return load_graph()


def get_neighbors(node_id: str, relationship: str | None = None) -> list[str]:
    """Get neighbors, optionally filtering by edge relationship."""
    G = get_graph()
    if node_id not in G:
        raise KeyError(f"Node '{node_id}' not found in graph.")
    if relationship is not None:
        return [nbr for nbr, attrs in G[node_id].items() if attrs.get("relationship") == relationship]
    return list(G.neighbors(node_id))