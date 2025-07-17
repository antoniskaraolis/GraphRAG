# mcp_server/utils.py
import networkx as nx
from .config import settings
import logging

logger = logging.getLogger("mcp-server")

# Global graph instance
G = None

def load_graph():
    """Load and prepare the knowledge graph"""
    global G
    try:
        logger.info(f"Loading graph from {settings.GRAPH_PATH}")
        G = nx.read_graphml(settings.GRAPH_PATH)
        
        # Convert embeddings
        for node, data in G.nodes(data=True):
            if 'embedding' in data and isinstance(data['embedding'], str):
                data['embedding'] = [float(x) for x in data['embedding'].split(';')]
                
        logger.info(f"Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges")
        return True
    except Exception as e:
        logger.error(f"Graph loading failed: {str(e)}")
        return False

def get_graph():
    """Get the loaded graph instance"""
    if G is None:
        load_graph()
    return G

def get_neighbors(node_id: str, relationship: str = None):
    """Get neighbors of a node with optional relationship filter"""
    if G is None:
        return []
    
    neighbors = []
    for neighbor in G.neighbors(node_id):
        if relationship:
            edge_data = G.get_edge_data(node_id, neighbor)
            if edge_data and edge_data.get('relationship') == relationship:
                neighbors.append(neighbor)
        else:
            neighbors.append(neighbor)
    return neighbors