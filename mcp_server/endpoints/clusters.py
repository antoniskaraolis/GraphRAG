# mcp_server/endpoints/clusters.py
from fastmcp import APIRouter
from ..models import ClusterInfo
from ..utils import get_graph
import logging

router = APIRouter()
logger = logging.getLogger("clusters-endpoint")

@router.endpoint("/list", response_model=list[ClusterInfo])
def list_clusters():
    """List all paper clusters"""
    G = get_graph()
    clusters = {}
    for node, data in G.nodes(data=True):
        if data.get('type') == 'paper' and 'cluster' in data:
            cluster_id = data['cluster']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(data['title'])
    
    return [
        ClusterInfo(
            cluster_id=cluster_id,
            paper_count=len(titles),
            sample_titles=titles[:3]
        )
        for cluster_id, titles in clusters.items()
    ]

@router.endpoint("/analyze")
def analyze_clusters():
    """Analyze cluster quality"""
    G = get_graph()
    try:
        from graphrag.clustering import analyze_clusters
        return analyze_clusters(G)
    except ImportError:
        return {"error": "Cluster analysis not available"}, 501
    except Exception as e:
        logger.error(f"Cluster analysis failed: {str(e)}")
        return {"error": str(e)}, 500
