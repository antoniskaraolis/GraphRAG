# mcp_server/endpoints/rag.py
from fastmcp import APIRouter
from ..models import RagRequest, RagResponse
from ..utils import get_graph
from graphrag.query import semantic_search
import openai
from ..config import settings
import logging

router = APIRouter()
logger = logging.getLogger("rag-endpoint")

@router.endpoint("/query", response_model=RagResponse)
def rag_query(req: RagRequest):
    """Answer questions using RAG pipeline"""
    try:
        # Set API key
        openai.api_key = settings.OPENAI_API_KEY
        
        # Get graph
        G = get_graph()
        
        # 1. Retrieve relevant papers
        papers = semantic_search(G, req.question, top_k=3)
        
        # 2. Build context
        context_parts = []
        for paper in papers:
            context_parts.append(
                f"Title: {G.nodes[paper]['title']}\n"
                f"Abstract: {G.nodes[paper]['abstract'][:500]}"
            )
        context = "\n\n".join(context_parts)[:req.max_context]
        
        # 3. Call OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a research assistant analyzing academic papers."},
                {"role": "user", "content": f"Question: {req.question}\nContext:\n{context}"}
            ],
            temperature=req.temperature
        )
        
        return RagResponse(
            answer=response.choices[0].message.content,
            sources=[G.nodes[paper]['title'] for paper in papers],
            context_length=len(context)
        )
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        return {"error": str(e)}, 500

@router.endpoint("/context")
def get_context(node_id: str):
    """Get context for a specific entity"""
    G = get_graph()
    if node_id not in G.nodes:
        return {"error": "Node not found"}, 404
        
    node_data = G.nodes[node_id]
    context = {}
    
    if node_data.get('type') == 'paper':
        context = {
            "title": node_data.get('title'),
            "abstract": node_data.get('abstract'),
            "year": node_data.get('year'),
            "url": node_data.get('url')
        }
    elif node_data.get('type') == 'author':
        context = {
            "name": node_data.get('name'),
            "papers": [G.nodes[p]['title'] for p in G.neighbors(node_id) 
                       if G.nodes[p].get('type') == 'paper']
        }
    elif node_data.get('type') == 'topic':
        context = {
            "category": node_data.get('category'),
            "papers": [G.nodes[p]['title'] for p in G.neighbors(node_id) 
                       if G.nodes[p].get('type') == 'paper'][:10]  # Top 10 papers
        }
    
    return {"node": node_id, "context": context}