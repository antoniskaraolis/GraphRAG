# mcp_server/endpoints/rag.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..models import RagRequest, RagResponse
from ..utils import get_graph
from graphrag.query import semantic_search
import openai
from ..config import settings
import logging

router = APIRouter()
logger = logging.getLogger("rag-endpoint")

@router.post("/query", response_model=RagResponse)
def rag_query(req: RagRequest):
    """Answer questions using RAG pipeline"""
    try:
        openai.api_key = settings.OPENAI_API_KEY
        
        G = get_graph()
        
        papers = semantic_search(G, req.question, top_k=3)
        
        context_parts = []
        sources = []
        for node, score, title, cluster in papers:
            context_parts.append(
                f"Title: {'title'}\n"
                f"Abstract: {G.nodes[node]['abstract'][:500]}"
            )
            sources.append(title)
        context = "\n\n".join(context_parts)[:req.max_context]
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a research assistant analyzing academic papers."},
                {"role": "user", "content": f"Question: {req.question}\nContext:\n{context}"}
            ],
            temperature=req.temperature
        )
        
        return RagResponse(
            answer=response.choices[0].message.content,
            sources=sources,
            context_length=len(context)
        )
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/context")
def get_context(node_id: str):
    """Get context for a specific entity"""
    G = get_graph()
    if node_id not in G.nodes:
        return JSONResponse(status_code=404, content={"error": "Node not found"})
        
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
                       if G.nodes[p].get('type') == 'paper'][:10]
        }
    
    return {"node": node_id, "context": context}
