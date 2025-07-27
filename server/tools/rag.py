from typing import Tuple, List, Dict, Any
import networkx as nx
from functools import lru_cache
from server import mcp
from server.config import settings
from server.graph_utils import get_graph
from graphrag.query import semantic_search
import openai

@lru_cache(maxsize=1)
def _load_graph() -> nx.DiGraph:
    """Cache and return the graph for RAG endpoints."""
    return get_graph()

@mcp.tool(
    name="rag/query",
    description="Answer a question using retrieved contexts and an optional LLM."
)
def rag_query(
    question: str,
    top_k: int = 5,
    provider: str = "openai",
    model: str = "gpt-4",
    temperature: float = 0.7
) -> Tuple[str, List[str]]:
    G = _load_graph()
    papers = semantic_search(G, question, top_k)
    contexts: List[str] = []
    ids: List[str] = []
    for node, _, title, _ in papers:
        ids.append(node)
        abstract = G.nodes[node].get("abstract", "")
        contexts.append(f"Title: {title}\\nAbstract: {abstract}")
    context_str = "\n\n".join(contexts)
    
    if (
        provider.lower() == "openai"
        and openai is not None
        and getattr(settings, "OPENAI_API_KEY", None)
    ):
        openai.api_key = settings.OPENAI_API_KEY 
        prompt = f"{context_str}\n\nQuestion: {question}\nAnswer:"
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=512
            )
            answer = resp.choices[0].message.content.strip()
        except Exception as e:
            answer = f"LLM call failed: {e}\n\nContext:\n{context_str}"
    else:
        answer = f"Context:\n{context_str}"
    
    return answer, ids

@mcp.tool(
    name="rag/context",
    description="Fetch metadata for a specific node."
)
def rag_context(node_id: str) -> Dict[str, Any]:
    G = _load_graph()
    if node_id not in G.nodes:
        raise ValueError(f"Node '{node_id}' not found.")
    data = dict(G.nodes[node_id])
    data.pop("embedding", None)
    return data
