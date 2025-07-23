from typing import Tuple, List, Dict, Any
import openai
import networkx as nx
from functools import lru_cache
from server import mcp
from server.config import settings
from server.graph_utils import get_graph
from graphrag.query import semantic_search

# Configure OpenAI API key
openai.api_key = settings.OPENAI_API_KEY

@lru_cache(maxsize=1)
def _load_graph() -> nx.Graph:
    """Load graph once for RAG endpoints."""
    return get_graph()

@mcp.tool(name="rag/query", description="Answer question using GPT-4 with retrieved contexts.")
def rag_query(question: str, top_k: int = 5) -> Tuple[str, List[str]]:
    G = _load_graph()
    # Retrieve top-k papers via semantic search
    papers = semantic_search(G, question, top_k)
    contexts: List[str] = []
    ids: List[str] = []
    for node, _, title, _ in papers:
        ids.append(node)
        abstract = G.nodes[node].get("abstract", "")
        # Properly escape newline within the f‑string
        contexts.append(f"Title: {title}\\nAbstract: {abstract}")
    # Build prompt with contexts
    context_str = "\n\n".join(contexts)
    prompt = (
        f"{context_str}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    # Call OpenAI GPT-4
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512
    )
    answer = resp.choices[0].message.content.strip()
    return answer, ids

@mcp.tool(name="rag/context", description="Fetch metadata for a specific node.")
def rag_context(node_id: str) -> Dict[str, Any]:
    G = _load_graph()
    if node_id not in G.nodes:
        raise ValueError(f"Node '{node_id}' not found.")
    data = dict(G.nodes[node_id])
    # Remove or shorten embedding before returning
    data.pop("embedding", None)
    return data
