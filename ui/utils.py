from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from openai import OpenAI

from server.graph_utils import get_graph 

from graphrag.clustering import cluster_papers, cluster_authors, analyze_clusters
from graphrag.query import semantic_search
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@lru_cache(maxsize=1)
def load_graph() -> nx.DiGraph:
    return get_graph()


def get_graph_stats(G: nx.DiGraph) -> Tuple[int, int, int]:
    papers = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "paper")
    authors = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "author")
    topics  = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "topic")

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "papers": papers,
        "authors": authors,
        "topics": topics,
    }


def search_entities(
    G: nx.DiGraph,
    query: str,
    entity_type: str = "paper",
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    query_lower = query.lower().strip()
    if not query_lower:
        return results
    if entity_type == "paper":
        sims = semantic_search(G, query, top_k)
        for node, score, title, cluster in sims:
            results.append(
                {
                    "id": node,
                    "label": title,
                    "type": "paper",
                    "score": float(score),
                    "cluster": cluster,
                }
            )
    elif entity_type == "author":
        for node, data in G.nodes(data=True):
            if data.get("type") == "author":
                name = data.get("name", "")
                if name and query_lower in name.lower():
                    results.append(
                        {
                            "id": node,
                            "label": name,
                            "type": "author",
                            "score": 1.0,
                            "cluster": data.get("author_cluster")
                            or data.get("author_community")
                            or data.get("author_subcluster"),
                        }
                    )
        results = results[:top_k]
    elif entity_type == "topic":
        for node, data in G.nodes(data=True):
            if data.get("type") == "topic":
                category = data.get("category", "")
                if category and query_lower in category.lower():
                    results.append(
                        {
                            "id": node,
                            "label": category,
                            "type": "topic",
                            "score": 1.0,
                        }
                    )
        results = results[:top_k]
    return results


def get_entity_details(G: nx.DiGraph, node_id: Any) -> Dict[str, Any]:
    data = G.nodes[node_id].copy() 
    data.pop("embedding", None)
    return data


def get_relationships(G: nx.DiGraph, node_id: Any) -> Dict[str, List[Any]]:
    related: Dict[str, List[Any]] = {
        "authors": [],
        "papers": [],
        "topics": []
    }
    if node_id not in G.nodes:
        return related
    node_type = G.nodes[node_id].get("type")
    for neigh in G.neighbors(node_id):
        dtype = G.nodes[neigh].get("type")
        if dtype == "author":
            related["authors"].append(neigh)
        elif dtype == "paper":
            related["papers"].append(neigh)
        elif dtype == "topic":
            related["topics"].append(neigh)
    if node_type == "paper":
        for _, tgt, data in G.out_edges(node_id, data=True):
            if data.get("relationship") == "CITES":
                related["citations"].append(tgt)
    return related


def run_clustering(
    G: nx.DiGraph,
    entity_type: str = "paper",
    method: str = "hybrid",
    n_clusters: int = 10,
    dim_method: str = "pca",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    graph = G.copy()
    if entity_type == "paper":
        graph = cluster_papers(graph, n_clusters=n_clusters)
        cluster_field = "cluster"
    else:
        graph = cluster_authors(graph, method=method, n_clusters=n_clusters)
        if method == "kmeans":
            cluster_field = "author_cluster"
        elif method == "louvain":
            cluster_field = "author_community"
        else:
            cluster_field = "author_subcluster"
    ids: List[Any] = []
    labels: List[str] = []
    clusters: List[Any] = []
    embeddings: List[np.ndarray] = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != entity_type:
            continue
        if cluster_field not in data or "embedding" not in data:
            continue
        emb = data["embedding"]
        if isinstance(emb, str):
            vec = np.array([float(x) for x in emb.split(";")])
        else:
            vec = np.array(emb)
        embeddings.append(vec)
        ids.append(node)
        clusters.append(data.get(cluster_field))
        if entity_type == "paper":
            labels.append(data.get("title", str(node)))
        else:
            labels.append(data.get("name", str(node)))
    if not embeddings:
        return pd.DataFrame(), {}
    X = np.vstack(embeddings)
    if dim_method.lower() == "tsne":
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, max(2, len(X) - 1)),
            init="pca",
            learning_rate="auto",
        )
        coords = reducer.fit_transform(X)
    else:
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(X)
    df = pd.DataFrame(
        {
            "id": ids,
            "label": labels,
            "cluster": clusters,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )
    cluster_counts: Dict[str, int] = (
        df["cluster"].value_counts().apply(int).to_dict()
    )
    stats = {
        "num_clusters": len(cluster_counts),
        "cluster_sizes": cluster_counts,
    }
    return df, stats


def rag_answer(
    question: str,
    G: nx.DiGraph,
    top_k: int = 5,
    model: str = "openai",
    temperature: float = 0.7,
) -> Tuple[str, List[Any]]:
    from server.config import settings
    
    papers = semantic_search(G, question, top_k)
    contexts: List[str] = []
    ids: List[Any] = []
    for node, score, title, cluster in papers:
        ids.append(node)
        abstract = G.nodes[node].get("abstract", "")
        contexts.append(f"Title: {title}\nAbstract: {abstract}")
    context_str = "\n\n".join(contexts)

    if (
        model == "openai"
        and settings is not None
        and getattr(settings, "OPENAI_API_KEY", None)
    ):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.OPENAI_API_KEY)

            prompt = f"{context_str}\n\nQuestion: {question}\nAnswer:"

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=512,
            )
            answer = response.choices[0].message.content.strip()

        except Exception as e:
            answer = f"LLM call failed: {e}\n\nContext:\n{context_str}"
    else:
        answer = f"Context:\n{context_str}"
    return answer, ids



def reasoning_path(G: nx.DiGraph, start_paper: Any) -> Dict[str, Any]:
    path: Dict[str, Any] = {
        "paper": start_paper,
        "similar_papers": [],
        "authors": [],
    }
    if start_paper not in G.nodes:
        return path
    if G.nodes[start_paper].get("type") != "paper":
        return path
    cluster = G.nodes[start_paper].get("cluster")
    if cluster is None:
        return path
    similar: List[Any] = [
        n
        for n, d in G.nodes(data=True)
        if d.get("type") == "paper" and d.get("cluster") == cluster and n != start_paper
    ]
    path["similar_papers"] = similar[:5]
    authors: set[Any] = set()
    for paper in similar[:5]:
        for neigh in G.neighbors(paper):
            if G.nodes[neigh].get("type") == "author":
                authors.add(neigh)
    path["authors"] = list(authors)[:5]
    return path