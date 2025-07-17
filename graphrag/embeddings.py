# graphrag/embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def add_embeddings(G):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Paper embeddings
    paper_nodes = [n for n, d in G.nodes(data=True) if d['type']=='paper']
    print(f"Embedding {len(paper_nodes)} papers...")

    for node in tqdm(paper_nodes):
        text = G.nodes[node]['title'] + " " + G.nodes[node]['abstract'][:500]
        embedding = model.encode(text, convert_to_numpy=True)
        G.nodes[node]['embedding'] = ';'.join(map(str, embedding))

    # Author embeddings
    author_nodes = [n for n, d in G.nodes(data=True) if d['type']=='author']
    print(f"Embedding {len(author_nodes)} authors...")

    for author in tqdm(author_nodes):
        papers = list(G.predecessors(author))
        embeddings = []
        for paper in papers:
            if 'embedding' in G.nodes[paper]:
                emb = np.array([float(x) for x in G.nodes[paper]['embedding'].split(';')])
                embeddings.append(emb)

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            G.nodes[author]['embedding'] = ';'.join(map(str, avg_embedding))
        elif 'name' in G.nodes[author]:
            embedding = model.encode(G.nodes[author]['name'], convert_to_numpy=True)
            G.nodes[author]['embedding'] = ';'.join(map(str, embedding))

    return G
