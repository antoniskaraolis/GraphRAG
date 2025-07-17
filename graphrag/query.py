# graphrag/query.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd

def semantic_search(G, query, top_k=5):
    print(f"\n=== Semantic Search: '{query}' ===")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embed = model.encode([query])[0]

    similarities = []
    for node, data in G.nodes(data=True):
        if data.get('type') == 'paper' and 'embedding' in data:
            if isinstance(data['embedding'], str):
                emb = np.array([float(x) for x in data['embedding'].split(';')])
            else:
                emb = data['embedding']
            sim = cosine_similarity([query_embed], [emb])[0][0]
            similarities.append((node, sim, data['title'], data.get('cluster')))

    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"Top {top_k} results:")
    for i, (node, score, title, cluster) in enumerate(similarities[:top_k]):
        print(f"\n[{i+1}] Score: {score:.4f}, Cluster: {cluster}")
        print(f"Title: {title}")
        print(f"URL: {G.nodes[node].get('url', 'N/A')}")
        print(f"Abstract: {G.nodes[node].get('abstract', '')[:150]}...")

    return similarities[0][0] if similarities else None

def relationship_queries(G, paper_node):
    print("\n=== Relationship Queries ===")
    print(f"\nAuthors of paper: {G.nodes[paper_node]['title'][:50]}...")
    authors = [n for n in G.neighbors(paper_node)
               if G.nodes[n].get('type') == 'author']
    for author in authors[:5]:
        print(f"- {G.nodes[author]['name']}")

    if authors:
        author_node = authors[0]
        print(f"\nCollaborators of {G.nodes[author_node]['name']}:")
        collaborators = [n for n in G.neighbors(author_node)
                         if G.nodes[n].get('type') == 'author'
                         and G.edges[(author_node, n)].get('relationship') == 'COAUTHOR_WITH']
        for collab in collaborators[:5]:
            print(f"- {G.nodes[collab]['name']}")

    return author_node if authors else None

def rag_queries(G):
    print("\n=== RAG Queries: Trending Topics ===")
    paper_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'paper']
    topic_counts = {}

    for paper in paper_nodes:
        topics = [n for n in G.neighbors(paper)
                 if G.nodes[n].get('type') == 'topic']
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Research Topics:")
    for i, (topic_node, count) in enumerate(top_topics):
        topic_name = G.nodes[topic_node].get('category', 'Unknown')
        print(f"{i+1}. {topic_name}: {count} papers")
        for paper in G.neighbors(topic_node):
            if G.nodes[paper].get('type') == 'paper':
                print(f"   - {G.nodes[paper]['title'][:60]}...")
                break

def multi_hop_exploration(G, start_paper):
    print("\n=== Multi-hop Exploration ===")
    print(f"Starting from paper: {G.nodes[start_paper]['title'][:50]}...")
    cluster = G.nodes[start_paper].get('cluster')
    similar_papers = [n for n, d in G.nodes(data=True)
                     if d.get('type') == 'paper'
                     and d.get('cluster') == cluster
                     and n != start_paper]

    print(f"\nFound {len(similar_papers)} papers in the same cluster (Cluster {cluster})")
    authors = set()
    for paper in similar_papers[:5]:
        paper_authors = [n for n in G.neighbors(paper)
                        if G.nodes[n].get('type') == 'author']
        authors.update(paper_authors)

    print(f"\nAuthors working on similar research:")
    for author in list(authors)[:5]:
        print(f"- {G.nodes[author]['name']}")
        author_papers = [n for n in G.neighbors(author)
                        if G.nodes[n].get('type') == 'paper'
                        and n != start_paper
                        and n not in similar_papers]

        if author_papers:
            print(f"  Other papers by this author:")
            for paper in author_papers[:2]:
                print(f"  - {G.nodes[paper]['title'][:60]}...")

def analyze_clusters(G):
    print("\n=== Paper Cluster Analysis ===")
    paper_nodes = [n for n, d in G.nodes(data=True)
                  if d.get('type') == 'paper' and 'cluster' in d]

    cluster_data = []
    for node in paper_nodes:
        data = G.nodes[node]
        cluster_data.append({
            'cluster': data['cluster'],
            'title': data['title'],
            'year': data['year'],
            'abstract': data['abstract']
        })

    df = pd.DataFrame(cluster_data)
    print("\nCluster Distribution:")
    print(df['cluster'].value_counts().sort_index())

    largest_cluster = df['cluster'].value_counts().idxmax()
    print(f"\nLargest Cluster: {largest_cluster} ({len(df[df['cluster']==largest_cluster])} papers)")
    sample_titles = df[df['cluster']==largest_cluster]['title'].sample(3).tolist()
    print("\nSample Papers:")
    for title in sample_titles:
        print(f"- {title[:70]}...")

def visualize_collaboration_network(G, output_file="collaboration_network.png"):
    print("\nVisualizing collaboration network...")
    author_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'author']
    collaboration_edges = [(u, v) for u, v, d in G.edges(data=True)
                          if d.get('relationship') == 'COAUTHOR_WITH']

    if not author_nodes or not collaboration_edges:
        print("No collaboration data found!")
        return

    H = G.subgraph(author_nodes)
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(H, seed=42)
    nx.draw_networkx_nodes(H, pos, node_size=20, alpha=0.8)
    nx.draw_networkx_edges(H, pos, alpha=0.1)

    degrees = dict(H.degree())
    top_authors = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    for author, deg in top_authors:
        plt.text(pos[author][0], pos[author][1],
                 f"{G.nodes[author]['name'][:15]}",
                 fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7))

    plt.title("Author Collaboration Network")
    plt.axis('off')
    plt.savefig(output_file)
    print(f"Saved collaboration network to {output_file}")
