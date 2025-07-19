# graphrag/clustering.py
import numpy as np
from sklearn.cluster import KMeans
import community.community_louvain as community_louvain
from sklearn.metrics import silhouette_score

def cluster_papers(G, n_clusters=10):
    paper_nodes = [n for n, d in G.nodes(data=True)
                  if d.get('type') == 'paper' and 'embedding' in d]

    if not paper_nodes:
        print("No papers with embeddings found!")
        return G

    embeddings = []
    for node in paper_nodes:
        emb_str = G.nodes[node]['embedding']
        embeddings.append(np.array([float(x) for x in emb_str.split(';')]))

    n_clusters = min(n_clusters, len(paper_nodes)//5, 10)
    if n_clusters < 2:
        print("Not enough papers for clustering")
        return G

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)

    for i, node in enumerate(paper_nodes):
        G.nodes[node]['cluster'] = int(kmeans.labels_[i])

    print(f"Created {n_clusters} paper clusters")
    return G

def cluster_authors(G, method='hybrid', n_clusters=10):
    author_nodes = [n for n, d in G.nodes(data=True)
                  if d.get('type') == 'author' and 'embedding' in d]

    if not author_nodes:
        print("No authors with embeddings found!")
        return G

    if method == 'kmeans':
        embeddings = []
        for node in author_nodes:
            emb_str = G.nodes[node]['embedding']
            embeddings.append(np.array([float(x) for x in emb_str.split(';')]))

        n_clusters = min(n_clusters, len(author_nodes)//5, 20)
        if n_clusters < 2:
            print("Not enough authors for clustering")
            return G

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)

        for i, node in enumerate(author_nodes):
            G.nodes[node]['author_cluster'] = int(kmeans.labels_[i])

        print(f"Created {n_clusters} author clusters using K-Means")

    elif method == 'louvain':
        collab_graph = G.edge_subgraph(
            [(u, v) for u, v, d in G.edges(data=True)
            if d.get('relationship') == 'COAUTHOR_WITH']
        ).to_undirected()

        partition = community_louvain.best_partition(collab_graph)
        for node, comm_id in partition.items():
            if G.nodes[node].get('type') == 'author':
                G.nodes[node]['author_community'] = int(comm_id)

        print(f"Created {max(partition.values())+1} author communities using Louvain")

    elif method == 'hybrid':
        collab_graph = G.edge_subgraph(
            [(u, v) for u, v, d in G.edges(data=True)
            if d.get('relationship') == 'COAUTHOR_WITH']
        ).to_undirected()

        partition = community_louvain.best_partition(collab_graph)
        for comm_id in set(partition.values()):
            comm_authors = [n for n in partition if partition[n] == comm_id
                          and G.nodes[n].get('type') == 'author']

            if len(comm_authors) < 5:
                continue

            embeddings = []
            for node in comm_authors:
                emb_str = G.nodes[node]['embedding']
                embeddings.append(np.array([float(x) for x in emb_str.split(';')]))

            if len(embeddings) > 5:
                n_subclusters = min(5, len(comm_authors)//2)
                kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
                kmeans.fit(embeddings)

                for i, node in enumerate(comm_authors):
                    G.nodes[node]['author_subcluster'] = int(kmeans.labels_[i])

        print(f"Created hybrid clusters for {len(set(partition.values()))} communities")

    return G

def analyze_clusters(G):
    """
    Analyze quality and statistics of paper clusters.
    Returns:
        dict with number of clusters, cluster sizes, and silhouette score (if possible).
    """
    # Find paper nodes with cluster assignment
    paper_nodes = [n for n, d in G.nodes(data=True)
                   if d.get('type') == 'paper' and 'cluster' in d and 'embedding' in d]
    if not paper_nodes:
        return {
            "error": "No clustered papers with embeddings found.",
            "num_clusters": 0,
            "cluster_sizes": {},
            "silhouette": None
        }

    clusters = {}
    embeddings = []
    labels = []
    for node in paper_nodes:
        cid = G.nodes[node]['cluster']
        clusters.setdefault(cid, []).append(node)
        emb_str = G.nodes[node]['embedding']
        embeddings.append(np.array([float(x) for x in emb_str.split(';')]))
        labels.append(cid)

    num_clusters = len(clusters)
    cluster_sizes = {str(cid): len(nodes) for cid, nodes in clusters.items()}

    # Compute silhouette score if there are >1 clusters and >1 paper per cluster
    silhouette = None
    if num_clusters > 1 and all(len(nodes) > 1 for nodes in clusters.values()):
        try:
            silhouette = silhouette_score(embeddings, labels)
        except Exception as e:
            silhouette = f"Could not compute: {e}"

    return {
        "num_clusters": num_clusters,
        "cluster_sizes": cluster_sizes,
        "silhouette": silhouette
    }
