# graphrag/graph.py
import pandas as pd
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

def build_graph(csv_files):
    try:
        papers = pd.read_csv(csv_files['papers'])
        authors = pd.read_csv(csv_files['authors'])
        topics = pd.read_csv(csv_files['topics'])
        authored_edges = pd.read_csv(csv_files['authored_edges'])
        topic_edges = pd.read_csv(csv_files['topic_edges'])
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return None

    G = nx.DiGraph()

    # Nodes
    for _, row in tqdm(papers.iterrows(), total=len(papers)):
        year = row['year'] if not pd.isna(row['year']) else None
        G.add_node(
            f"paper_{row['id']}",
            type="paper",
            title=row['title'],
            abstract=row['abstract'],
            year=year,
            url=row['url'],
            version_count=row['version_count']
        )

    for _, row in tqdm(authors.iterrows(), total=len(authors)):
        G.add_node(f"author_{row['id']}", type="author", name=row['id'])

    for _, row in tqdm(topics.iterrows(), total=len(topics)):
        G.add_node(f"topic_{row['id']}", type="topic", category=row['id'])

    # Edges
    for _, row in tqdm(authored_edges.iterrows(), total=len(authored_edges)):
        G.add_edge(
            f"paper_{row['paper_id']}",
            f"author_{row['author_id']}",
            relationship="AUTHORED_BY"
        )

    for _, row in tqdm(topic_edges.iterrows(), total=len(topic_edges)):
        G.add_edge(
            f"paper_{row['paper_id']}",
            f"topic_{row['topic_id']}",
            relationship="BELONGS_TO"
        )

    # Collaborations
    author_papers = defaultdict(list)
    for _, row in authored_edges.iterrows():
        author_papers[row['author_id']].append(row['paper_id'])

    for author, papers_list in tqdm(author_papers.items()):
        coauthors = set()
        for paper in papers_list:
            paper_authors = authored_edges[
                authored_edges['paper_id'] == paper
            ]['author_id'].tolist()
            coauthors.update(paper_authors)

        coauthors.discard(author)

        for coauthor in coauthors:
            if not G.has_edge(f"author_{author}", f"author_{coauthor}"):
                G.add_edge(
                    f"author_{author}",
                    f"author_{coauthor}",
                    relationship="COAUTHOR_WITH"
                )

    return G
