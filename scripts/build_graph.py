#!/usr/bin/env python3
# scripts/build_graph.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
from graphrag.data_processing import prepare_dataset, process_papers
from graphrag.graph import build_graph
from graphrag.embeddings import add_embeddings
from graphrag.clustering import cluster_papers, cluster_authors

TARGETS = {
    "Computer Science": 0.125,
    "Economics": 0.125,
    "Electrical Engineering and Systems Science": 0.125,
    "Mathematics": 0.125,
    "Physics": 0.125,
    "Quantitative Biology": 0.125,
    "Quantitative Finance": 0.125,
    "Statistics": 0.125
}
TOTAL_PAPERS = 300

if __name__ == "__main__":
    # Data Preparation
    input_data = "data/raw/arxiv_data_10000.json"
    sampled_data = "data/processed/arxiv_data_test10k.json"
    prepare_dataset(input_data, sampled_data, TARGETS, TOTAL_PAPERS)

    # CSV
    csv_files = process_papers(sampled_data, "data/processed/graph_data")

    # Graph
    G = build_graph(csv_files)

    # Embeddings
    G = add_embeddings(G)

    # Clustering
    G = cluster_papers(G)
    G = cluster_authors(G, method='hybrid')

    # Save graph
    output_path = "data/processed/graph.graphml"
    nx.write_graphml(G, output_path)
    print(f"Graph saved to {output_path}")
    print(f"Graph contains {len(G.nodes)} nodes and {len(G.edges)} edges")
