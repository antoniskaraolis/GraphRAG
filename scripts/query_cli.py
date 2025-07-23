#!/usr/bin/env python3
# scripts/query_cli.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker.*")

import networkx as nx
import numpy as np
from graphrag.query import (
    semantic_search,
    relationship_queries,
    rag_queries,
    multi_hop_exploration,
    analyze_clusters,
    visualize_collaboration_network
)

def convert_embeddings(G):
    for node, data in G.nodes(data=True):
        if 'embedding' in data and isinstance(data['embedding'], str):
            data['embedding'] = np.array(
                [float(x) for x in data['embedding'].split(';')]
            )

def main():
    graph_path = "data/processed/graph.graphml"
    G = nx.read_graphml(graph_path)

    convert_embeddings(G)

    current_paper = None

    while True:
        print("\n===== GraphRAG Query System =====")
        print("1. Semantic search")
        print("2. Relationship queries")
        print("3. RAG queries (trending topics)")
        print("4. Multi-hop exploration")
        print("5. Analyze clusters")
        print("6. Visualize collaboration network")
        print("7. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            query = input("Enter search query: ")
            results = semantic_search(G, query)
            if results:
                current_paper = results[0][0]
                title = G.nodes[current_paper].get('title', 'Unknown title')
                print(f"Selected top paper: {title}")
            else:
                print("No matching papers found.")
                current_paper = None
                
        elif choice == "2":
            if current_paper:
                relationship_queries(G, current_paper)
            else:
                print("Please perform a semantic search first")

        elif choice == "3":
            rag_queries(G)

        elif choice == "4":
            if current_paper:
                multi_hop_exploration(G, current_paper)
            else:
                print("Please perform a semantic search first")

        elif choice == "5":
            analyze_clusters(G)

        elif choice == "6":
            output = input("Output filename [collaboration_network.png]: ") or "collaboration_network.png"
            visualize_collaboration_network(G, output)

        elif choice == "7":
            print("Exit")
            break

        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    main()
