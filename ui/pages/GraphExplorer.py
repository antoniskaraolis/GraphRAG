from typing import Dict, List, Set

import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from ui import utils


def _build_subgraph_nodes_edges(
    G, types: Set[str], max_nodes: int
) -> tuple[List[Node], List[Edge], Set[str]]:
    nodes: List[Node] = []
    included_ids: Set[str] = set()
    colours: Dict[str, str] = {
        "paper": "#1f77b4",
        "author": "#2ca02c",  
        "topic": "#d62728",   
    }
    for node, data in G.nodes(data=True):
        if data.get("type") not in types:
            continue
        label = data.get("title") or data.get("name") or data.get("category") or str(node)
        colour = colours.get(data.get("type"), "#7f7f7f")
        nodes.append(
            Node(
                id=str(node),
                label=str(label)[:50],
                size=15,
                color=colour,
                shape="ellipse",
            )
        )
        included_ids.add(node)
        if len(included_ids) >= max_nodes:
            break
    edges: List[Edge] = []
    for u, v, data in G.edges(data=True):
        if u in included_ids and v in included_ids:
            edges.append(Edge(str(u), str(v)))
    return nodes, edges, included_ids


def show() -> None:
    st.header("Graph Explorer")
    G = utils.load_graph()
    with st.expander("Settings", expanded=True):
        selected_types = st.multiselect(
            "Entity types to include",
            ["paper", "author", "topic"],
            default=["paper", "author"],
        )
        max_nodes = st.slider(
            "Maximum number of nodes to display",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
        )
    with st.spinner("Building subgraph..."):
        nodes, edges, included = _build_subgraph_nodes_edges(
            G, set(selected_types), max_nodes
        )
    config = Config(
        width=900,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="yellow",
        collapsible=False,
    )
    st.markdown("### Interactive Graph")
    value = agraph(
        nodes=nodes,
        edges=edges,
        config=config,
    )
    if value and "selected_node" in value and value["selected_node"]:
        node_id = value["selected_node"]["id"]
        with st.expander("Selected node details", expanded=True):
            st.write(f"**Node ID:** {node_id}")
            try:
                utils_data = utils.get_entity_details(G, node_id)
                st.json(utils_data, expanded=False)
                rels = utils.get_relationships(G, node_id)
                st.markdown("**Relationships**")
                for key, nodes_list in rels.items():
                    if not nodes_list:
                        continue
                    st.write(f"{key.capitalize()} ({len(nodes_list)}):")
                    for n in nodes_list[:10]:
                        label = G.nodes[n].get("title") or G.nodes[n].get("name") or G.nodes[n].get("category") or str(n)
                        st.write(f"- {n}: {label}")
            except Exception as exc:
                st.error(f"Could not retrieve details for node {node_id}: {exc}")
    else:
        st.info("Click on a node to view its details and relationships.")
