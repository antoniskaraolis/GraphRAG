from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from ui import utils


def _render_entity_details(G: Any, node_id: Any) -> None:
    data = utils.get_entity_details(G, node_id)
    st.markdown("**Attributes**")
    st.json(data, expanded=False)
    rels = utils.get_relationships(G, node_id)
    st.markdown("**Related entities**")
    for key, nodes in rels.items():
        if not nodes:
            continue
        st.write(f"{key.capitalize()} ({len(nodes)}):")
        for n in nodes[:10]:
            label = G.nodes[n].get("title") or G.nodes[n].get("name") or G.nodes[n].get("category") or str(n)
            st.write(f"- {n}: {label}")


def show() -> None:
    st.header("Search")
    G = utils.load_graph()
    query = st.text_input("Enter your search term", key="search_query")
    entity_type = st.selectbox(
        "Search for",
        ["paper", "author", "topic"],
        index=0,
        key="search_entity_type",
    )
    top_k = st.slider("Top K results", 1, 50, 10, key="search_top_k")
    results: List[Dict[str, Any]] = []
    if st.button("Run Search", key="run_search_button"):
        with st.spinner("Running search..."):
            results = utils.search_entities(G, query, entity_type, top_k)
    if results:
        st.success(f"Found {len(results)} matching {entity_type}s")
        df = pd.DataFrame(results)
        display_cols = ["label", "score", "cluster", "id"]
        st.dataframe(df[display_cols], use_container_width=True)
        selected_label = st.selectbox(
            "Select a result to explore its details",
            options=[res["label"] for res in results],
            key="search_select_result",
        )
        selected_node = None
        for res in results:
            if res["label"] == selected_label:
                selected_node = res["id"]
                break
        if selected_node is not None:
            with st.expander("Entity details and relationships", expanded=True):
                _render_entity_details(G, selected_node)
    elif query:
        st.warning("No results found. Try adjusting your query or entity type.")
    else:
        st.info("Enter a query and click Run Search to begin.")
