from typing import List

import streamlit as st

from ui import utils


def _record_recent_query(query: str) -> None:
    if not query:
        return
    if "recent_queries" not in st.session_state:
        st.session_state.recent_queries = []
    recent: List[str] = st.session_state.recent_queries
    if query in recent:
        recent.remove(query)
    recent.insert(0, query)
    st.session_state.recent_queries = recent[:10] 


def show() -> None:
    st.header("Dashboard")
    G = utils.load_graph()
    stats = utils.get_graph_stats(G)
    papers = stats["papers"]
    authors = stats["authors"]
    topics = stats["topics"]
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Papers", f"{papers:,}")
    col2.metric("Authors", f"{authors:,}")
    col3.metric("Topics", f"{topics:,}")
    col4.metric("Number of Nodes", f"{num_nodes:,}")
    col5.metric("Number of Edges", f"{num_edges:,}")

    st.markdown("---")
    st.subheader("Quick Search")
    query = st.text_input(
        "Search for papers, authors or topics",
        key="dashboard_search_query",
    )
    entity_type = st.selectbox(
        "Entity type", ["paper", "author", "topic"], key="dashboard_entity_type"
    )
    top_k = st.slider("Number of results", 1, 20, 10, key="dashboard_top_k")
    if st.button("Search", key="dashboard_search_button"):
        with st.spinner("Searching..."):
            results = utils.search_entities(G, query, entity_type, top_k)
        _record_recent_query(query)
        if results:
            st.success(f"Found {len(results)} results")
            for res in results:
                with st.expander(res["label"]):
                    st.write(f"ID: {res['id']}")
                    st.write(f"Type: {res['type']}")
                    st.write(f"Score: {res['score']:.4f}")
                    st.write(f"Cluster: {res.get('cluster')}")
        else:
            st.warning("No matches found.")

    st.markdown("---")
    st.subheader("Recent Queries")
    recent = st.session_state.get("recent_queries", [])
    if recent:
        for i, q in enumerate(recent):
            st.write(f"{i+1}. {q}")
    else:
        st.info("No recent queries yet. Perform a search to populate this list.")
