from typing import List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from ui import utils


def _display_cluster_results(df: pd.DataFrame, stats: dict, title: str) -> None:
    if df.empty:
        st.warning("No clustered entities found for the selected configuration.")
        return
    st.markdown(f"**{title}**")
    st.write(f"Number of clusters: {stats.get('num_clusters')}")
    cluster_sizes = stats.get("cluster_sizes", {})
    size_df = pd.DataFrame({"cluster": list(cluster_sizes.keys()), "size": list(cluster_sizes.values())})
    size_chart = px.bar(size_df, x="cluster", y="size", color="cluster", title="Cluster Sizes")
    st.plotly_chart(size_chart, use_container_width=True)
    scatter = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data=["label", "id"],
        title="2D Projection of Clusters",
    )
    st.plotly_chart(scatter, use_container_width=True)
    with st.expander("Sample entities per cluster", expanded=False):
        for cid in sorted(cluster_sizes.keys(), key=lambda x: int(x)):
            members = df[df["cluster"] == cid].head(10)[["label", "id"]]
            st.markdown(f"**Cluster {cid} ({cluster_sizes[cid]} entities)**")
            st.table(members)


def show() -> None:
    st.header("Clustering")
    G = utils.load_graph()
    entity_type = st.selectbox("Cluster entity type", ["paper", "author"], key="cluster_entity_type")
    method_options: List[str]
    if entity_type == "paper":
        method_options = ["kmeans"]
    else:
        method_options = ["kmeans", "louvain", "hybrid"]
    if len(method_options) > 1:
        selected_methods = st.multiselect(
            "Select clustering methods",
            method_options,
            default=[method_options[0]],
            key="cluster_methods",
        )
    else:
        selected_methods = method_options
    n_clusters = st.slider("Number of clusters (K‑Means)", 2, 20, 10, key="cluster_n_clusters")
    dim_method = st.selectbox("Dimensionality reduction", ["pca", "tsne"], key="cluster_dim_method")
    if st.button("Compute Clusters", key="compute_clusters_button"):
        tabs: List[Tuple[str, pd.DataFrame, dict]] = []
        with st.spinner("Clustering in progress..."):
            for method in selected_methods:
                df, stats = utils.run_clustering(
                    G,
                    entity_type=entity_type,
                    method=method,
                    n_clusters=n_clusters,
                    dim_method=dim_method,
                )
                tabs.append((method.capitalize(), df, stats))
        if tabs:
            tab_labels = [f"{label}" for label, _, _ in tabs]
            tab_objects = st.tabs(tab_labels)
            for (label, df, stats), tab in zip(tabs, tab_objects):
                with tab:
                    _display_cluster_results(df, stats, f"{entity_type.capitalize()} clustering using {label}")
        else:
            st.warning("No methods selected.")
    else:
        st.info("Select parameters and click 'Compute Clusters' to run clustering.")
