import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)


from importlib import import_module
from typing import Callable, Dict
from openai import OpenAI
import streamlit as st


def load_pages() -> Dict[str, Callable[[], None]]:
    pages = {
        "Dashboard": "Dashboard",
        "Search": "Search",
        "Graph Explorer": "GraphExplorer",
        "Clustering": "Clustering",
        "RAG Query": "RAGQuery",
    }
    loaded: Dict[str, Callable[[], None]] = {}
    for label, module_name in pages.items():
        module = import_module(f"ui.pages.{module_name}")
        if hasattr(module, "show"):
            loaded[label] = getattr(module, "show")
    return loaded


def main() -> None:
    st.set_page_config(
        page_title="GraphRAG Explorer",
        page_icon="graphrag",
        layout="wide",
    )
    pages = load_pages()
    with st.sidebar:
        st.title("GraphRAG Explorer")
        selected = st.radio(
            "Navigate",
            list(pages.keys()),
            format_func=lambda x: x,
        )
    show_fn = pages.get(selected)
    if show_fn is not None:
        show_fn()
    else:
        st.error("Selected page could not be loaded.")


if __name__ == "__main__":
    main()
