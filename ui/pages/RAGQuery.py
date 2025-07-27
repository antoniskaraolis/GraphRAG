import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import streamlit as st

from ui import utils


def _display_evidence(G, paper_ids: List[Any]) -> None:
    st.markdown("**Supporting Papers**")
    if not paper_ids:
        st.info("No evidence returned.")
        return
    for pid in paper_ids:
        data = G.nodes[pid]
        title = data.get("title", pid)
        abstract = data.get("abstract", "")
        url = data.get("url")
        st.markdown(f"- **{title}** ({pid})")
        if url:
            st.markdown(f"  - [Link]({url})")
        st.markdown(f"  - {abstract[:300]}{'...' if len(abstract) > 300 else ''}")


def show() -> None:
    st.header("RAG Query")
    G = utils.load_graph()
    question = st.text_area("Ask a question about the research graph", key="rag_question")
    top_k = st.slider("Number of papers to retrieve", 1, 20, 5, key="rag_top_k")
    provider = st.selectbox("LLM provider", ["openai", "simple"], key="rag_provider")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.1, key="rag_temperature")
    if st.button("Run Query", key="rag_run_button"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating answer..."):
                answer, evidence = utils.rag_answer(
                    question,
                    G,
                    top_k=top_k,
                    model="openai" if provider == "openai" else "simple",
                    temperature=temperature,
                )
            st.subheader("Answer")
            st.write(answer)
            _display_evidence(G, evidence)
            if evidence:
                path = utils.reasoning_path(G, evidence[0])
                with st.expander("Reasoning path", expanded=False):
                    paper_title = G.nodes[path["paper"]].get("title", path["paper"])
                    st.write(f"Start paper: {paper_title}")
                    st.write("Similar papers:")
                    for p in path["similar_papers"]:
                        label = G.nodes[p].get("title", p)
                        st.write(f"- {label}")
                    st.write("Authors:")
                    for a in path["authors"]:
                        label = G.nodes[a].get("name", a)
                        st.write(f"- {label}")
            export_data: Dict[str, Any] = {
                "question": question,
                "answer": answer,
                "evidence": evidence,
            }
            json_str = json.dumps(export_data, indent=2)
            filename = f"rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            st.download_button(
                label="Download results as JSON",
                data=json_str,
                file_name=filename,
                mime="application/json",
            )
    else:
        st.info("Enter a question and click 'Run Query' to retrieve an answer.")
