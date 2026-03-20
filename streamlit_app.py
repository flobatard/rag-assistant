from pathlib import Path

import streamlit as st

from ask_local_llm import ask_llm
from rag_pipeline import create_vectorstore, search

DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "context_messages" not in st.session_state:
    st.session_state.context_messages = []
if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RAG Assistant")
    st.header("Documents")

    # Upload
    uploaded_files = st.file_uploader(
        "Ajouter des PDFs",
        type="pdf",
        accept_multiple_files=True,
    )
    if uploaded_files:
        for f in uploaded_files:
            (DATA_FOLDER / f.name).write_bytes(f.read())
        st.success(f"{len(uploaded_files)} fichier(s) ajouté(s)")

    # List + delete
    pdfs = sorted(DATA_FOLDER.glob("*.pdf"))
    if pdfs:
        st.subheader("Fichiers disponibles")
        for pdf in pdfs:
            col_name, col_btn = st.columns([4, 1])
            col_name.caption(pdf.name)
            if col_btn.button("🗑️", key=f"del_{pdf.name}", help=f"Supprimer {pdf.name}"):
                pdf.unlink()
                st.rerun()
    else:
        st.info("Aucun fichier dans data/")

    # Indexation
    st.divider()
    index_btn = st.button(
        "Indexer les documents",
        disabled=not pdfs,
        use_container_width=True,
        type="primary",
    )
    if index_btn:
        with st.spinner("Indexation en cours…"):
            st.session_state.vectorstore = create_vectorstore(str(DATA_FOLDER))
        st.success("Index prêt !")

    if "vectorstore" in st.session_state:
        st.caption("Index actif")
    else:
        st.caption("Index non construit")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_ctx, tab_llm = st.tabs(["🔎 Recherche contextuelle", "🤖 Chat avec LLM"])

# ── Tab 1 : context search ────────────────────────────────────────────────────
with tab_ctx:
    for msg in st.session_state.context_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt_ctx := st.chat_input("Votre question…", key="context_input"):
        if "vectorstore" not in st.session_state:
            st.warning("Veuillez d'abord indexer les documents via la barre latérale.")
        else:
            st.session_state.context_messages.append({"role": "user", "content": prompt_ctx})
            with st.chat_message("user"):
                st.markdown(prompt_ctx)

            chunks = search(prompt_ctx, st.session_state.vectorstore)
            response = "\n\n---\n\n".join(chunks)

            st.session_state.context_messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

# ── Tab 2 : LLM chat ──────────────────────────────────────────────────────────
with tab_llm:
    for msg in st.session_state.llm_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("context"):
                with st.expander("Contexte utilisé"):
                    st.text(msg["context"])

    if prompt_llm := st.chat_input("Votre question…", key="llm_input"):
        if "vectorstore" not in st.session_state:
            st.warning("Veuillez d'abord indexer les documents via la barre latérale.")
        else:
            st.session_state.llm_messages.append({"role": "user", "content": prompt_llm})
            with st.chat_message("user"):
                st.markdown(prompt_llm)

            chunks = search(prompt_llm, st.session_state.vectorstore)
            context = "\n\n".join(chunks)

            try:
                with st.spinner("Le LLM réfléchit…"):
                    answer = ask_llm(context, prompt_llm)
            except Exception as e:
                st.error(f"Impossible de joindre Ollama : {e}")
                st.stop()

            st.session_state.llm_messages.append(
                {"role": "assistant", "content": answer, "context": context}
            )
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Contexte utilisé"):
                    st.text(context)
