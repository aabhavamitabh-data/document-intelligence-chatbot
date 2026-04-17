# app.py
# The web interface for your Document Intelligence Chatbot.
# Run with: streamlit run app.py

import streamlit as st
import os
import tempfile
from pathlib import Path
from ingestor import ingest_pdf
from embedder import embed_and_store, load_collection
from qa_chain import answer_question

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Intelligence Chatbot",
    page_icon="📄",
    layout="wide"
)

# ── Custom CSS for a cleaner look ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
  .answer-box {
        background-color: #1e3a5f;
        border-left: 4px solid #2563eb;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #ffffff;
    }
    .source-box {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        color: #555;
        margin-top: 0.5rem;
    }
    .status-ready {
        color: #16a34a;
        font-weight: 500;
    }
    .status-empty {
        color: #dc2626;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ─────────────────────────────────────────────
# Streamlit reruns the entire script on every interaction.
# st.session_state persists data between reruns — like a memory for the app.
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📄 Document Intelligence Chatbot</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload any PDF and ask questions about it in plain English.</div>',
            unsafe_allow_html=True)

# ── Layout: two columns ──────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2])

# ── LEFT COLUMN: Upload + status ─────────────────────────────────────────────
with left_col:
    st.subheader("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF — research papers, reports, manuals, articles."
    )

    if uploaded_file is not None:
        # Show a process button once a file is selected
        if st.button("Process Document", type="primary", use_container_width=True):
            with st.spinner("Reading and indexing your document..."):
                try:
                    # Save uploaded file to a temp location so our
                    # ingestor can read it from disk
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Get clean document name
                    doc_name = Path(uploaded_file.name).stem

                    # Run the full ingestion + embedding pipeline
                    chunks = ingest_pdf(tmp_path)
                    embed_and_store(chunks, doc_name=doc_name)

                    # Clean up the temp file
                    os.unlink(tmp_path)

                    # Mark as ready in session state
                    st.session_state.document_loaded = True
                    st.session_state.doc_name = doc_name
                    st.session_state.chat_history = []  # clear old chat

                    st.success(f"Ready! Indexed {len(chunks)} passages.")

                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

    # Status indicator
    st.divider()
    st.markdown("**Status**")
    if st.session_state.document_loaded:
        st.markdown(
            f'<span class="status-ready">● Ready — {st.session_state.doc_name}</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-empty">● No document loaded</span>',
            unsafe_allow_html=True
        )

    # Sample questions to guide the user
    st.divider()
    st.markdown("**Try asking:**")
    sample_questions = [
        "What is this document about?",
        "Summarise the key points",
        "What are the main findings?",
        "Who are the key people mentioned?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, disabled=not st.session_state.document_loaded):
            st.session_state.pending_question = q

# ── RIGHT COLUMN: Chat interface ─────────────────────────────────────────────
with right_col:
    st.subheader("Ask Questions")

    # Display chat history
    if st.session_state.chat_history:
        for exchange in st.session_state.chat_history:
            # User message
            with st.chat_message("user"):
                st.write(exchange["question"])
            # Assistant answer
            with st.chat_message("assistant"):
                st.markdown(
                    f'<div class="answer-box">{exchange["answer"]}</div>',
                    unsafe_allow_html=True
                )
                # Show source passages in an expander
                with st.expander(f"View {len(exchange['sources'])} source passages"):
                    for i, src in enumerate(exchange["sources"]):
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>Passage {i+1}</strong> '
                            f'(chunk {src["index"]} from <em>{src["source"]}</em>)<br><br>'
                            f'{src["text"][:400]}{"..." if len(src["text"]) > 400 else ""}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
    else:
        if st.session_state.document_loaded:
            st.info("Your document is ready. Ask anything below.")
        else:
            st.info("Upload and process a PDF on the left to get started.")

    # Chat input box — always at the bottom
    if st.session_state.document_loaded:
        # Handle questions from sample buttons
        default_q = st.session_state.get("pending_question", "")
        if default_q:
            st.session_state.pending_question = ""

        user_question = st.chat_input("Ask a question about your document...")

        # Process either typed question or button-selected question
        question_to_answer = user_question or default_q

        if question_to_answer:
            with st.spinner("Searching document and generating answer..."):
                try:
                    result = answer_question(question_to_answer)
                    # Add to chat history
                    st.session_state.chat_history.append(result)
                    st.rerun()  # refresh to show new message
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    else:
        st.chat_input("Upload a document first...", disabled=True)