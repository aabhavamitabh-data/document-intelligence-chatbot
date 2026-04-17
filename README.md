# Document Intelligence Chatbot

An AI-powered web application that lets you upload any PDF and have a 
natural language conversation with it — built on a production-grade 
RAG (Retrieval-Augmented Generation) pipeline.

## Live Demo
[Click here to try it →](YOUR-HUGGINGFACE-URL-GOES-HERE)

## What it does
- Upload any PDF document
- Ask questions in plain English
- Get precise, grounded answers with source citations
- Powered by semantic search — finds relevant passages by meaning, not keywords

## How it works
PDF → Text Extraction → Chunking → Embeddings → ChromaDB
↓
User Question → Embedding → Semantic Search → Relevant Chunks
↓
LLM (Llama 3 via Groq) → Answer

## Tech Stack
| Component | Technology |
|-----------|-----------|
| PDF parsing | PyMuPDF |
| Text chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector database | ChromaDB |
| LLM | Llama 3.3 70B via Groq API |
| Web interface | Streamlit |
| Deployment | Hugging Face Spaces |

## Architecture

The project is structured with clear separation of concerns:

- `ingestor.py` — PDF reading and text chunking
- `embedder.py` — converts chunks to vector embeddings and stores in ChromaDB
- `retriever.py` — semantic search against the vector database
- `qa_chain.py` — prompt engineering and LLM integration
- `app.py` — Streamlit web interface

## Running locally

```bash
# Clone the repo
git clone https://github.com/YOUR-USERNAME/document-intelligence-chatbot
cd document-intelligence-chatbot

# Create conda environment
conda create -n doc-chatbot python=3.11
conda activate doc-chatbot

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```

## Key concepts demonstrated
- **RAG pipeline** — retrieval-augmented generation from scratch
- **Vector embeddings** — semantic search beyond keyword matching  
- **Prompt engineering** — grounded, citation-backed LLM responses
- **Production code structure** — modular, single-responsibility files
- **API integration** — Groq LLM API with error handling