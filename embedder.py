# embedder.py
# This file has ONE job: take text chunks and store them in ChromaDB
# as embeddings so we can search them by meaning later.

from sentence_transformers import SentenceTransformer
import chromadb
from ingestor import ingest_pdf


# We use a small but powerful open-source embedding model.
# It runs entirely on your machine — no API key needed for this step.
# "all-MiniLM-L6-v2" is fast, accurate, and used in real production systems.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# This is the folder where ChromaDB will save your vector database.
# It will be created automatically the first time you run this.
CHROMA_PATH = "chroma_db"

# We give our collection of document chunks a name inside ChromaDB.
COLLECTION_NAME = "documents"


def get_embedding_model():
    """
    Loads the embedding model.
    The first time this runs it downloads ~90MB from the internet.
    After that it's cached locally and loads in seconds.
    """
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model ready.")
    return model


def embed_and_store(chunks: list, doc_name: str = "document") -> chromadb.Collection:
    """
    Takes a list of text chunks, converts each one to an embedding,
    and stores everything in ChromaDB.

    doc_name: a label so you know which document these chunks came from.
               Useful later when you support multiple documents.
    """
    model = get_embedding_model()

    # Create (or connect to) the ChromaDB database on disk
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Delete the collection if it already exists so we start fresh.
    # This prevents duplicate chunks if you run the script twice.
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print("Cleared existing collection.")

    collection = client.create_collection(COLLECTION_NAME)

    print(f"Embedding {len(chunks)} chunks — this may take 30-60 seconds...")
    if not chunks:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned image. Try a text-based PDF instead."
        )

    # Convert all chunks to embeddings in one batch (faster than one at a time)
    embeddings = model.encode(chunks, show_progress_bar=True)

    # Store each chunk in ChromaDB with:
    # - a unique ID
    # - the embedding (the numbers)
    # - the original text (so we can retrieve it later)
    # - metadata (which document it came from)
    collection.add(
        ids=[f"{doc_name}_chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=[{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]
    )

    print(f"Stored {collection.count()} chunks in ChromaDB.")
    return collection


def load_collection() -> chromadb.Collection:
    """
    Connects to an existing ChromaDB database and returns the collection.
    This is what the rest of the app calls when answering questions —
    it doesn't re-embed, it just opens what's already saved on disk.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    return collection


# Test it directly when you run this file
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python embedder.py path/to/your/file.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    # Get a clean name for the document (just the filename, no path)
    import os
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Step 1: ingest the PDF into chunks
    chunks = ingest_pdf(pdf_path)

    # Step 2: embed and store
    collection = embed_and_store(chunks, doc_name=doc_name)

    # Step 3: do a quick test search to prove it's working
    print("\n--- TEST SEARCH ---")
    test_query = "Who is this document about?"
    print(f"Query: '{test_query}'")

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([test_query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )

    print("\nTop 2 most relevant chunks found:")
    for i, doc in enumerate(results["documents"][0]):
        print(f"\nResult {i+1}:\n{doc[:300]}...")
        print("-" * 40)

    print("\nPhase 3 complete. ChromaDB is populated and searchable.")
    