# retriever.py
# ONE job: given a question, find the most relevant chunks from ChromaDB.
# This is the "retrieval" part of Retrieval-Augmented Generation.

from sentence_transformers import SentenceTransformer
from embedder import load_collection, EMBEDDING_MODEL


def retrieve_relevant_chunks(query: str, n_results: int = 6) -> list[dict]:
    """
    Takes a plain English question and returns the most semantically
    relevant chunks from ChromaDB.

    It does NOT answer the question — it just finds the right evidence.
    The LLM in qa_chain.py does the answering.

    Returns a list of dicts, each with:
      - 'text'   : the chunk content
      - 'source' : which document it came from
      - 'index'  : which chunk number it was
    """
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Convert the question into an embedding using the same model
    # we used to embed the document chunks.
    # IMPORTANT: you must use the same model for both — otherwise the
    # numbers won't be in the same "space" and similarity won't work.
    query_embedding = model.encode([query]).tolist()

    collection = load_collection()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    # Reformat the results into a clean list of dicts
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":   results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "index":  results["metadatas"][0][i].get("chunk_index", i)
        })

    return chunks


# Quick test when run directly
if __name__ == "__main__":
    query = input("Enter a test question: ")
    chunks = retrieve_relevant_chunks(query)

    print(f"\nTop {len(chunks)} relevant chunks for: '{query}'\n")
    for i, chunk in enumerate(chunks):
        print(f"[Chunk {chunk['index']} from '{chunk['source']}']")
        print(chunk["text"][:400])
        print("-" * 50)