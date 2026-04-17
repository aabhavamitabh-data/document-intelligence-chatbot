# qa_chain.py
# ONE job: take a question + relevant chunks, send them to Groq (Llama 3),
# and return a grounded, precise answer.

import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from retriever import retrieve_relevant_chunks

# Load .env file using explicit path so it always finds it
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Grab the key and immediately check it loaded
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Check your .env file contains: GROQ_API_KEY=gsk_your_key_here"
    )

print(f"Groq API key loaded: {api_key[:8]}...")

# Create the Groq client
client = Groq(api_key=api_key)

# Llama 3.3 70B — powerful, fast, and completely free on Groq
# This is a genuinely impressive model, not a toy
MODEL = "llama-3.3-70b-versatile"


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Builds the full prompt we send to the LLM.

    We give it:
    1. A clear role: precise document assistant
    2. A strict rule: only use the provided context, never guess
    3. The retrieved evidence chunks with source labels
    4. The user's question
    """
    context_block = ""
    for i, chunk in enumerate(chunks):
        context_block += f"\n[Passage {i+1} from '{chunk['source']}']\n"
        context_block += chunk["text"]
        context_block += "\n"

    prompt = f"""You are a precise document assistant. Your job is to answer
questions based ONLY on the provided context passages below.

Rules:
- If the answer is in the context, answer clearly and specifically.
- If the context does not contain enough information to answer,
  say "I couldn't find that in the document." Do not guess.
- Keep answers concise but complete.
- Answer naturally — do not mention "passages" or "context" in your answer.

Context:
{context_block}

Question: {question}

Answer:"""

    return prompt


def answer_question(question: str) -> dict:
    """
    Full RAG pipeline: question → retrieve → prompt → Llama 3 → answer.

    Returns a dict with:
      - 'answer'  : the LLM's response
      - 'sources' : the chunks used as evidence
      - 'question': the original question
    """
    # Step 1: retrieve the most relevant chunks from ChromaDB
    chunks = retrieve_relevant_chunks(question, n_results=4)

    # Step 2: build the prompt with context baked in
    prompt = build_prompt(question, chunks)

    # Step 3: call Groq's Llama 3 model
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a precise document assistant. Answer only from the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        # Low temperature = factual and precise.
        # We want the model to stick to the document, not get creative.
        max_tokens=1024
    )

    answer = response.choices[0].message.content.strip()

    return {
        "question": question,
        "answer":   answer,
        "sources":  chunks
    }


# Test it directly when you run this file
if __name__ == "__main__":
    print("Document Q&A — powered by RAG + Llama 3 via Groq")
    print("(Make sure you ran embedder.py on your PDF first)\n")

    while True:
        question = input("Ask a question (or 'quit' to exit): ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\nSearching document and generating answer...\n")
        result = answer_question(question)

        print(f"Answer: {result['answer']}")
        print(f"\nBased on {len(result['sources'])} passages from the document.")
        print("-" * 60 + "\n")