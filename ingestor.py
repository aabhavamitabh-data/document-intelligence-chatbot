# ingestor.py
# This file has ONE job: take a PDF file and turn it into clean text chunks.
# Think of it as the "document reader" stage of our pipeline.

import fitz  # this is PyMuPDF — 'fitz' is its internal name, don't worry about it
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF file and extracts all the text from every page.
    Returns one big string containing the entire document's text.
    """
    doc = fitz.open(pdf_path)  # open the PDF
    full_text = ""

    for page in doc:                         # loop through every page
        full_text += page.get_text()         # extract text from that page

    doc.close()
    return full_text


def split_text_into_chunks(text: str) -> list:
    """
    Takes a long string of text and splits it into overlapping chunks.

    Why overlapping? Imagine an important sentence sits right at the boundary
    between two chunks. Without overlap, that sentence gets cut in half and
    loses its meaning. Overlap ensures no context is lost at the edges.

    chunk_size=500   → each chunk is ~500 characters long
    chunk_overlap=50 → the last 50 characters of one chunk repeat
                       at the start of the next chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=75,
        separators=["\n\n", "\n", ".", " ", ""]
        # it tries to split at paragraph breaks first,
        # then line breaks, then sentences, then words
    )

    chunks = splitter.split_text(text)
    return chunks


def ingest_pdf(pdf_path: str) -> list:
    """
    Master function that combines both steps above.
    Give it a path to a PDF → get back a clean list of text chunks.
    This is the function the rest of the app will call.
    """
    print(f"Opening: {pdf_path}")

    raw_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(raw_text)} characters of text")

    chunks = split_text_into_chunks(raw_text)
    print(f"Split into {len(chunks)} chunks")

    # Preview the first 3 chunks so you can see what we're working with
    print("\n--- PREVIEW: First 3 chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:\n{chunk}")
        print("-" * 40)

    return chunks


# This block only runs when you execute this file directly.
# It won't run when other files import from this file.
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestor.py path/to/your/file.pdf")
    else:
        pdf_path = sys.argv[1]
        chunks = ingest_pdf(pdf_path)
        print(f"\nDone. Total chunks ready for embedding: {len(chunks)}")
