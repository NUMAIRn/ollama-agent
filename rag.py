import sys
import hashlib
import pickle
from pathlib import Path

import numpy as np
import ollama
from pypdf import PdfReader

CHAT_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4                          
CACHE_DIR = Path(".rag_cache")


def load_pdf(path: str) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for page_num, text in pages:
        for start in range(0, len(text), step):
            piece = text[start:start + CHUNK_SIZE].strip()
            if len(piece) > 50:
                chunks.append({"page": page_num, "text": piece})
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    vectors = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = ollama.embed(model=EMBED_MODEL, input=batch)
        vectors.extend(response["embeddings"])
    matrix = np.array(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return matrix


def build_index(pdf_path: str) -> tuple[list[dict], np.ndarray]:
    """Build (or load from cache) the chunk list and embedding matrix."""
    CACHE_DIR.mkdir(exist_ok=True)
    file_hash = hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()
    cache_file = CACHE_DIR / f"{file_hash}_{EMBED_MODEL.replace(':', '_')}.pkl"

    if cache_file.exists():
        print("Loaded cached index.")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Reading PDF...")
    pages = load_pdf(pdf_path)
    if not pages:
        sys.exit("No extractable text found (is the PDF scanned? You'd need OCR).")

    chunks = chunk_pages(pages)
    print(f"Embedding {len(chunks)} chunks from {len(pages)} pages...")
    embeddings = embed_texts([c["text"] for c in chunks])

    with open(cache_file, "wb") as f:
        pickle.dump((chunks, embeddings), f)
    return chunks, embeddings


def retrieve(query: str, chunks: list[dict], embeddings: np.ndarray) -> list[dict]:
    """Return the TOP_K most similar chunks to the query (cosine similarity)."""
    q = embed_texts([query])[0]
    scores = embeddings @ q
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    return [{**chunks[i], "score": float(scores[i])} for i in top_idx]


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a PDF document. "
    "Use ONLY the provided context to answer. If the answer is not in the context, "
    "say you couldn't find it in the document. Be concise and accurate."
)


def chat_loop(chunks: list[dict], embeddings: np.ndarray) -> None:
    history: list[dict] = []
    print("\nChatbot ready! Ask questions about your PDF. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        hits = retrieve(question, chunks, embeddings)
        context = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += history[-4:]
        messages.append({
            "role": "user",
            "content": f"Context from the document:\n{context}\n\nQuestion: {question}",
        })

        print("Bot: ", end="", flush=True)
        answer = ""
        for part in ollama.chat(model=CHAT_MODEL, messages=messages, stream=True):
            token = part["message"]["content"]
            answer += token
            print(token, end="", flush=True)

        pages = sorted({h["page"] for h in hits})
        print(f"\n     (sources: pages {', '.join(map(str, pages))})\n")

        # Store only the plain question/answer in history (not the big context)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python rag_chatbot.py path/to/document.pdf")
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        sys.exit(f"File not found: {pdf_path}")

    chunks, embeddings = build_index(pdf_path)
    chat_loop(chunks, embeddings)


if __name__ == "__main__":
    main()