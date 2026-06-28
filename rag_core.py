import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import faiss
from fastembed import TextEmbedding
from groq import Groq
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "files" / "gullivers-travels.pdf.pdf"
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are the host of a 20 Questions game. The secret answer is the book "
    "\"Gulliver's Travels\". The player will ask yes/no questions to guess it. "
    "Answer strictly based on the book context provided below, with 'Yes', 'No', "
    "or a short hint about genre/author if the context doesn't clearly say. "
    "If the player guesses the title (or something close to it), congratulate "
    "them and declare they won.\n\nBook context:\n{context}"
)

_groq_client: Groq | None = None
_embedding_model: TextEmbedding | None = None
_chunks: List[str] = []
_vectorstore: faiss.IndexFlatL2 | None = None
_sessions: Dict[str, List[dict]] = {}


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _groq_client


def _embed(texts: List[str]) -> np.ndarray:
    return np.array(list(_embedding_model.embed(texts)))


def build_index(pdf_path: Path = PDF_PATH) -> None:
    global _embedding_model, _chunks, _vectorstore

    print("Initializing embeddings...")
    _embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

    print(f"Loading PDF from: {pdf_path}")
    text = "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf_path)).pages)

    size, overlap = 1000, 100
    _chunks = [text[i:i + size] for i in range(0, len(text), size - overlap)]
    print(f"Split document into {len(_chunks)} chunks")

    embeddings_np = _embed(_chunks)
    _vectorstore = faiss.IndexFlatL2(embeddings_np.shape[1])
    _vectorstore.add(embeddings_np)
    print("Vectorstore created with FAISS.")


def _retrieve(query: str, k: int = 3) -> List[str]:
    query_embedding = _embed([query])
    _, indices = _vectorstore.search(query_embedding, k=k)
    return [_chunks[i] for i in indices[0]]


def reset_session(session_id: str) -> None:
    _sessions[session_id] = []


def chat(session_id: str, user_message: str) -> str:
    history = _sessions.setdefault(session_id, [])

    context = "\n\n".join(_retrieve(user_message))
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.1,
    )
    answer = response.choices[0].message.content

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})

    return answer
