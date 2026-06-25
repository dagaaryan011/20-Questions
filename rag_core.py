import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch.nn.functional as F
import faiss
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

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
_embedding_model: SentenceTransformer | None = None
_chunks: List[str] = []
_vectorstore: faiss.IndexFlatL2 | None = None
_sessions: Dict[str, List[dict]] = {}


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _groq_client


def _embed(texts: List[str]) -> np.ndarray:
    embeddings = _embedding_model.encode(texts, convert_to_tensor=True)
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().detach().numpy()


def build_index(pdf_path: Path = PDF_PATH) -> None:
    global _embedding_model, _chunks, _vectorstore

    print("Initializing embeddings...")
    _embedding_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )

    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )
    docs = text_splitter.split_documents(pages)
    _chunks = [doc.page_content for doc in docs]
    print(f"Split document into {len(_chunks)} chunks")

    document_texts = ["search_document: " + c for c in _chunks]
    embeddings_np = _embed(document_texts)

    _vectorstore = faiss.IndexFlatL2(embeddings_np.shape[1])
    _vectorstore.add(embeddings_np)
    print("Vectorstore created with FAISS.")


def _retrieve(query: str, k: int = 3) -> List[str]:
    query_embedding = _embed(["search_query: " + query])
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
