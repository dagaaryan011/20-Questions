from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag_core


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_core.build_index()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    reply = rag_core.chat(req.session_id, req.message)
    return {"reply": reply}


@app.post("/reset")
def reset(req: ResetRequest):
    rag_core.reset_session(req.session_id)
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
