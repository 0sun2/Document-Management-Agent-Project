import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
import pdfplumber
from docx import Document as DocxDocument
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    from chromadb import PersistentClient
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )
except ImportError as error:
    raise RuntimeError(
        "chromadb is required. Install dependencies from backend/requirements.txt"
    ) from error

# .env 파일 로드
load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "storage"
DOCS_DIR = DATA_DIR / "originals"
CHROMA_DIR = DATA_DIR / "chroma"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:9000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "qwen3-4b-ft")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

ALLOWED_ORIGINS = list(
    filter(
        None,
        [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",")],
    )
) or DEFAULT_ALLOWED_ORIGINS

EMBEDDER_NAME = os.getenv(
    "EMBEDDER_MODEL", "intfloat/multilingual-e5-large-instruct"
)

app = FastAPI(title="RAG FastAPI backend for grounded-query")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskPayload(BaseModel):
    question: str = Field(..., description="User question")
    top_k: int = Field(
        5,
        ge=1,
        le=10,
        description="Number of vector search results to retrieve",
    )


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks: int


class AskResponse(BaseModel):
    answer: str
    matches: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    used_web_fallback: bool
    source: str
    top_similarity: float


def load_embedder() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDER_NAME, normalize_embeddings=True
    )


embedding_function = load_embedder()

chroma_client = PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
    embedding_function=embedding_function,
)


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    if not text.strip():
        return []

    chunks: List[str] = []
    position = 0
    length = len(text)
    step = max(chunk_size - overlap, 1)

    while position < length:
        end = min(position + chunk_size, length)
        chunk = text[position:end].strip()
        if chunk:
            chunks.append(chunk)
        position += step
    return chunks


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)

    if suffix == ".docx":
        document = DocxDocument(path)
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        return "\n".join(paragraphs)

    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file format: {suffix}. Supported: pdf, docx, txt, md",
    )


async def query_vllm(prompt: str) -> str:
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful Korean assistant that answers strictly "
                    "from the provided context. If context is insufficient, say so."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.getenv("VLLM_TEMPERATURE", "0.2")),
        "max_tokens": int(os.getenv("VLLM_MAX_TOKENS", "1024")),
    }

    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            f"{VLLM_BASE_URL}/chat/completions", json=payload, headers=headers
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = getattr(error.response, "text", "") or str(error)
            raise HTTPException(
                status_code=502,
                detail=f"vLLM request failed: {detail}",
            ) from error

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="No choices returned from vLLM")

    message = choices[0].get("message", {})
    return message.get("content", "").strip()


def build_prompt(question: str, chunks: List[str]) -> str:
    context_text = "\n\n".join(
        f"[{idx + 1}] {chunk}" for idx, chunk in enumerate(chunks)
    )
    return (
        "아래는 사용자의 질문과 관련 문서 조각입니다.\n"
        "[CONTEXT]\n"
        f"{context_text}\n\n"
        f"[QUESTION]\n{question}\n\n"
        "컨텍스트에 있는 정보만 사용해 한국어로 정확하게 답변해 주세요. "
        "근거가 없는 내용은 추론하지 말고 부족하면 모른다고 말하세요."
    )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    original_filename: Optional[str] = Form(None),
) -> UploadResponse:
    doc_id = document_id or str(uuid4())
    filename = original_filename or file.filename
    document_dir = DOCS_DIR / doc_id

    if document_dir.exists():
        shutil.rmtree(document_dir)
    document_dir.mkdir(parents=True, exist_ok=True)

    destination = document_dir / filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_file(destination)
    chunks = chunk_text(text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="문서에서 텍스트를 추출하지 못했습니다.",
        )

    existing_ids = collection.get(
        where={"doc_id": doc_id}
    ).get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    ids = [f"{doc_id}_chunk_{idx}" for idx, _ in enumerate(chunks)]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": idx,
        }
        for idx in range(len(chunks))
    ]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunks=len(chunks),
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskPayload) -> AskResponse:
    results = collection.query(
        query_texts=[payload.question],
        n_results=payload.top_k,
        include=["metadatas", "documents", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found.",
        )

    similarities = [
        max(0.0, 1.0 - distance) for distance in distances if distance is not None
    ]

    prompt = build_prompt(payload.question, documents)
    answer = await query_vllm(prompt)

    matches: List[Dict[str, Any]] = []
    for idx, metadata in enumerate(metadatas):
        matches.append(
            {
                "doc_id": metadata.get("doc_id"),
                "filename": metadata.get("filename"),
                "chunk_index": metadata.get("chunk_index"),
                "chunk_text": documents[idx],
                "similarity": similarities[idx] if idx < len(similarities) else None,
            }
        )

    top_similarity = max(similarities) if similarities else 0.0

    return AskResponse(
        answer=answer,
        matches=matches,
        citations=[],
        used_web_fallback=False,
        source="db",
        top_similarity=top_similarity,
    )


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> Dict[str, str]:
    ids_to_delete = collection.get(
        where={"doc_id": doc_id}
    ).get("ids", [])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    document_dir = DOCS_DIR / doc_id
    if document_dir.exists():
        shutil.rmtree(document_dir)

    return {"status": "deleted", "doc_id": doc_id}


def run() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("FASTAPI_HOST", "127.0.0.1"),
        port=int(os.getenv("FASTAPI_PORT", "8000")),
        reload=bool(int(os.getenv("FASTAPI_RELOAD", "0"))),
    )


if __name__ == "__main__":
    run()
