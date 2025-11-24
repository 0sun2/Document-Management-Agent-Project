"""Document management API routes."""
import shutil
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.schemas import UploadResponse, DocumentListResponse, DocumentSummary
from models.database import get_supabase_client
from services.document_service import (
    extract_text_from_file,
    save_doc_metadata,
    load_doc_metadata,
    get_doc_file_metadata,
    _tokenize_text,
    _extract_keywords_from_filename,
)
from services.embedding_service import embed_passages
from core.config import DOCS_DIR
from core.logging import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    q: Optional[str] = Query(default=None, alias="q")
) -> DocumentListResponse:
    """
    List all documents with optional search.

    Args:
        q: Optional search query to filter documents by keywords

    Returns:
        List of document summaries
    """
    sb = get_supabase_client()
    documents: List[DocumentSummary] = []
    search_tokens = set(_tokenize_text(q)) if q else set()

    if sb is not None:
        try:
            res = sb.table("rag_docs").select("doc_id, filename, chunk_index").execute()
            rows = res.data or []

            docs: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                doc_id = row.get("doc_id")
                if not doc_id:
                    continue
                entry = docs.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "filename": row.get("filename") or "unknown",
                        "chunks": 0,
                    },
                )
                entry["chunks"] += 1

            for info in docs.values():
                size_bytes, uploaded_at = get_doc_file_metadata(info["doc_id"])
                meta = load_doc_metadata(info["doc_id"])
                filename = meta.get("filename") if meta else info["filename"]
                if meta and meta.get("uploaded_at"):
                    try:
                        uploaded_at = datetime.fromisoformat(meta["uploaded_at"])
                    except Exception:
                        pass
                documents.append(
                    DocumentSummary(
                        doc_id=info["doc_id"],
                        filename=filename or info["filename"],
                        chunks=info["chunks"],
                        size_bytes=size_bytes,
                        uploaded_at=uploaded_at,
                    )
                )
        except Exception as e:
            logger.warning(f"Supabase query failed, using local fallback: {e}")

    if not documents:
        # Fallback to local storage if Supabase unavailable
        for doc_dir in DOCS_DIR.iterdir():
            if not doc_dir.is_dir():
                continue
            files = [f for f in doc_dir.iterdir() if f.is_file()]
            if not files:
                continue
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            stat = latest_file.stat()
            meta = load_doc_metadata(doc_dir.name)
            filename = meta.get("filename") if meta else latest_file.name
            uploaded_at = None
            if meta and meta.get("uploaded_at"):
                try:
                    uploaded_at = datetime.fromisoformat(meta["uploaded_at"])
                except Exception:
                    uploaded_at = datetime.fromtimestamp(stat.st_mtime)
            else:
                uploaded_at = datetime.fromtimestamp(stat.st_mtime)
            documents.append(
                DocumentSummary(
                    doc_id=doc_dir.name,
                    filename=filename or latest_file.name,
                    chunks=0,
                    size_bytes=stat.st_size,
                    uploaded_at=uploaded_at,
                )
            )

    if search_tokens:
        filtered_docs = []
        for doc in documents:
            meta = load_doc_metadata(doc.doc_id)
            keywords = set(meta.get("keywords", []) if meta else [])
            keywords.update(_extract_keywords_from_filename(doc.filename))
            if keywords.intersection(search_tokens):
                filtered_docs.append(doc)
        if filtered_docs:
            documents = filtered_docs

    documents.sort(key=lambda d: d.filename.lower())
    return DocumentListResponse(documents=documents)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    original_filename: Optional[str] = Form(None),
) -> UploadResponse:
    """
    Upload and index a document.

    Args:
        file: Document file to upload
        document_id: Optional pre-generated document ID
        original_filename: Optional original filename

    Returns:
        Upload response with document metadata

    Raises:
        HTTPException: If Supabase is not configured or upload fails
    """
    sb = get_supabase_client()
    if sb is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured: Add SUPABASE_URL / SUPABASE_ANON_KEY to .env",
        )

    # Use frontend-provided UUID or generate new one
    doc_id = document_id or str(uuid4())
    filename = original_filename or file.filename

    # Save file locally
    document_dir = DOCS_DIR / doc_id
    if document_dir.exists():
        shutil.rmtree(document_dir)
    document_dir.mkdir(parents=True, exist_ok=True)
    destination = document_dir / filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    upload_time = datetime.utcnow()
    file_stat = destination.stat()
    size_bytes = file_stat.st_size
    timestamp_float = upload_time.timestamp()
    os.utime(destination, (timestamp_float, timestamp_float))

    # Extract text
    text = extract_text_from_file(destination)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Failed to extract text from document")

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(text)

    if not chunks:
        raise HTTPException(status_code=400, detail="Document has no chunks")

    # Generate embeddings
    vecs = embed_passages(chunks)

    # Save to Supabase
    rows = []
    for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
        rows.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "content": chunk,
                "embedding": vec.tolist(),
            }
        )

    try:
        sb.table("rag_docs").insert(rows).execute()
    except Exception as e:
        logger.error(f"Supabase insert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Supabase save failed: {str(e)}")

    save_doc_metadata(doc_id, filename, upload_time)

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunks=len(chunks),
        size_bytes=size_bytes,
        uploaded_at=upload_time,
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> Dict[str, str]:
    """
    Delete a document and its embeddings.

    Args:
        doc_id: Document ID to delete

    Returns:
        Success message

    Raises:
        HTTPException: If deletion fails
    """
    sb = get_supabase_client()
    if sb is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured"
        )

    try:
        # Delete from Supabase
        sb.table("rag_docs").delete().eq("doc_id", doc_id).execute()

        # Delete local files
        document_dir = DOCS_DIR / doc_id
        if document_dir.exists():
            shutil.rmtree(document_dir)

        logger.info(f"Document deleted: {doc_id}")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )
