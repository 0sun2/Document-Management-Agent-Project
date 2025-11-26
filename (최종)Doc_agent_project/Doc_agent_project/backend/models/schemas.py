"""Pydantic schemas for request/response models."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AskPayload(BaseModel):
    """Request payload for question answering."""
    question: str = Field(..., description="User question")
    top_k: int = Field(
        5,
        ge=1,
        le=10,
        description="Number of vector search results to retrieve",
    )


class UploadResponse(BaseModel):
    """Response after document upload."""
    doc_id: str
    filename: str
    chunks: int
    size_bytes: int
    uploaded_at: datetime


class DocumentSummary(BaseModel):
    """Summary information for a document."""
    doc_id: str
    filename: str
    chunks: int
    size_bytes: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    """Response containing list of documents."""
    documents: List[DocumentSummary]


class AskResponse(BaseModel):
    """Response for question answering."""
    answer: str
    matches: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    used_web_fallback: bool
    source: str
    top_similarity: float
    web_search_error: Optional[str] = None


class AgentPayload(BaseModel):
    """Request payload for agent."""
    question: str = Field(..., description="User question for agent")


class AgentResponse(BaseModel):
    """Response from agent."""
    answer: str
    tool_calls_made: int
