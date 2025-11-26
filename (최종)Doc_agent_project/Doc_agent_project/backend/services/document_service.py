"""Document processing service."""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pdfplumber
from docx import Document as DocxDocument
from fastapi import HTTPException
from core.config import DOCS_DIR
from core.logging import setup_logger

logger = setup_logger(__name__)

# Document metadata cache
_DOC_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}


def extract_text_from_file(path: Path) -> str:
    """
    Extract text from supported document formats.

    Args:
        path: Path to document file

    Returns:
        Extracted text content

    Raises:
        HTTPException: If file format is unsupported
    """
    suffix = path.suffix.lower()

    try:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract text from {path}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text: {str(e)}"
        )


def _tokenize_text(text: str) -> List[str]:
    """Tokenize text for keyword extraction."""
    return [
        token
        for token in re.findall(r"[0-9A-Za-z가-힣]+", (text or "").lower())
        if len(token) >= 2
    ]


# Korean stopwords for RAG coverage calculation
STOPWORDS = {
    # 지시/요청 동사
    "요약해", "요약해줘", "요약", "정리해", "정리해줘", "정리",
    "설명해", "설명해줘", "설명", "알려줘", "알려", "가르쳐줘",
    # 문서 관련 메타 단어
    "문서", "문서의", "문서에", "문서에서", "이", "그", "저", "해당",
    "위", "아래", "본", "다음", "내용", "핵심", "주요",
    # 질문 조사/어미
    "무엇", "어떤", "어떻게", "왜", "언제", "어디",
    "인가", "인지", "일까", "있나", "있는지",
    # 기타 불용어
    "대해", "대한", "관한", "관련", "통해", "위한", "같은",
}


def filter_stopwords(tokens: set) -> set:
    """
    Remove stopwords from token set.

    Args:
        tokens: Set of tokens

    Returns:
        Filtered token set without stopwords
    """
    return tokens - STOPWORDS


def _extract_keywords_from_filename(filename: str) -> List[str]:
    """Extract keywords from filename."""
    stem = Path(filename).stem
    return _tokenize_text(stem)


def load_doc_metadata(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Load document metadata from JSON file.

    Args:
        doc_id: Document ID

    Returns:
        Metadata dictionary or None if not found
    """
    cached = _DOC_METADATA_CACHE.get(doc_id)
    if cached and cached.get("_loaded"):
        return cached

    meta_path = DOCS_DIR / doc_id / "meta.json"
    if not meta_path.exists():
        return cached

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        data["_loaded"] = True
        _DOC_METADATA_CACHE[doc_id] = data
        return data
    except Exception as exc:
        logger.warning(f"Failed to load metadata for {doc_id}: {exc}")
        return cached


def save_doc_metadata(doc_id: str, filename: str, uploaded_at: datetime) -> None:
    """
    Save document metadata to JSON file.

    Args:
        doc_id: Document ID
        filename: Original filename
        uploaded_at: Upload timestamp
    """
    meta = {
        "doc_id": doc_id,
        "filename": filename,
        "uploaded_at": uploaded_at.isoformat(),
        "keywords": _extract_keywords_from_filename(filename),
    }
    meta_path = DOCS_DIR / doc_id / "meta.json"
    try:
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _DOC_METADATA_CACHE[doc_id] = meta
    except Exception as exc:
        logger.warning(f"Failed to write metadata for {doc_id}: {exc}")


def get_doc_file_metadata(doc_id: str) -> Tuple[int, datetime]:
    """
    Get document file size and modification time.

    Args:
        doc_id: Document ID

    Returns:
        Tuple of (size_bytes, uploaded_at)
    """
    document_dir = DOCS_DIR / doc_id
    if not document_dir.exists():
        return 0, datetime.fromtimestamp(0)

    files = [path for path in document_dir.glob("*") if path.is_file()]
    if not files:
        return 0, datetime.fromtimestamp(0)

    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    stat = latest_file.stat()
    uploaded_at = datetime.fromtimestamp(stat.st_mtime)
    return stat.st_size, uploaded_at


def compute_match_score(
    row: Dict[str, Any], question_token_set: set, keyword_boost: float = 0.03
) -> float:
    """
    Compute enhanced match score with keyword boosting.

    Args:
        row: Document match row with similarity score
        question_token_set: Set of tokenized question words
        keyword_boost: Boost value per keyword match

    Returns:
        Enhanced similarity score
    """
    base = float(row.get("similarity") or 0.0)
    doc_tokens: set = set()

    metadata = load_doc_metadata(row.get("doc_id", ""))
    if metadata:
        doc_tokens.update(metadata.get("keywords") or [])
    doc_tokens.update(_extract_keywords_from_filename(row.get("filename", "")))

    overlap = len(doc_tokens.intersection(question_token_set))
    content_tokens = set(_tokenize_text(row.get("content", "")))
    lexical_overlap = len(content_tokens.intersection(question_token_set))
    overlap += lexical_overlap

    boost = min(overlap, 3) * keyword_boost
    return base + boost
