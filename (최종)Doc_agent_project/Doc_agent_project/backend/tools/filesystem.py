"""Filesystem tools for file operations."""
import os
import glob
import pathlib
import mimetypes
import base64
from typing import Dict, Any
from core.config import DOC_ROOT
from core.logging import setup_logger

logger = setup_logger(__name__)


def _safe_resolve_user_path(path: str) -> str:
    """
    Safely resolve user path within DOC_ROOT.

    Args:
        path: User-provided path

    Returns:
        Absolute path within DOC_ROOT

    Raises:
        ValueError: If path is outside DOC_ROOT
    """
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = pathlib.Path(DOC_ROOT) / p
    real = os.path.abspath(os.path.realpath(str(p)))
    root = os.path.abspath(DOC_ROOT)
    if not (real == root or real.startswith(root + os.sep)):
        raise ValueError("Path outside of DOC_ROOT is not allowed")
    return real


def _read_text_file(path: str, max_bytes: int) -> str:
    """Read text file with multiple encoding attempts."""
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    for enc in ("utf-8", "utf-16", "cp949", "euc-kr"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return f"[binary head b64] {base64.b64encode(data[:4096]).decode('ascii')}"


def _read_pdf_file(path: str, max_bytes: int) -> str:
    """Read PDF file content."""
    try:
        from pypdf import PdfReader
    except Exception as e:
        return f"[pdf read error] pypdf not installed or failed to import: {e}. Try: pip install pypdf"

    try:
        reader = PdfReader(path)
        pages = []
        total = 0
        for i, pg in enumerate(reader.pages):
            txt = pg.extract_text() or ""
            pages.append(f"[page {i+1}]\n{txt}\n")
            total += len(txt.encode("utf-8"))
            if total >= max_bytes:
                break
        if not any(p.strip() for p in pages):
            return "[pdf read warning] No extractable text. It may be a scanned PDF."
        return "".join(pages)
    except Exception as e:
        return f"[pdf read error] {e}"


def tool_filesystem_glob(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for files using glob pattern.

    Args:
        args: Pattern and limit parameters

    Returns:
        Dictionary with root and matched files
    """
    pattern = args.get("pattern") or "**/*"
    limit = int(args.get("limit") or 50)
    base = pathlib.Path(DOC_ROOT)

    # Validate absolute paths
    if pathlib.Path(pattern).is_absolute():
        try:
            real_pat = os.path.abspath(os.path.realpath(pattern))
            root = os.path.abspath(DOC_ROOT)
            if not (real_pat == root or real_pat.startswith(root + os.sep)):
                return {"error": "pattern outside of DOC_ROOT is not allowed"}
            pat = real_pat
        except Exception:
            return {"error": "invalid absolute pattern"}
    else:
        pat = str(base / pattern)

    matches = sorted(glob.glob(pat, recursive=True))[:limit]

    # Double-check all results are within DOC_ROOT
    rels = []
    for m in matches:
        real = os.path.abspath(os.path.realpath(m))
        root = os.path.abspath(DOC_ROOT)
        if not (real == root or real.startswith(root + os.sep)):
            continue
        rels.append(os.path.relpath(real, start=DOC_ROOT))

    return {"root": DOC_ROOT, "matches": rels}


def tool_filesystem_read(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read file content with format detection.

    Args:
        args: Path and max_bytes parameters

    Returns:
        Dictionary with file content or error
    """
    raw = args.get("path", "")
    maxb = int(args.get("max_bytes") or 200000)

    try:
        real = _safe_resolve_user_path(raw)
        if not os.path.exists(real):
            return {"error": f"file not found: {raw}"}

        ext = (pathlib.Path(real).suffix or "").lower()
        mime = mimetypes.guess_type(real)[0] or ""

        if ext == ".pdf" or "pdf" in mime:
            content = _read_pdf_file(real, maxb)
        else:
            content = _read_text_file(real, maxb)

        size = os.path.getsize(real)
        trim_note = ""
        if size > maxb:
            trim_note = f"\n\n[truncated: returned first {maxb} bytes of {size} bytes]"

        return {
            "path": os.path.relpath(real, DOC_ROOT),
            "content_head": (content[:maxb] + trim_note),
        }
    except Exception as e:
        logger.error(f"Failed to read file {raw}: {e}")
        return {"error": str(e)}
