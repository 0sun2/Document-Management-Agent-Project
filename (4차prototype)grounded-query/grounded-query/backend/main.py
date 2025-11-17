import os
import shutil
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import json
import glob
import pathlib
import mimetypes
import base64
import re
from datetime import datetime

import httpx
import requests
import pdfplumber
from docx import Document as DocxDocument
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from openai import OpenAI

# .env 파일 로드
load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "storage"
DOCS_DIR = DATA_DIR / "originals"

DOCS_DIR.mkdir(parents=True, exist_ok=True)

# DOC_ROOT for tool access (file reading/glob)
DOC_ROOT = os.path.abspath(str(DOCS_DIR))

DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

# Supabase 설정
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# vLLM 설정
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:9000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "qwen3-4b-ft")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

# CORS 설정
ALLOWED_ORIGINS = list(
    filter(
        None,
        [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",")],
    )
) or DEFAULT_ALLOWED_ORIGINS

# 임베딩 모델 설정
EMBEDDER_NAME = os.getenv("EMBEDDER_MODEL", "intfloat/multilingual-e5-large-instruct")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# 웹 검색 및 유사도 설정
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.72"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
DOC_ONLY_THRESHOLD = float(os.getenv("DOC_ONLY_THRESHOLD", "0.85"))
WEB_FALLBACK_THRESHOLD = float(os.getenv("WEB_FALLBACK_THRESHOLD", "0.65"))
QUESTION_COVERAGE_THRESHOLD = float(os.getenv("QUESTION_COVERAGE_THRESHOLD", "0.6"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
MAX_FINAL_PROMPT_CHUNKS = int(os.getenv("MAX_FINAL_PROMPT_CHUNKS", "6"))
KEYWORD_BOOST = float(os.getenv("RAG_KEYWORD_BOOST", "0.03"))

app = FastAPI(title="RAG FastAPI backend for grounded-query")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase 클라이언트 초기화
sb: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[INFO] Supabase 초기화 성공")
    except Exception as e:
        print(f"[WARN] Supabase 초기화 실패: {e}")
else:
    print("[WARN] Supabase 미설정: .env에 SUPABASE_URL / SUPABASE_ANON_KEY 추가 필요")

# 임베더 초기화
embedder = SentenceTransformer(EMBEDDER_NAME)
print(f"[INFO] 임베딩 모델 로드 완료: {EMBEDDER_NAME}")

# OpenAI 클라이언트 초기화 (Tool 사용을 위해)
openai_client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
print(f"[INFO] OpenAI 클라이언트 초기화 완료: {VLLM_BASE_URL}")


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
    size_bytes: int
    uploaded_at: datetime


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    size_bytes: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: List[DocumentSummary]


class AskResponse(BaseModel):
    answer: str
    matches: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    used_web_fallback: bool
    source: str
    top_similarity: float


class AgentPayload(BaseModel):
    question: str = Field(..., description="User question for agent")


class AgentResponse(BaseModel):
    answer: str
    tool_calls_made: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOLS 정의 및 실행 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "http_fetch",
            "description": (
                "HTTP로 외부 데이터를 가져온다. 웹 검색이 필요할 때는 반드시 Serper API를 사용해야 한다. "
                "절대 직접 웹사이트 URL을 호출하지 말고, Serper API를 사용하라.\n"
                "Serper API 사용법:\n"
                "- URL: https://google.serper.dev/search (반드시 이 URL을 사용)\n"
                "- Method: POST\n"
                "- Headers: {\"X-API-KEY\": \"your_api_key\", \"Content-Type\": \"application/json\"}\n"
                "- Body: {\"q\": \"검색어\", \"num\": 5}\n"
                "예시: 검색어가 '목포대학교 장학금'이면 body에 {\"q\": \"목포대학교 장학금\", \"num\": 5}를 넣어라."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "enum": ["GET", "POST"], "default": "GET"},
                    "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                    "params": {"type": "object", "additionalProperties": {"type": ["string", "number", "boolean"]}},
                    "body": {"type": ["string", "object", "array"]},
                    "timeout": {"type": "number", "default": 20}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "serper_search",
            "description": "Serper API를 사용해 웹 검색을 수행한다. query에 한국어 검색어를 넣어라.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_glob",
            "description": "DOC_ROOT 내부에서 패턴으로 파일을 찾는다. 예: **/*.txt, *.md 등",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "limit": {"type": "integer", "default": 50}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_read",
            "description": "DOC_ROOT 내부 파일을 읽어 앞부분을 반환한다. 텍스트/Markdown/PDF 지원.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer", "default": 200000}
                },
                "required": ["path"]
            }
        }
    }
]


def _safe_resolve_user_path(path: str) -> str:
    """DOC_ROOT 내부로 제한하는 안전한 경로 해결"""
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = pathlib.Path(DOC_ROOT) / p
    real = os.path.abspath(os.path.realpath(str(p)))
    root = os.path.abspath(DOC_ROOT)
    if not (real == root or real.startswith(root + os.sep)):
        raise ValueError("Path outside of DOC_ROOT is not allowed")
    return real


def _tokenize_text(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[0-9A-Za-z가-힣]+", (text or "").lower())
        if len(token) >= 2
    ]


def _extract_keywords_from_filename(filename: str) -> List[str]:
    stem = pathlib.Path(filename).stem
    return _tokenize_text(stem)


_DOC_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_doc_metadata(doc_id: str) -> Optional[Dict[str, Any]]:
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
        print(f"[WARN] Failed to load metadata for {doc_id}: {exc}")
        return cached


def _save_doc_metadata(doc_id: str, filename: str, uploaded_at: datetime) -> None:
    meta = {
        "doc_id": doc_id,
        "filename": filename,
        "uploaded_at": uploaded_at.isoformat(),
        "keywords": _extract_keywords_from_filename(filename),
    }
    meta_path = DOCS_DIR / doc_id / "meta.json"
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Failed to write metadata for {doc_id}: {exc}")
    _DOC_METADATA_CACHE[doc_id] = meta


def _compute_match_score(row: Dict[str, Any], question_token_set: set) -> float:
    base = float(row.get("similarity") or 0.0)
    doc_tokens: set = set()
    metadata = _load_doc_metadata(row.get("doc_id", ""))
    if metadata:
        doc_tokens.update(metadata.get("keywords") or [])
    doc_tokens.update(_extract_keywords_from_filename(row.get("filename", "")))
    overlap = len(doc_tokens.intersection(question_token_set))
    content_tokens = set(_tokenize_text(row.get("content", "")))
    lexical_overlap = len(content_tokens.intersection(question_token_set))
    overlap += lexical_overlap
    boost = min(overlap, 3) * KEYWORD_BOOST
    return base + boost


def _build_search_query(question: str, missing_tokens: set) -> str:
    if missing_tokens:
        tokens_str = " ".join(sorted(missing_tokens))
        return f"{tokens_str} 장학금 제도"
    return question


def _prepare_serper_request(args: Dict[str, Any], search_query: str) -> Dict[str, Any]:
    prepared = dict(args)
    prepared["url"] = "https://google.serper.dev/search"
    prepared["method"] = "POST"
    headers = dict(prepared.get("headers") or {})
    if SERPER_API_KEY and "X-API-KEY" not in headers:
        headers["X-API-KEY"] = SERPER_API_KEY
    elif "X-API-KEY" not in headers:
        headers["X-API-KEY"] = "your_api_key"
    headers.setdefault("Content-Type", "application/json")
    prepared["headers"] = headers
    body = prepared.get("body")
    if not isinstance(body, dict):
        body = {}
    body["q"] = search_query
    body.setdefault("num", 5)
    prepared["body"] = body
    return prepared


def _generate_search_queries(question: str, doc_summary: str, missing_tokens: set) -> List[str]:
    prompt = (
        "다음 질문에 답하기 위해 문서에 없는 정보를 찾아야 합니다.\n"
        f"[질문]\n{question}\n\n"
        f"[문서 요약]\n{doc_summary or '문서 요약 없음'}\n\n"
        "문서에 없는 키워드: "
        + (", ".join(sorted(missing_tokens)) if missing_tokens else "없음")
        + "\n\n"
        "1~3개의 한국어 검색 질의를 JSON 배열로 출력하세요. 예: [\"목포대학교 장학금 제도\"]"
    )
    try:
        resp = openai_client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "당신은 검색 질의를 제안하는 도우미입니다. JSON 배열만 출력하세요."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=256,
        )
        text = resp.choices[0].message.content or "[]"
        queries = json.loads(text)
        if isinstance(queries, list):
            return [str(q) for q in queries if isinstance(q, str)][:3]
    except Exception as exc:
        print(f"[WARN] 검색 질의 생성 실패: {exc}")
    return [question]


def _get_doc_file_metadata(doc_id: str) -> Tuple[int, datetime]:
    """문서 파일의 크기와 최종 수정 시각을 반환"""
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


def _read_text_file(path: str, max_bytes: int) -> str:
    """텍스트 파일 읽기 (다양한 인코딩 시도)"""
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    for enc in ("utf-8", "utf-16", "cp949", "euc-kr"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return f"[binary head b64] {base64.b64encode(data[:4096]).decode('ascii')}"


def _read_pdf_file(path: str, max_bytes: int) -> str:
    """PDF 파일 읽기"""
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


def tool_http_fetch(args: Dict[str, Any]) -> Dict[str, Any]:
    """HTTP 요청 실행 (Serper API 지원)"""
    url = args.get("url")
    if not url:
        return {"error": "URL이 제공되지 않았습니다."}
    
    method = (args.get("method") or "GET").upper()
    headers = args.get("headers") or {}
    body = args.get("body")
    params = args.get("params") or None
    timeout = float(args.get("timeout") or 20)
    
    # Serper API 호출인 경우 API 키 자동 추가
    if url and "google.serper.dev" in url and SERPER_API_KEY:
        if "X-API-KEY" not in headers:
            headers["X-API-KEY"] = SERPER_API_KEY
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        print(f"[DEBUG] Serper API 호출: {url}")
    
    try:
        if method == "POST":
            if isinstance(body, (dict, list)):
                r = requests.post(url, headers=headers, json=body, params=params, timeout=timeout, allow_redirects=True)
            else:
                r = requests.post(url, headers=headers, data=body, params=params, timeout=timeout, allow_redirects=True)
        else:
            r = requests.get(url, headers=headers, params=params, timeout=timeout, allow_redirects=True)

        ctype = (r.headers.get("content-type") or "").lower()
        out: Dict[str, Any] = {"status": r.status_code, "content_type": r.headers.get("content-type"), "url": r.url}
        
        print(f"[DEBUG] HTTP 응답: status={r.status_code}, url={r.url}")
        
        if "application/json" in ctype:
            try:
                out["json"] = r.json()
            except Exception:
                out["text_head"] = r.text[:4000]
        elif any(bin_kw in ctype for bin_kw in ("application/octet-stream", "application/pdf", "image/", "audio/", "video/")):
            out["binary_head_b64"] = base64.b64encode(r.content[:4096]).decode("ascii")
        else:
            out["text_head"] = r.text[:4000]
        return out
    except Exception as e:
        print(f"[ERROR] HTTP 요청 실패: {e}")
        return {"error": str(e)}


def tool_serper_search(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query")
    if not query:
        return {"error": "query가 필요합니다."}
    num = int(args.get("num") or 5)
    prepared = _prepare_serper_request({"body": {"num": num}}, query)
    return tool_http_fetch(prepared)


def tool_filesystem_glob(args: Dict[str, Any]) -> Dict[str, Any]:
    """파일 패턴 검색"""
    pattern = args.get("pattern") or "**/*"
    limit = int(args.get("limit") or 50)
    base = pathlib.Path(DOC_ROOT)

    # 절대경로 검증/차단
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

    # 결과도 DOC_ROOT 밖은 이중 차단
    rels = []
    for m in matches:
        real = os.path.abspath(os.path.realpath(m))
        root = os.path.abspath(DOC_ROOT)
        if not (real == root or real.startswith(root + os.sep)):
            continue
        rels.append(os.path.relpath(real, start=DOC_ROOT))

    return {"root": DOC_ROOT, "matches": rels}


def tool_filesystem_read(args: Dict[str, Any]) -> Dict[str, Any]:
    """파일 내용 읽기"""
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
        return {"path": os.path.relpath(real, DOC_ROOT), "content_head": (content[:maxb] + trim_note)}
    except Exception as e:
        return {"error": str(e)}


TOOL_EXEC = {
    "http_fetch": tool_http_fetch,
    "serper_search": tool_serper_search,
    "filesystem_glob": tool_filesystem_glob,
    "filesystem_read": tool_filesystem_read,
}


def _parse_tool_calls_from_text(text: str) -> Optional[List[dict]]:
    """<tool_call>...</tool_call> 텍스트 파싱 (백업용)"""
    calls = []
    if not text:
        return None
    
    # 완전한 <tool_call>...</tool_call> 형식 파싱
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text or "", flags=re.DOTALL):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
            name = obj.get("name")
            args = obj.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            calls.append({
                "id": f"toolcall_{len(calls)+1}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
        except Exception as e:
            print(f"[DEBUG] tool_call 파싱 실패: {e}, raw: {raw[:100]}...")
            continue
    
    # 닫는 태그가 없는 경우도 처리 (<tool_call>로 시작하는 경우)
    if not calls and "<tool_call>" in text:
        # <tool_call> 다음부터 JSON 추출 시도
        start_idx = text.find("<tool_call>")
        if start_idx >= 0:
            json_start = text.find("{", start_idx)
            if json_start >= 0:
                # JSON 끝 찾기 (중괄호 매칭)
                brace_count = 0
                json_end = json_start
                for i in range(json_start, len(text)):
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    try:
                        raw = text[json_start:json_end]
                        obj = json.loads(raw)
                        name = obj.get("name")
                        args = obj.get("arguments", {})
                        if not isinstance(args, str):
                            args = json.dumps(args, ensure_ascii=False)
                        calls.append({
                            "id": "toolcall_1",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        })
                        print(f"[DEBUG] 불완전한 tool_call 파싱 성공: {name}")
                    except Exception as e:
                        print(f"[DEBUG] 불완전한 tool_call 파싱 실패: {e}")
    
    return calls or None


def _as_tool_calls_dict_list(tool_calls) -> List[dict]:
    """SDK 객체/딕셔너리를 OpenAI 포맷 dict 리스트로 표준화"""
    out: List[dict] = []
    if not tool_calls:
        return out
    for idx, tc in enumerate(tool_calls, 1):
        if isinstance(tc, dict):
            out.append(tc)
        else:
            out.append({
                "id": getattr(tc, "id", f"toolcall_{idx}"),
                "type": "function",
                "function": {
                    "name": getattr(getattr(tc, "function", None), "name", None),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", "{}") or "{}",
                },
            })
    return out


def _append_and_maybe_parse_tool_text(messages: List[Dict[str, Any]], msg) -> None:
    """메시지를 파싱하고 tool_calls 표준화"""
    content = (msg.content or "")
    tool_calls = getattr(msg, "tool_calls", None)

    # native tool_calls → dict 표준화
    tool_calls = _as_tool_calls_dict_list(tool_calls)

    # 없으면 <tool_call> 백업 파서 시도
    if (not tool_calls) and ("<tool_call>" in content):
        parsed = _parse_tool_calls_from_text(content)
        if parsed:
            tool_calls = parsed
            content = ""

    # while 조건이 감지할 수 있도록 msg에도 반영
    try:
        msg.tool_calls = tool_calls if tool_calls else None
        msg.content = content
    except Exception:
        pass

    messages.append({
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls if tool_calls else None
    })


AGENT_SYSTEM_MSG = (
    "당신은 문서 기반 도구 사용 에이전트다. 반드시 필요한 경우 filesystem_glob 또는 "
    "filesystem_read를 사용해서 파일 내용을 실제로 읽고 답해야 한다.\n"
    f"- 파일 접근은 DOC_ROOT 내부에서만: {DOC_ROOT}\n"
    "- 사용자가 파일명을 말하면 반드시 filesystem_read(path='파일명')를 호출해 내용을 읽고 요약하라.\n"
    "- 일반 HTTP 호출은 http_fetch 사용. JSON이면 핵심 필드만 요약.\n"
    "- 문서에 없는 내용을 질문받으면 serper_search tool을 사용해서 Serper API로 웹 검색을 수행하라. query에 한국어 검색어만 넣어라.\n"
    "- 파일 요약을 요청받으면 **순수한 한국어 텍스트 요약만** 출력한다. 태그/JSON/메타정보/코드블록/마크다운 제목을 쓰지 않는다.\n"
)

RAG_SYSTEM_MSG = (
    "당신은 문서 우선 질문 답변 시스템이다. 항상 제공된 문서 컨텍스트를 최우선으로 사용하라.\n"
    "- 문서에 관련 정보가 조금이라도 있으면 반드시 활용하고, 부족한 부분만 웹 검색으로 보강하라.\n"
    "- 웹 검색이 필요하면 serper_search tool을 사용한다. http_fetch로 직접 검색 URL을 호출하지 마라.\n"
    "- serper_search tool은 query 필드에 한국어 검색어만 넣으면 된다.\n"
    "- 웹 검색 결과를 이용할 때는 출처를 명시하고, 문서 정보와 조화롭게 통합해 답변하라.\n"
    "- 모든 관련 정보를 종합하여 자세하고 검증 가능한 답변을 제공하라."
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 임베딩 유틸 (E5 instruct 포맷 + L2 정규화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _normalize(v: np.ndarray) -> np.ndarray:
    """L2 정규화"""
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def embed_passages(texts: List[str]) -> np.ndarray:
    """문서 청크 임베딩 (E5 instruct 포맷: "passage:")"""
    prepped = [f"passage: {t}" for t in texts]
    vecs = embedder.encode(prepped, convert_to_numpy=True, normalize_embeddings=False)
    return _normalize(vecs).astype(np.float32)


def embed_query(q: str) -> np.ndarray:
    """쿼리 임베딩 (E5 instruct 포맷: "query:")"""
    prepped = [f"query: {q}"]
    vec = embedder.encode(prepped, convert_to_numpy=True, normalize_embeddings=False)
    return _normalize(vec).astype(np.float32)


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


async def search_web_serper(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Serper API를 사용하여 웹 검색 수행

    Returns:
        List of search results with title, link, snippet
    """
    if not SERPER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="SERPER_API_KEY is not configured"
        )

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": num_results
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = getattr(error.response, "text", "") or str(error)
            raise HTTPException(
                status_code=502,
                detail=f"Serper API request failed: {detail}",
            ) from error
        except httpx.TimeoutException as error:
            raise HTTPException(
                status_code=504,
                detail="Serper API request timed out",
            ) from error

    data = response.json()

    # organic 검색 결과 추출
    organic_results = data.get("organic", [])
    citations = []

    for result in organic_results[:num_results]:
        citations.append({
            "title": result.get("title", ""),
            "url": result.get("link", ""),
            "snippet": result.get("snippet", "")
        })

    return citations


def build_prompt(question: str, chunks: List[str], web_results: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    LLM 프롬프트 생성 - 문서 청크와 선택적으로 웹 검색 결과 포함
    """
    # 문서 컨텍스트 구성
    context_parts = []

    if chunks:
        context_text = "\n\n".join(
            f"[문서 {idx + 1}] {chunk}" for idx, chunk in enumerate(chunks)
        )
        context_parts.append(f"[문서 컨텍스트]\n{context_text}")

    # 웹 검색 결과 추가
    if web_results:
        web_text = "\n\n".join(
            f"[웹 {idx + 1}] {result.get('title', '')}\n출처: {result.get('url', '')}\n내용: {result.get('snippet', '')}"
            for idx, result in enumerate(web_results)
        )
        context_parts.append(f"[웹 검색 결과]\n{web_text}")

    full_context = "\n\n".join(context_parts)

    return (
        "아래는 사용자의 질문과 관련 정보입니다.\n\n"
        f"{full_context}\n\n"
        f"[질문]\n{question}\n\n"
        "위 정보를 바탕으로 한국어로 정확하게 답변해 주세요. "
        "제공된 정보만 사용하고, 정보가 부족하면 솔직하게 말하세요. "
        "웹 검색 결과를 사용한 경우 출처를 명시하세요."
    )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents(q: Optional[str] = Query(default=None, alias="q")) -> DocumentListResponse:
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
                size_bytes, uploaded_at = _get_doc_file_metadata(info["doc_id"])
                meta = _load_doc_metadata(info["doc_id"])
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
            print(f"[WARN] Supabase 문서 조회 실패, 로컬 폴백 사용: {e}")

    if not documents:
        # Supabase 미설정이거나 실패한 경우 로컬 저장소 스캔
        for doc_dir in DOCS_DIR.iterdir():
            if not doc_dir.is_dir():
                continue
            files = [f for f in doc_dir.iterdir() if f.is_file()]
            if not files:
                continue
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            stat = latest_file.stat()
            meta = _load_doc_metadata(doc_dir.name)
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
            meta = _load_doc_metadata(doc.doc_id)
            keywords = set(meta.get("keywords", []) if meta else [])
            keywords.update(_extract_keywords_from_filename(doc.filename))
            if keywords.intersection(search_tokens):
                filtered_docs.append(doc)
        if filtered_docs:
            documents = filtered_docs

    documents.sort(key=lambda d: d.filename.lower())
    return DocumentListResponse(documents=documents)


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    original_filename: Optional[str] = Form(None),
) -> UploadResponse:
    if sb is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase 미설정: .env에 SUPABASE_URL / SUPABASE_ANON_KEY를 추가하세요."
        )

    # 프론트엔드에서 전달한 UUID 사용, 없으면 새로 생성
    doc_id = document_id or str(uuid4())
    filename = original_filename or file.filename

    # 파일 저장 (선택사항, 로컬에 백업하고 싶은 경우)
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

    # 텍스트 추출
    text = extract_text_from_file(destination)
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="텍스트를 추출하지 못했습니다."
        )

    # RecursiveCharacterTextSplitter로 청킹 (원래 프로젝트와 동일)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_text(text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="청크가 비었습니다."
        )

    # 임베딩 생성 (E5 instruct 포맷)
    vecs = embed_passages(chunks)

    # Supabase에 저장
    rows = []
    for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
        rows.append({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "content": chunk,
            "embedding": vec.tolist(),
        })

    try:
        sb.table("rag_docs").insert(rows).execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Supabase 저장 실패: {str(e)}"
        )

    _save_doc_metadata(doc_id, filename, upload_time)

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunks=len(chunks),
        size_bytes=size_bytes,
        uploaded_at=upload_time,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskPayload) -> AskResponse:
    """
    Tool calling 방식으로 RAG + 웹 검색 수행
    1) Supabase RAG 검색
    2) RAG 결과를 컨텍스트로 제공하고, 모델이 필요시 tool 호출로 웹 검색
    3) LLM(Qwen@vLLM)이 tool을 사용하여 최종 답변 생성
    """
    matches: List[Dict[str, Any]] = []
    source = "db"
    web_citations = []
    used_web_fallback = False

    # 1. Supabase RAG 검색
    rag_rows = []
    if sb is not None:
        top_k = max(1, min(20, int(payload.top_k or DEFAULT_TOP_K)))
        qvec = embed_query(payload.question)[0].tolist()

        params = {
            "query_embedding": qvec,
            "match_count": top_k,
            "filename_filter": None,
            "doc_id_filter": None,
        }

        try:
            res = sb.rpc("match_rag_docs", params).execute()
            rag_rows = res.data or []
        except Exception as e:
            print(f"[ERROR] Supabase RAG 검색 실패: {e}")
            rag_rows = []

    question_tokens = set(_tokenize_text(payload.question))
    if rag_rows:
        for row in rag_rows:
            row["similarity"] = float(row.get("similarity") or 0.0)
            row["score"] = _compute_match_score(row, question_tokens)
        rag_rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    selected_rows = rag_rows[:DEFAULT_TOP_K] if rag_rows else []
    context_rows = selected_rows[:MAX_CONTEXT_CHUNKS]
    context_chunks = [r.get("content", "") for r in context_rows if r.get("content")]
    doc_all_tokens: set = set()
    for row in context_rows:
        doc_all_tokens.update(_tokenize_text(row.get("content", "")))

    matches = [
        {
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "chunk_index": r["chunk_index"],
            "similarity": r.get("similarity"),
            "chunk_text": r.get("content", ""),
        }
        for r in selected_rows
    ]

    # 유사도 통계
    top_similarity = float(selected_rows[0].get("similarity", 0.0)) if selected_rows else 0.0

    # 유사도가 낮거나 문서가 없으면 tool calling으로 웹 검색 유도
    matched_tokens = question_tokens.intersection(doc_all_tokens)
    missing_tokens = question_tokens - doc_all_tokens
    search_query_override = _build_search_query(payload.question, missing_tokens)
    coverage_ratio = len(matched_tokens) / max(1, len(question_tokens))
    doc_has_question_terms = len(matched_tokens) > 0
    doc_only_candidate = (
        bool(context_chunks)
        and doc_has_question_terms
        and coverage_ratio >= QUESTION_COVERAGE_THRESHOLD
        and top_similarity >= DOC_ONLY_THRESHOLD
    )
    require_web_search = (not context_chunks) or bool(missing_tokens)
    doc_confident = doc_only_candidate and not missing_tokens
    use_web_search = require_web_search
    
    if use_web_search:
        print(f"[DEBUG] 웹 검색 필요. 유사도={top_similarity:.3f}, coverage={coverage_ratio:.2f}")
        source = "web" if not context_chunks else "hybrid"
    else:
        print(f"[DEBUG] 문서 기반 답변. 유사도={top_similarity:.3f}, coverage={coverage_ratio:.2f}")
        source = "db"

    # 사전 문서 요약
    doc_summary_text = ""
    if context_chunks:
        try:
            doc_summary_prompt = build_prompt(
                payload.question,
                context_chunks[:MAX_FINAL_PROMPT_CHUNKS]
            )
            doc_summary_text = (await query_vllm(doc_summary_prompt)).strip()
        except Exception as exc:
            print(f"[WARN] 문서 요약 생성 실패: {exc}")

    search_queries: List[str] = []
    if require_web_search:
        search_queries = _generate_search_queries(payload.question, doc_summary_text, missing_tokens)

    if doc_summary_text and doc_confident and not use_web_search:
        return AskResponse(
            answer=doc_summary_text,
            matches=matches,
            citations=[],
            used_web_fallback=False,
            source="db",
            top_similarity=round(top_similarity, 4),
        )
    
    # 3. Tool calling 방식으로 LLM 호출
    doc_filenames = {row.get("filename") for row in context_rows if row.get("filename")}
    doc_summary_line = ""
    if doc_filenames:
        doc_summary_line = "이전 단계에서 다음 문서들을 참조했습니다: " + ", ".join(sorted(doc_filenames))
    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                RAG_SYSTEM_MSG
                + ("\n" + doc_summary_line if doc_summary_line else "")
                + (
                    "\n현재 문서에는 다음 키워드가 빠져 있습니다: "
                    + ", ".join(sorted(missing_tokens))
                    if missing_tokens else ""
                )
            )
        },
    ]
    
    # 컨텍스트 구성
    if context_chunks:
        context_text = "\n\n".join(
            f"[문서 {idx + 1}] {chunk}" for idx, chunk in enumerate(context_chunks)
        )
        if use_web_search:
            # 유사도가 낮으면 tool calling으로 웹 검색 유도
            messages.append({
                "role": "user",
                "content": (
                    f"[문서 컨텍스트]\n{context_text}\n\n"
                    f"[질문]\n{payload.question}\n\n"
                    "위 문서 내용을 먼저 요약하고 활용하여 초안을 만든 뒤, 반드시 serper_search tool을 사용해 Serper API로 추가 정보를 검색하고, "
                    "웹 결과는 문서 정보와 함께 통합해서 설명하세요."
                    + (
                        "\n특히 문서에 없는 키워드: "
                        + ", ".join(sorted(missing_tokens))
                        + " 에 대한 정보를 검색하세요."
                        if missing_tokens else ""
                    )
                )
            })
        else:
            # 유사도가 충분하면 문서 기반 답변
            # 하지만 모델이 정보가 없다고 판단하면 tool을 호출하도록 안내
            messages.append({
                "role": "user",
                "content": (
                    f"[문서 컨텍스트]\n{context_text}\n\n"
                    f"[질문]\n{payload.question}\n\n"
                    "위 문서를 바탕으로 자세하고 완전하게 답변하세요. "
                    "만약 문서에 관련 정보가 부족하다고 판단되면, serper_search tool을 사용해서 Serper API로 웹 검색을 수행하세요."
                )
            })
    else:
        # 문서가 없으면 tool calling으로 웹 검색 유도
        messages.append({
            "role": "user",
            "content": f"[질문]\n{payload.question}\n\n문서가 없으므로 serper_search tool을 사용해서 Serper API로 웹 검색을 수행하세요."
        })
    if doc_summary_text:
        messages.append({
            "role": "assistant",
            "content": f"<doc_summary>\n{doc_summary_text}\n</doc_summary>"
        })
        messages.append({
            "role": "user",
            "content": (
                "위 문서 요약을 최우선으로 활용해 답변하세요. "
                + (
                    "문서에 없는 키워드("
                    + ", ".join(sorted(missing_tokens))
                    + ")에 대한 정보가 필요하므로 반드시 serper_search tool을 호출해 보완하세요."
                    if missing_tokens else
                    "추가 정보가 명확히 필요할 때만 serper_search tool을 호출하세요."
                )
            )
        })

    # 4. Tool calling 루프로 LLM 답변 생성
    steps = 0
    max_steps = 6
    query_queue = list(search_queries)
    
    while steps < max_steps:
        steps += 1
        
        # LLM 호출 (항상 tool 사용 가능)
        try:
            resp = openai_client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=2048,
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"vLLM request failed: {str(e)}"
            )
        
        msg = resp.choices[0].message
        _append_and_maybe_parse_tool_text(messages, msg)
        
        # Tool 호출이 없으면 답변 완료
        # 하지만 모델이 "문서에 없다"고 판단하면 tool 호출 유도
        if not getattr(msg, "tool_calls", None):
            content_lower = (msg.content or "").lower()
            # 모델이 "문서에 없다", "정보가 없다", "포함되어 있지 않다"고 답변했는지 확인
            no_info_keywords = ["포함되어 있지 않", "정보가 없", "문서에 없", "찾을 수 없", "답변할 수 없", "알 수 없", "제공할 수 없"]
            has_no_info = any(keyword in content_lower for keyword in no_info_keywords)
            
            print(f"[DEBUG] Step {steps}: Tool 호출 없음, content: {msg.content[:100] if msg.content else 'None'}...")
            print(f"[DEBUG] 정보 부족 감지: {has_no_info}, use_web_search: {use_web_search}")
            
            if (require_web_search or has_no_info) and steps == 1:
                # 유사도가 낮거나 정보가 없는데 tool을 호출하지 않았으면 다시 유도
                print("[DEBUG] 정보 부족 감지 → tool 호출 재유도")
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                })
                messages.append({
                    "role": "user",
                    "content": "문서에 없는 내용이므로 반드시 serper_search tool을 사용해서 Serper API로 웹 검색을 수행해야 합니다. tool을 호출하세요."
                })
                continue
            break
        else:
            print(f"[DEBUG] Step {steps}: Tool 호출 감지됨: {len(msg.tool_calls)}개")
        
        # Tool 실행
        next_messages = []
        for call in msg.tool_calls:
            if isinstance(call, dict):
                name = call["function"]["name"]
                args_str = call["function"]["arguments"]
                call_id = call.get("id") or f"toolcall_{steps}"
            else:
                name = call.function.name
                args_str = call.function.arguments or "{}"
                call_id = call.id

            try:
                args = json.loads(args_str or "{}")
            except Exception:
                args = {}

            search_query_for_tool = query_queue.pop(0) if query_queue else search_query_override

            print(f"[DEBUG] Tool 실행: {name}, args: {json.dumps(args, ensure_ascii=False)[:200]}...")

            if name == "serper_search":
                args = dict(args)
                if not args.get("query"):
                    args["query"] = search_query_for_tool
                args["num"] = int(args.get("num") or 5)
                result = tool_serper_search(args)
                used_web_fallback = True
                source = "web" if not context_chunks else "hybrid"
                if isinstance(result, dict) and "json" in result:
                    serper_data = result["json"]
                    organic = serper_data.get("organic", [])
                    print(f"[DEBUG] serper_search 결과: {len(organic)}개")
                    for item in organic[:5]:
                        web_citations.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                        })
                next_messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
                continue
            
            # http_fetch tool인 경우, URL을 먼저 체크해서 Serper API로 변환
            tool_url = args.get("url", "") if name == "http_fetch" else ""
            
            # Serper API가 아닌 직접 URL 호출인 경우, 자동으로 Serper API로 변환
            # Google 검색 URL이나 다른 웹사이트 URL을 호출하는 경우도 감지
            is_web_search_attempt = (
                name == "http_fetch" and
                tool_url and 
                "google.serper.dev" not in tool_url and
                (
                    "google.com/search" in tool_url or
                    "search" in tool_url.lower() or
                    any(domain in tool_url for domain in [".ac.kr", ".edu", ".org", ".com"])
                )
            )
            
            if is_web_search_attempt:
                # 모델이 잘못된 URL을 호출했을 때, 질문을 추출해서 Serper API로 재호출
                print(f"[WARN] 모델이 Serper API 대신 직접 URL 호출: {tool_url}")
                print(f"[DEBUG] 질문에서 검색어 추출하여 Serper API로 재호출...")
                
                # URL에서 검색어 추출 시도 (Google 검색 URL인 경우)
                search_query = search_query_for_tool
                if "google.com/search" in tool_url:
                    try:
                        from urllib.parse import urlparse, parse_qs
                        parsed = urlparse(tool_url)
                        params = parse_qs(parsed.query)
                        if "q" in params:
                            search_query = params["q"][0]
                            print(f"[DEBUG] URL에서 검색어 추출: {search_query}")
                    except Exception as e:
                        print(f"[DEBUG] URL 파싱 실패, 질문 사용: {e}")
                
                # Serper API로 재호출
                serper_args = _prepare_serper_request({}, search_query)
                
                print(f"[DEBUG] Serper API로 재호출: {search_query}")
                result = tool_http_fetch(serper_args)
                tool_url = "https://google.serper.dev/search"  # URL 업데이트
            else:
                # 정상적인 tool 실행
                if name == "http_fetch" and tool_url and "google.serper.dev" in tool_url:
                    args = _prepare_serper_request(args, search_query_for_tool)
                    tool_url = "https://google.serper.dev/search"
                result = TOOL_EXEC.get(name, lambda a: {"error": f"unknown tool: {name}"})(args)
            
            if name == "http_fetch" and tool_url and "google.serper.dev" in tool_url:
                used_web_fallback = True
                source = "web" if not context_chunks else "hybrid"
                print(f"[DEBUG] Serper API 응답 처리 중...")
                try:
                    if isinstance(result, dict):
                        if "error" in result:
                            print(f"[WARN] Tool 실행 에러: {result['error']}")
                        elif "json" in result:
                            serper_data = result["json"]
                            organic = serper_data.get("organic", [])
                            print(f"[DEBUG] Serper 검색 결과: {len(organic)}개")
                            for item in organic[:5]:
                                web_citations.append({
                                    "title": item.get("title", ""),
                                    "url": item.get("link", ""),
                                    "snippet": item.get("snippet", ""),
                                })
                            print(f"[DEBUG] Tool calling으로 웹 검색 완료: {len(web_citations)}개 결과")
                        else:
                            print(f"[WARN] Serper 응답에 json이 없음: {list(result.keys())}")
                except Exception as e:
                    print(f"[WARN] Serper 응답 파싱 실패: {e}")
                    import traceback
                    traceback.print_exc()

            next_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

        messages.extend(next_messages)

    # 최종 답변 추출
    final_text = msg.content or ""
    if not final_text:
        final_text = "도구 실행이 완료되었지만 모델 응답이 비었습니다."

    # 문서 + 웹 하이브리드 요약 (문서 우선)
    if used_web_fallback and (context_chunks or doc_summary_text):
        prompt_chunks = context_chunks[:MAX_FINAL_PROMPT_CHUNKS]
        if doc_summary_text:
            prompt_chunks = [doc_summary_text] + prompt_chunks
        web_results_for_prompt = web_citations[:5] if web_citations else None
        try:
            hybrid_prompt = (
                build_prompt(payload.question, prompt_chunks, web_results_for_prompt)
                + "\n\n[지시]\n"
                "1. 문서에서 확인된 내용은 먼저 요약하고,\n"
                "2. 웹 검색에서 찾은 다른 기관의 정보를 비교 대상으로 명확히 정리하며,\n"
                "3. 두 제도의 차이와 공통점이 드러나도록 표나 bullet로 정리하세요.\n"
                "4. 문서에 없는 정보라도 웹 검색에 있으면 반드시 포함하세요."
                + (
                    "\n5. 특히 문서에 없는 키워드("
                    + ", ".join(sorted(missing_tokens))
                    + ")에 대한 정보를 웹 결과에서 찾아 비교하세요."
                    if missing_tokens else ""
                )
            )
            combined_answer = await query_vllm(hybrid_prompt)
            if combined_answer:
                final_text = combined_answer.strip()
                print("[DEBUG] 문서+웹 하이브리드 요약으로 답변 재생성")
        except Exception as e:
            print(f"[WARN] 하이브리드 요약 생성 실패: {e}")

    # 4. 응답 반환
    return AskResponse(
        answer=final_text,
        matches=matches,
        citations=web_citations,
        used_web_fallback=used_web_fallback,
        source=source,
        top_similarity=round(top_similarity, 4),
    )


@app.post("/agent", response_model=AgentResponse)
async def agent_query(payload: AgentPayload) -> AgentResponse:
    """
    Tool을 사용하는 에이전트 엔드포인트
    파일 읽기, HTTP 요청 등 다양한 도구 활용 가능
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM_MSG},
        {"role": "user", "content": payload.question},
    ]

    # 첫 번째 LLM 호출
    try:
        resp = openai_client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"vLLM agent request failed: {str(e)}"
        )

    msg = resp.choices[0].message
    _append_and_maybe_parse_tool_text(messages, msg)

    # 최대 6번까지 tool 실행 반복
    steps = 0
    total_tool_calls = 0
    while getattr(msg, "tool_calls", None) and steps < 6:
        steps += 1

        # 툴 실행
        next_messages = []
        for call in msg.tool_calls:
            # call은 객체(OpenAI SDK) 또는 dict일 수 있음
            if isinstance(call, dict):
                name = call["function"]["name"]
                args_str = call["function"]["arguments"]
                call_id = call.get("id") or f"toolcall_{steps}"
            else:
                name = call.function.name
                args_str = call.function.arguments or "{}"
                call_id = call.id

            try:
                args = json.loads(args_str or "{}")
            except Exception:
                args = {}

            result = TOOL_EXEC.get(name, lambda a: {"error": f"unknown tool: {name}"})(args)
            total_tool_calls += 1

            next_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

        messages.extend(next_messages)

        # 후속 호출
        try:
            resp = openai_client.chat.completions.create(
                model=VLLM_MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"vLLM agent follow-up request failed: {str(e)}"
            )

        msg = resp.choices[0].message
        _append_and_maybe_parse_tool_text(messages, msg)

        if not getattr(msg, "tool_calls", None):
            break

    final_text = msg.content or ""
    if not final_text:
        final_text = "도구 실행이 완료되었지만 모델 응답이 비었습니다. 같은 요청을 한 번 더 시도하거나, '결과를 요약해줘'라고 물어봐 주세요."

    return AgentResponse(
        answer=final_text,
        tool_calls_made=total_tool_calls
    )


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> Dict[str, str]:
    if sb is not None:
        try:
            # Supabase에서 해당 doc_id를 가진 모든 청크 삭제
            sb.table("rag_docs").delete().eq("doc_id", doc_id).execute()
        except Exception as e:
            print(f"[WARN] Supabase 삭제 실패: {e}")

    # 로컬 파일도 삭제
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
