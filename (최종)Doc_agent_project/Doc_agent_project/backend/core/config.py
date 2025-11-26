"""Application configuration management."""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "storage"
DOCS_DIR = DATA_DIR / "originals"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# DOC_ROOT for tool access
DOC_ROOT = os.path.abspath(str(DOCS_DIR))

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# vLLM configuration
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:9000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "qwen3-4b-ft")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.2"))
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "1024"))

# CORS configuration
DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

ALLOWED_ORIGINS: List[str] = list(
    filter(
        None,
        [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",")],
    )
) or DEFAULT_ALLOWED_ORIGINS

# Embedding model configuration
EMBEDDER_NAME = os.getenv("EMBEDDER_MODEL", "intfloat/multilingual-e5-large-instruct")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# Search configuration
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.72"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
DOC_ONLY_THRESHOLD = float(os.getenv("DOC_ONLY_THRESHOLD", "0.85"))
WEB_FALLBACK_THRESHOLD = float(os.getenv("WEB_FALLBACK_THRESHOLD", "0.65"))
QUESTION_COVERAGE_THRESHOLD = float(os.getenv("QUESTION_COVERAGE_THRESHOLD", "0.6"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "6"))
MAX_FINAL_PROMPT_CHUNKS = int(os.getenv("MAX_FINAL_PROMPT_CHUNKS", "6"))
KEYWORD_BOOST = float(os.getenv("RAG_KEYWORD_BOOST", "0.03"))

# FastAPI configuration
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

# System prompts
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
