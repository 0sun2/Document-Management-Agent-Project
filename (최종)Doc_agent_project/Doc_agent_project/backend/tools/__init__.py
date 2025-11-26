"""Tool implementations for agent."""
from typing import Dict, Any, Callable

from tools.http_fetch import tool_http_fetch
from tools.serper_search import tool_serper_search
from tools.filesystem import tool_filesystem_glob, tool_filesystem_read

# Tool registry
TOOL_EXEC: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "http_fetch": tool_http_fetch,
    "serper_search": tool_serper_search,
    "filesystem_glob": tool_filesystem_glob,
    "filesystem_read": tool_filesystem_read,
}

# Tool definitions for OpenAI format
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
                '- Headers: {"X-API-KEY": "your_api_key", "Content-Type": "application/json"}\n'
                '- Body: {"q": "검색어", "num": 5}\n'
                "예시: 검색어가 '목포대학교 장학금'이면 body에 {\"q\": \"목포대학교 장학금\", \"num\": 5}를 넣어라."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "default": "GET",
                    },
                    "headers": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "params": {
                        "type": "object",
                        "additionalProperties": {
                            "type": ["string", "number", "boolean"]
                        },
                    },
                    "body": {"type": ["string", "object", "array"]},
                    "timeout": {"type": "number", "default": 20},
                },
                "required": ["url"],
            },
        },
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
                    "num": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
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
                    "limit": {"type": "integer", "default": 50},
                },
                "required": ["pattern"],
            },
        },
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
                    "max_bytes": {"type": "integer", "default": 200000},
                },
                "required": ["path"],
            },
        },
    },
]

__all__ = ["TOOL_EXEC", "TOOLS"]
