"""Serper search tool implementation."""
from typing import Dict, Any
from tools.http_fetch import tool_http_fetch
from core.logging import setup_logger

logger = setup_logger(__name__)


def _prepare_serper_request(args: Dict[str, Any], search_query: str) -> Dict[str, Any]:
    """Prepare Serper API request parameters."""
    prepared = dict(args)
    prepared["url"] = "https://google.serper.dev/search"
    prepared["method"] = "POST"

    headers = dict(prepared.get("headers") or {})
    headers.setdefault("Content-Type", "application/json")
    prepared["headers"] = headers

    body = prepared.get("body")
    if not isinstance(body, dict):
        body = {}
    body["q"] = search_query
    body.setdefault("num", 5)
    prepared["body"] = body

    return prepared


def tool_serper_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute web search using Serper API.

    Args:
        args: Search parameters (query, num)

    Returns:
        Search results from Serper API
    """
    query = args.get("query")
    if not query:
        return {"error": "query가 필요합니다."}

    num = int(args.get("num") or 5)
    prepared = _prepare_serper_request({"body": {"num": num}}, query)
    return tool_http_fetch(prepared)
