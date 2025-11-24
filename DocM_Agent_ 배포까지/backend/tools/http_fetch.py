"""HTTP fetch tool implementation."""
import base64
from typing import Dict, Any
import requests
from core.config import SERPER_API_KEY
from core.logging import setup_logger

logger = setup_logger(__name__)


def tool_http_fetch(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute HTTP request with Serper API support.

    Args:
        args: Request parameters (url, method, headers, body, params, timeout)

    Returns:
        Response dictionary with status, content_type, and body
    """
    url = args.get("url")
    if not url:
        return {"error": "URL이 제공되지 않았습니다."}

    method = (args.get("method") or "GET").upper()
    headers = args.get("headers") or {}
    body = args.get("body")
    params = args.get("params") or None
    timeout = float(args.get("timeout") or 20)

    # Auto-add Serper API key if calling Serper
    if url and "google.serper.dev" in url and SERPER_API_KEY:
        if "X-API-KEY" not in headers:
            headers["X-API-KEY"] = SERPER_API_KEY
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        logger.info(f"Serper API call: {url}")

    try:
        if method == "POST":
            if isinstance(body, (dict, list)):
                r = requests.post(
                    url,
                    headers=headers,
                    json=body,
                    params=params,
                    timeout=timeout,
                    allow_redirects=True,
                )
            else:
                r = requests.post(
                    url,
                    headers=headers,
                    data=body,
                    params=params,
                    timeout=timeout,
                    allow_redirects=True,
                )
        else:
            r = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout,
                allow_redirects=True,
            )

        ctype = (r.headers.get("content-type") or "").lower()
        out: Dict[str, Any] = {
            "status": r.status_code,
            "content_type": r.headers.get("content-type"),
            "url": r.url,
        }

        logger.info(f"HTTP response: status={r.status_code}, url={r.url}")

        if "application/json" in ctype:
            try:
                out["json"] = r.json()
            except Exception:
                out["text_head"] = r.text[:4000]
        elif any(
            bin_kw in ctype
            for bin_kw in (
                "application/octet-stream",
                "application/pdf",
                "image/",
                "audio/",
                "video/",
            )
        ):
            out["binary_head_b64"] = base64.b64encode(r.content[:4096]).decode("ascii")
        else:
            out["text_head"] = r.text[:4000]
        return out
    except Exception as e:
        logger.error(f"HTTP request failed: {e}")
        return {"error": str(e)}
