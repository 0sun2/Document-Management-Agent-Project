"""Search service for web and vector search."""
import json
import httpx
from typing import List, Dict, Any, Set
from fastapi import HTTPException
from core.config import SERPER_API_KEY, VLLM_MODEL_NAME
from core.logging import setup_logger
from services.llm_service import openai_client

logger = setup_logger(__name__)


async def search_web_serper(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Perform web search using Serper API.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        List of search results with title, link, snippet

    Raises:
        HTTPException: If Serper API request fails
    """
    if not SERPER_API_KEY:
        raise HTTPException(
            status_code=500, detail="SERPER_API_KEY is not configured"
        )

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = getattr(error.response, "text", "") or str(error)
            logger.error(f"Serper API request failed: {detail}")
            raise HTTPException(
                status_code=502,
                detail=f"Serper API request failed: {detail}",
            ) from error
        except httpx.TimeoutException as error:
            logger.error("Serper API request timed out")
            raise HTTPException(
                status_code=504,
                detail="Serper API request timed out",
            ) from error

    data = response.json()

    # Extract organic search results
    organic_results = data.get("organic", [])
    citations = []

    for result in organic_results[:num_results]:
        citations.append(
            {
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
            }
        )

    return citations


def generate_search_queries(
    question: str, doc_summary: str, missing_tokens: Set[str]
) -> List[str]:
    """
    Generate relevant search queries using LLM.

    Args:
        question: User's question
        doc_summary: Summary of document context
        missing_tokens: Keywords missing from documents

    Returns:
        List of search queries (1-3 queries)
    """
    prompt = (
        "다음 질문에 답하기 위해 문서에 없는 정보를 찾아야 합니다.\n"
        f"[질문]\n{question}\n\n"
        f"[문서 요약]\n{doc_summary or '문서 요약 없음'}\n\n"
        "문서에 없는 키워드: "
        + (", ".join(sorted(missing_tokens)) if missing_tokens else "없음")
        + "\n\n"
        "1~3개의 한국어 검색 질의를 JSON 배열로 출력하세요. "
        "예: [\"목포대학교 장학금 제도\", \"2024학년도 장학금 신청 방법\"]"
    )

    try:
        resp = openai_client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 검색 질의를 제안하는 도우미입니다. JSON 배열만 출력하세요.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=256,
        )
        text = resp.choices[0].message.content or "[]"

        # JSON 파싱 시도
        queries = json.loads(text)
        if isinstance(queries, list):
            valid_queries = [str(q) for q in queries if isinstance(q, str)][:3]
            if valid_queries:
                logger.info(f"Generated search queries: {valid_queries}")
                return valid_queries
    except Exception as exc:
        logger.warning(f"Failed to generate search queries: {exc}")

    # Fallback: use the question as-is
    return [question]
