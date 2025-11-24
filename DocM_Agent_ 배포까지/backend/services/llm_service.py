"""LLM service for text generation."""
import httpx
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from openai import OpenAI
from core.config import (
    VLLM_BASE_URL,
    VLLM_MODEL_NAME,
    VLLM_API_KEY,
    VLLM_TEMPERATURE,
    VLLM_MAX_TOKENS,
)
from core.logging import setup_logger

logger = setup_logger(__name__)

# OpenAI client for tool usage
openai_client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
logger.info(f"OpenAI client initialized: {VLLM_BASE_URL}")


async def query_vllm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Query vLLM server for text generation.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt

    Returns:
        Generated text response

    Raises:
        HTTPException: If vLLM request fails
    """
    default_system = (
        "You are a helpful Korean assistant that answers strictly "
        "from the provided context. If context is insufficient, say so."
    )

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt or default_system,
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": VLLM_TEMPERATURE,
        "max_tokens": VLLM_MAX_TOKENS,
    }

    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}

    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            response = await client.post(
                f"{VLLM_BASE_URL}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = getattr(error.response, "text", "") or str(error)
            logger.error(f"vLLM request failed: {detail}")
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


def build_prompt(
    question: str,
    chunks: List[str],
    web_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build LLM prompt with document chunks and optional web results.

    Args:
        question: User question
        chunks: Document text chunks
        web_results: Optional web search results

    Returns:
        Formatted prompt string
    """
    context_parts = []

    if chunks:
        context_text = "\n\n".join(
            f"[문서 {idx + 1}] {chunk}" for idx, chunk in enumerate(chunks)
        )
        context_parts.append(f"[문서 컨텍스트]\n{context_text}")

    if web_results:
        web_text = "\n\n".join(
            f"[웹 {idx + 1}] {result.get('title', '')}\n"
            f"출처: {result.get('url', '')}\n"
            f"내용: {result.get('snippet', '')}"
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
