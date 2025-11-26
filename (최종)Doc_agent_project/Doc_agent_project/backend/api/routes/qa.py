"""Question answering API routes."""
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException

from models.schemas import AskPayload, AskResponse
from models.database import get_supabase_client
from services.embedding_service import embed_query
from services.document_service import (
    compute_match_score,
    _tokenize_text,
    filter_stopwords,
)
from services.llm_service import query_vllm, build_prompt
from services.search_service import search_web_serper, generate_search_queries
from core.config import (
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    DOC_ONLY_THRESHOLD,
    MAX_CONTEXT_CHUNKS,
    KEYWORD_BOOST,
    QUESTION_COVERAGE_THRESHOLD,
    WEB_FALLBACK_THRESHOLD,
)
from core.logging import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskPayload) -> AskResponse:
    """
    Answer question using RAG + optional web search.

    This endpoint:
    1. Performs vector search in Supabase
    2. Evaluates document relevance and coverage
    3. Optionally performs web search if needed
    4. Generates answer using LLM

    Args:
        payload: Question and search parameters

    Returns:
        Answer with citations and metadata
    """
    sb = get_supabase_client()
    matches: List[Dict[str, Any]] = []
    source = "db"
    web_citations = []
    used_web_fallback = False
    web_search_error: str | None = None
    requested_top_k = max(1, min(20, int(payload.top_k or DEFAULT_TOP_K)))

    # 1. Vector search in Supabase
    rag_rows = []
    if sb is not None:
        qvec = embed_query(payload.question)[0].tolist()

        params = {
            "query_embedding": qvec,
            "match_count": requested_top_k,
            "filename_filter": None,
            "doc_id_filter": None,
        }

        try:
            res = sb.rpc("match_rag_docs", params).execute()
            rag_rows = res.data or []
        except Exception as e:
            logger.error(f"Supabase RAG search failed: {e}")
            rag_rows = []

    # 2. Enhance scores with keyword matching
    question_tokens = set(_tokenize_text(payload.question))
    top_similarity = 0.0
    if rag_rows:
        for row in rag_rows:
            row["similarity"] = float(row.get("similarity") or 0.0)
            top_similarity = max(top_similarity, row["similarity"])
            row["score"] = compute_match_score(row, question_tokens, KEYWORD_BOOST)
        rag_rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    # Select top chunks
    selected_rows = rag_rows[:requested_top_k] if rag_rows else []
    context_rows = selected_rows[:MAX_CONTEXT_CHUNKS]
    context_chunks = [
        r.get("content", "") for r in context_rows if r.get("content")
    ]

    # Build match results
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

    # 3. Determine if web search is needed based on similarity and coverage
    doc_all_tokens: set = set()
    for row in context_rows:
        doc_all_tokens.update(_tokenize_text(row.get("content", "")))

    # Filter out stopwords for more accurate coverage calculation
    filtered_question_tokens = filter_stopwords(question_tokens)
    filtered_doc_tokens = filter_stopwords(doc_all_tokens)

    # Calculate coverage based on meaningful tokens only
    matched_tokens = filtered_question_tokens.intersection(filtered_doc_tokens)
    missing_tokens = filtered_question_tokens - filtered_doc_tokens

    # If all question tokens were stopwords, use original tokens
    if not filtered_question_tokens:
        coverage_ratio = 1.0  # All meta words, 100% coverage
        doc_has_question_terms = bool(question_tokens.intersection(doc_all_tokens))
    else:
        coverage_ratio = len(matched_tokens) / len(filtered_question_tokens)
        doc_has_question_terms = len(matched_tokens) > 0

    # Determine if document alone is sufficient (high confidence)
    doc_only_candidate = (
        bool(context_chunks)
        and doc_has_question_terms
        and coverage_ratio >= QUESTION_COVERAGE_THRESHOLD
        and top_similarity >= DOC_ONLY_THRESHOLD
    )

    similarity_fallback_threshold = max(WEB_FALLBACK_THRESHOLD, SIMILARITY_THRESHOLD)

    if not context_chunks:
        use_web_search = True
        logger.info("No documents found - web search required")
    elif doc_only_candidate:
        use_web_search = False
        logger.info(
            f"Docs sufficient (similarity {top_similarity:.3f}, "
            f"coverage {coverage_ratio:.2f}) - using documents only"
        )
    elif coverage_ratio < QUESTION_COVERAGE_THRESHOLD and bool(missing_tokens):
        use_web_search = True
        logger.info(
            f"Low coverage ({coverage_ratio:.2f}) and missing tokens - "
            f"web search needed"
        )
    elif top_similarity < similarity_fallback_threshold:
        use_web_search = True
        logger.info(
            f"Similarity below fallback threshold "
            f"({top_similarity:.3f} < {similarity_fallback_threshold}) - adding web search"
        )
    else:
        use_web_search = False
        logger.info(
            f"Acceptable similarity ({top_similarity:.3f}) - using documents only"
        )

    # 4. Generate document summary for better search query generation
    doc_summary = ""
    if context_chunks and use_web_search:
        try:
            summary_prompt = (
                "아래 문서 조각을 웹 검색 질의에 활용할 수 있도록 3문장 이내로 요약하고 "
                "핵심 키워드를 쉼표로 나열해라.\n"
                f"[질문]\n{payload.question}\n"
                f"[문서 조각]\n{'\n\n'.join(context_chunks[:3])}"
            )
            doc_summary = await query_vllm(summary_prompt)
            logger.info("Generated concise document summary for search query generation")
        except Exception as e:
            logger.warning(f"Failed to generate document summary: {e}")

    # 5. Perform web search if needed
    if use_web_search:
        logger.info(
            f"Web search needed. similarity={top_similarity:.3f}, "
            f"coverage={coverage_ratio:.2f}, missing={missing_tokens}"
        )
        try:
            # Generate smart search queries using LLM
            search_queries = generate_search_queries(
                payload.question, doc_summary, missing_tokens
            )
            logger.info(f"Search queries: {search_queries}")

            # Perform web search with the first (best) query
            if search_queries:
                web_citations = await search_web_serper(
                    search_queries[0], num_results=5
                )
                used_web_fallback = True
                source = "web" if not context_chunks else "hybrid"
        except Exception as e:
            web_search_error = str(e)
            logger.warning(f"Web search failed: {e}")
    else:
        logger.info(
            f"Document-based answer. similarity={top_similarity:.3f}, "
            f"coverage={coverage_ratio:.2f}"
        )
        source = "db"

    # 6. Generate answer using LLM
    try:
        prompt = build_prompt(payload.question, context_chunks, web_citations or None)
        answer = await query_vllm(prompt)
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise HTTPException(
            status_code=502, detail=f"Failed to generate answer: {str(e)}"
        )

    return AskResponse(
        answer=answer,
        matches=matches,
        citations=web_citations,
        used_web_fallback=used_web_fallback,
        source=source,
        top_similarity=round(top_similarity, 4),
        web_search_error=web_search_error,
    )
