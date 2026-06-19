"""QA Router — multimodal RAG endpoint with SSE streaming."""

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.schemas.qa import (
    QuestionRequest,
    AskResponse,
    TextCitation,
    ImageCitation,
    RetrievedChunkPreview,
    ImageChunkPreview,
)
from app.services.retriever import retrieve_multimodal
from app.services.rag_pipeline import (
    generate_answer_stream,
    build_full_context,
    PROMPT_STRATEGIES,
)
from app.services.streaming import multimodal_ask_sse
from app.services.embedding import EmbeddingError

router = APIRouter(tags=["qa"])


@router.post("/ask")
async def ask_question(request: QuestionRequest):
    """Multimodal QA with SSE streaming — supports text + image retrieval.

    Automatically routes to Vision LLM (glm-4.6v-flash) when relevant images are found,
    otherwise uses Text LLM (glm-4-flash).

    Request body: {question, top_k, top_m, strategy, force_vision}
    """
    # Validate strategy
    if request.strategy not in PROMPT_STRATEGIES:
        valid = ", ".join(sorted(PROMPT_STRATEGIES.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"无效的 Prompt 策略: '{request.strategy}'。可选: {valid}",
        )

    try:
        retrieval = await retrieve_multimodal(
            query=request.question,
            top_k=request.top_k,
            top_m=request.top_m,
            load_images=True,
        )
    except EmbeddingError as e:
        raise HTTPException(status_code=503, detail=str(e))

    answer_stream = generate_answer_stream(
        question=request.question,
        retrieval=retrieval,
        strategy=request.strategy,
        use_vision=request.force_vision,
    )

    return EventSourceResponse(multimodal_ask_sse(retrieval, answer_stream))


@router.post("/ask/sync", response_model=AskResponse)
async def ask_question_sync(request: QuestionRequest):
    """Multimodal QA with non-streaming response (for testing)."""
    if request.strategy not in PROMPT_STRATEGIES:
        valid = ", ".join(sorted(PROMPT_STRATEGIES.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"无效的 Prompt 策略: '{request.strategy}'。可选: {valid}",
        )

    try:
        retrieval = await retrieve_multimodal(
            query=request.question,
            top_k=request.top_k,
            top_m=request.top_m,
            load_images=True,
        )
    except EmbeddingError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Collect full answer
    answer_parts = []
    async for token in generate_answer_stream(
        question=request.question,
        retrieval=retrieval,
        strategy=request.strategy,
        use_vision=request.force_vision,
    ):
        answer_parts.append(token)
    answer = "".join(answer_parts)

    return AskResponse(
        answer=answer,
        text_citations=[
            TextCitation(
                ref_label=t.ref_label,
                doc_id=t.doc_id,
                filename=t.filename,
                chunk_index=t.chunk_index,
                content_snippet=t.content[:300],
            )
            for t in retrieval.texts
        ],
        image_citations=[
            ImageCitation(
                ref_label=img.ref_label,
                image_id=img.image_id,
                filename=img.filename,
                caption=img.caption,
            )
            for img in retrieval.images
        ],
        retrieved_texts=[
            RetrievedChunkPreview(
                ref_label=t.ref_label,
                doc_id=t.doc_id,
                filename=t.filename,
                content_preview=t.content_preview,
                score=t.score,
            )
            for t in retrieval.texts
        ],
        retrieved_images=[
            ImageChunkPreview(
                ref_label=img.ref_label,
                image_id=img.image_id,
                filename=img.filename,
                caption=img.caption,
                score=img.score,
            )
            for img in retrieval.images
        ],
        use_vision=retrieval.has_images,
    )
