"""
SSE streaming helpers — extended from Week 2 for multimodal outputs.

Events for multimodal QA:
  - retrieval: text + image chunks retrieved
  - answer*: answer tokens
  - citations: text [T*] and image [F*] citations
  - done
"""

import json
from typing import AsyncIterator

from app.services.retriever import TextChunk, ImageChunk, MultimodalRetrievalResult


def _sse_event(event: str, data: object) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


async def multimodal_ask_sse(
    retrieval: MultimodalRetrievalResult,
    answer_stream: AsyncIterator[str],
) -> AsyncIterator[dict]:
    """Generate SSE events for the multimodal /qa/ask endpoint.

    Events: retrieval → answer* → citations → done
    """
    # 1. Retrieval results — both text and image
    retrieval_data = {
        "texts": [
            {
                "ref_label": t.ref_label,
                "doc_id": t.doc_id,
                "filename": t.filename,
                "content_preview": t.content_preview,
                "score": t.score,
            }
            for t in retrieval.texts
        ],
        "images": [
            {
                "ref_label": img.ref_label,
                "image_id": img.image_id,
                "filename": img.filename,
                "caption": img.caption,
                "score": img.score,
            }
            for img in retrieval.images
        ],
    }
    yield _sse_event("retrieval", retrieval_data)

    # 2. Answer tokens
    async for token in answer_stream:
        yield _sse_event("answer", token)

    # 3. Citations
    citations_data = {
        "text_citations": [
            {
                "ref_label": t.ref_label,
                "doc_id": t.doc_id,
                "filename": t.filename,
                "chunk_index": t.chunk_index,
                "content_snippet": t.content[:300],
            }
            for t in retrieval.texts
        ],
        "image_citations": [
            {
                "ref_label": img.ref_label,
                "image_id": img.image_id,
                "filename": img.filename,
                "caption": img.caption,
            }
            for img in retrieval.images
        ],
    }
    yield _sse_event("citations", citations_data)

    # 4. Done
    yield _sse_event("done", {"use_vision": retrieval.has_images})
