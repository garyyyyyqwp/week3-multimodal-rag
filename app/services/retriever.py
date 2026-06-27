"""
Multimodal Retriever — joint text + image retrieval with merged results.

Responsible for:
  - Querying both text and image collections in parallel
  - Normalizing and merging results
  - Deciding whether retrieved results warrant Vision LLM usage
"""

import base64
from dataclasses import dataclass, field
from typing import Optional

from app.services.vector_store import get_vector_store, VectorStore
from app.utils.config import RAG_TOP_K, RAG_TOP_M


@dataclass
class TextChunk:
    doc_id: str
    filename: str
    chunk_index: int
    content: str
    content_preview: str
    score: float
    ref_label: str = ""  # e.g., [T1], [T2]


@dataclass
class ImageChunk:
    doc_id: str
    filename: str
    image_id: str
    file_path: str
    caption: str
    score: float
    image_base64: str = ""   # loaded on demand for Vision LLM
    ref_label: str = ""      # e.g., [F1], [F2]


@dataclass
class MultimodalRetrievalResult:
    texts: list[TextChunk]
    images: list[ImageChunk]
    has_images: bool = False


def _load_image_base64(file_path: str) -> str:
    """Load an image file and return its base64 data URL."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    with Image.open(file_path) as pil_img:
        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")
        pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def retrieve_multimodal(
    query: str,
    top_k: int = RAG_TOP_K,
    top_m: int = RAG_TOP_M,
    store: VectorStore | None = None,
    load_images: bool = False,
) -> MultimodalRetrievalResult:
    """Joint multimodal retrieval: text chunks + image chunks.

    Args:
        query: Natural language query
        top_k: Number of text chunks to retrieve
        top_m: Number of image chunks to retrieve
        store: VectorStore instance (uses singleton if None)
        load_images: If True, load image base64 into results (for Vision LLM)

    Returns:
        MultimodalRetrievalResult with both text and image chunks, labeled with
        reference markers ([T1], [T2], ..., [F1], [F2], ...).
    """
    if store is None:
        store = await get_vector_store()

    results = await store.search_multimodal(query, top_k=top_k, top_m=top_m)

    # Build TextChunks
    texts = []
    for i, t in enumerate(results["texts"]):
        texts.append(TextChunk(
            doc_id=t["doc_id"],
            filename=t["filename"],
            chunk_index=t.get("chunk_index", 0),
            content=t["content"],
            content_preview=t["content_preview"],
            score=t["score"],
            ref_label=f"[T{i + 1}]",
        ))

    # Build ImageChunks
    images = []
    for i, img in enumerate(results["images"]):
        b64 = ""
        if load_images and img.get("file_path"):
            try:
                b64 = _load_image_base64(img["file_path"])
            except Exception:
                b64 = ""
        images.append(ImageChunk(
            doc_id=img["doc_id"],
            filename=img["filename"],
            image_id=img.get("image_id", ""),
            file_path=img.get("file_path", ""),
            caption=img.get("caption", ""),
            score=img["score"],
            image_base64=b64,
            ref_label=f"[F{i + 1}]",
        ))

    has_images = len(images) > 0

    return MultimodalRetrievalResult(
        texts=texts,
        images=images,
        has_images=has_images,
    )
