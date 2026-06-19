"""
CLIP multimodal embedding — encodes images and text into a shared vector space.

Uses OpenAI's CLIP ViT-B/32 via HuggingFace Transformers at runtime,
with a fallback to a stub for environments without torch installed.
"""

from typing import Sequence

import numpy as np
from PIL import Image

from app.utils.config import CLIP_MODEL_NAME, CLIP_DEVICE

# Lazy-loaded CLIP model
_model = None
_processor = None
_stub_mode: bool | None = None  # True = fallback, False = real model


class CLIPEmbeddingError(Exception):
    """Raised when CLIP embedding generation fails."""
    pass


def _is_stub_mode() -> bool:
    """Check whether CLIP is available, caching the result."""
    global _stub_mode
    if _stub_mode is not None:
        return _stub_mode
    try:
        import torch  # noqa: F401
        from transformers import CLIPModel, CLIPProcessor  # noqa: F401
        _stub_mode = False
    except ImportError:
        _stub_mode = True
    return _stub_mode


def _load_clip():
    """Lazy-load the CLIP model and processor."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    if _is_stub_mode():
        raise CLIPEmbeddingError(
            "CLIP 模型不可用：缺少 torch 或 transformers 依赖。"
            "请运行: pip install torch transformers"
        )

    from transformers import CLIPModel, CLIPProcessor

    _model = CLIPModel.from_pretrained(f"openai/{CLIP_MODEL_NAME}")
    _processor = CLIPProcessor.from_pretrained(f"openai/{CLIP_MODEL_NAME}")
    _model.to(CLIP_DEVICE)
    _model.eval()
    return _model, _processor


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


async def clip_encode_images(image_paths: list[str]) -> list[list[float]]:
    """Encode image files to CLIP embedding vectors.

    Args:
        image_paths: List of local file paths to image files.

    Returns:
        List of embedding vectors (512-d for ViT-B/32).
    """
    if not image_paths:
        return []

    if _is_stub_mode():
        # Stub: return random 512-d unit vectors for offline testing
        rng = np.random.RandomState(hash(image_paths[0]) % (2**31))
        return [rng.randn(512).astype(np.float64).tolist() for _ in image_paths]

    model, processor = _load_clip()

    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(CLIP_DEVICE) for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    embeddings = image_features.cpu().numpy().tolist()
    for img in images:
        img.close()
    return embeddings


async def clip_encode_text(text: str) -> list[float]:
    """Encode a text query to CLIP embedding vector.

    Args:
        text: The text to encode (e.g., a search query).

    Returns:
        512-d embedding vector (for ViT-B/32).
    """
    if _is_stub_mode():
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(512).astype(np.float64).tolist()

    model, processor = _load_clip()

    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(CLIP_DEVICE) for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy().flatten().tolist()


async def clip_encode_image_batch(
    images: list[Image.Image],
) -> list[list[float]]:
    """Encode a list of PIL Images directly (no disk read)."""
    if not images:
        return []

    if _is_stub_mode():
        rng = np.random.RandomState(42)
        return [rng.randn(512).astype(np.float64).tolist() for _ in images]

    model, processor = _load_clip()

    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(CLIP_DEVICE) for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().tolist()
