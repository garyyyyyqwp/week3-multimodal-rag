from openai import AsyncOpenAI

from app.utils.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    VISION_MODEL,
    VISION_API_KEY,
    VISION_BASE_URL,
)

_text_client: AsyncOpenAI | None = None
_vision_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Get the text LLM client (e.g., glm-4-flash)."""
    global _text_client
    if _text_client is None:
        _text_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _text_client


def get_vision_client() -> AsyncOpenAI:
    """Get the Vision LLM client (e.g., glm-4v)."""
    global _vision_client
    if _vision_client is None:
        _vision_client = AsyncOpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)
    return _vision_client


def get_model() -> str:
    return OPENAI_MODEL


def get_vision_model() -> str:
    return VISION_MODEL
