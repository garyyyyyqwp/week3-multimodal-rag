from openai import AsyncOpenAI, APIError, AuthenticationError, RateLimitError

from app.utils.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
)

_client: AsyncOpenAI | None = None


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


def _get_embedding_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
    return _client


def get_embedding_dimension() -> int:
    """Return the expected embedding dimension for the configured model."""
    dims = {
        "embedding-2": 1024,
        "embedding-3": 1024,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    return dims.get(EMBEDDING_MODEL, 1024)


_EMBEDDING_BATCH_SIZE: int = 8


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings.

    Automatically splits large batches to stay within API limits.
    """
    client = _get_embedding_client()
    all_embeddings: list[list[float]] = []

    # Process in batches to avoid API token/size limits
    total_batches = (len(texts) + _EMBEDDING_BATCH_SIZE - 1) // _EMBEDDING_BATCH_SIZE
    for i in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + _EMBEDDING_BATCH_SIZE]
        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
        except AuthenticationError as e:
            raise EmbeddingError(
                f"Embedding API 认证失败，请检查 EMBEDDING_API_KEY 是否正确。"
                f"当前 Base URL: {EMBEDDING_BASE_URL}"
            ) from e
        except RateLimitError as e:
            detail = ""
            if hasattr(e, "body") and isinstance(e.body, dict):
                detail = e.body.get("error", {}).get("message", "")
            msg = detail or str(e)
            if "余额" in msg or "资源包" in msg:
                raise EmbeddingError(
                    f"智谱 API 余额不足（{msg}）。"
                    f"请到 https://open.bigmodel.cn 充值后再试。"
                ) from e
            raise EmbeddingError(
                f"Embedding API 请求频率过高（{msg}），请稍后重试。"
            ) from e
        except APIError as e:
            status_code = getattr(e, "status_code", None)
            if status_code == 404:
                raise EmbeddingError(
                    f"Embedding 模型 '{EMBEDDING_MODEL}' 在当前 API ({EMBEDDING_BASE_URL}) 上不可用。"
                    f"请确认模型名称是否正确，或更换 EMBEDDING_BASE_URL。"
                ) from e
            else:
                raise EmbeddingError(
                    f"Embedding API 调用失败 (HTTP {status_code}): {str(e)}"
                ) from e
        except Exception as e:
            raise EmbeddingError(
                f"Embedding 生成失败: {str(e)}。请检查 EMBEDDING_BASE_URL ({EMBEDDING_BASE_URL}) 和网络连接。"
            ) from e

        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


async def embed_single(text: str) -> list[float]:
    """Generate embedding for a single text string."""
    results = await embed_texts([text])
    return results[0]
