"""Test fixtures and configuration for Week 3 tests."""

import os
import shutil
import tempfile
from pathlib import Path
import uuid

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


class _MockEmbeddings:
    """Return deterministic 1024-d vectors for testing (matching Zhipu embedding-2)."""
    def __init__(self):
        import hashlib

    def __call__(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        import struct
        results = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            # Generate 1024 values from hash
            vec = []
            for i in range(0, 1024):
                seed = int.from_bytes(h[(i * 4) % 32:((i * 4) % 32) + 4], 'big', signed=True)
                vec.append(seed / 2**32 * 0.1)
            results.append(vec)
        return results


_MOCK_EMB = _MockEmbeddings()


@pytest.fixture(scope="function", autouse=True)
def setup_test_env(monkeypatch):
    """Ensure tests use isolated ChromaDB and test config."""
    tmpdir = tempfile.mkdtemp(prefix="chroma_test_")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", tmpdir)
    monkeypatch.setenv("TEXT_COLLECTION_NAME", "test_text_collection")
    monkeypatch.setenv("IMAGE_COLLECTION_NAME", "test_image_collection")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "512")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "50")
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("RAG_TOP_M", "2")
    monkeypatch.setenv("EXPERIMENT_DATA_DIR", f"{tmpdir}/experiments")
    monkeypatch.setenv("IMAGE_SAVE_DIR", f"{tmpdir}/images")

    # Unset real API keys
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test-placeholder")
    monkeypatch.setenv("VISION_API_KEY", "sk-test-placeholder")

    # Reset VectorStore singleton
    import app.services.vector_store as vs
    vs._store = None

    async def _get_vector_store():
        if vs._store is not None:
            return vs._store
        store = vs.VectorStore(
            persist_dir=tmpdir,
            text_collection_name="test_text_collection",
            image_collection_name="test_image_collection",
        )
        vs._store = store
        return store

    monkeypatch.setattr(vs, "get_vector_store", _get_vector_store)

    # Patch consumer modules
    import app.routers.documents as documents_mod
    import app.services.rag_pipeline as rag_mod
    import app.services.retriever as retriever_mod
    import app.services.experiment_runner as er_mod
    monkeypatch.setattr(documents_mod, "get_vector_store", _get_vector_store)
    monkeypatch.setattr(rag_mod, "get_vector_store", _get_vector_store)
    monkeypatch.setattr(retriever_mod, "get_vector_store", _get_vector_store)
    monkeypatch.setattr(er_mod, "EXPERIMENT_DATA_DIR", f"{tmpdir}/experiments")

    # Reset experiment store
    import app.services.experiment_runner as er
    er._experiments.clear()
    er._SAVE_DIR = Path(f"{tmpdir}/experiments")
    er._SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Patch IMAGE_SAVE_DIR in parser too
    import app.services.parser as parser_mod
    monkeypatch.setattr(parser_mod, "IMAGE_SAVE_DIR", f"{tmpdir}/images")

    yield

    # Cleanup
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def mock_embedding(monkeypatch):
    """Replace real embedding with deterministic mock."""
    async def mock_embed_texts(texts: list[str]) -> list[list[float]]:
        return _MOCK_EMB(texts)

    async def mock_embed_single(text: str) -> list[float]:
        return _MOCK_EMB([text])[0]

    # Patch embedding in vector_store since it imports embed_texts directly
    monkeypatch.setattr("app.services.vector_store.embed_texts", mock_embed_texts)
    # Patch embedding module itself
    monkeypatch.setattr("app.services.embedding.embed_texts", mock_embed_texts)
    monkeypatch.setattr("app.services.embedding.embed_single", mock_embed_single)


@pytest.fixture(autouse=True)
def mock_clip(monkeypatch):
    """Replace CLIP encoding with deterministic mock (512-d)."""
    import hashlib

    async def mock_clip_encode_images(image_paths: list[str]) -> list[list[float]]:
        results = []
        for p in image_paths:
            h = hashlib.sha256(p.encode()).digest()
            vec = []
            for i in range(512):
                seed = int.from_bytes(h[(i * 2) % 32:((i * 2) % 32) + 2], 'big', signed=True)
                vec.append(seed / 2**16 * 0.1)
            results.append(vec)
        return results

    async def mock_clip_encode_text(text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(512):
            seed = int.from_bytes(h[(i * 2) % 32:((i * 2) % 32) + 2], 'big', signed=True)
            vec.append(seed / 2**16 * 0.1)
        return vec

    monkeypatch.setattr("app.services.clip_embedding.clip_encode_images", mock_clip_encode_images)
    monkeypatch.setattr("app.services.clip_embedding.clip_encode_text", mock_clip_encode_text)
    # Force stub mode off so our mocks are used
    monkeypatch.setattr("app.services.clip_embedding._is_stub_mode", lambda: False)


@pytest.fixture
def sample_txt_content() -> bytes:
    """Sample TXT content with specific facts for testing."""
    text = """深度学习基础笔记

## 神经网络基础

神经网络由多个神经元层组成。每个神经元接收输入，通过激活函数产生输出。

## 反向传播算法

反向传播算法由Rumelhart等人在1986年提出。该算法利用链式法则计算损失函数相对于每个权重的梯度。

关键步骤：
1. 前向传播：计算网络输出
2. 计算损失：比较预测值与真实值
3. 反向传播误差：从输出层向输入层传播
4. 更新权重：使用梯度下降优化器

张三的生日是1990年5月1日。他在北京工作。

## Transformer架构

Transformer架构由Vaswani等人在2017年提出，核心创新是自注意力机制（Self-Attention）。
它抛弃了传统的RNN结构，完全基于注意力机制处理序列数据。
"""
    return text.encode("utf-8")


@pytest.fixture
def sample_test_cases() -> list[dict]:
    """Sample test cases for experiment testing."""
    return [
        {
            "question": "张三的生日是什么时候？",
            "reference_answer": "张三的生日是1990年5月1日。",
            "has_image": False,
        },
        {
            "question": "反向传播算法是谁提出的？",
            "reference_answer": "反向传播算法由Rumelhart等人在1986年提出。",
            "has_image": False,
        },
        {
            "question": "Transformer架构的核心创新是什么？",
            "reference_answer": "核心创新是自注意力机制（Self-Attention）。",
            "has_image": False,
        },
    ]


@pytest.fixture
def mock_llm_responses(monkeypatch):
    """Mock LLM to return deterministic responses based on question content."""
    from unittest.mock import AsyncMock, MagicMock

    answers_map = {
        "生日": "根据资料[T1]，张三的生日是1990年5月1日。",
        "反向传播": "根据资料[T1]，反向传播算法由Rumelhart等人在1986年提出。",
        "Transformer": "根据资料[T1]，Transformer架构的核心创新是自注意力机制。",
        "自注意力": "根据资料[T1]，Transformer架构的核心创新是自注意力机制。",
    }

    def _find_answer(question: str) -> str:
        for key, ans in answers_map.items():
            if key in question:
                return ans
        return "根据已有资料，我无法完全回答这个问题。"

    async def mock_chat_create(**kwargs):
        import json

        messages = kwargs.get("messages", [])
        user_content = ""
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                user_content += content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_content += part.get("text", "")

        # If tool_choice is set (judge mode), return tool call response
        if kwargs.get("tool_choice") is not None:
            tool_response = MagicMock()
            tool_call = MagicMock()
            tool_call.function.name = "submit_evaluation"
            tool_call.function.arguments = json.dumps({
                "factual_accuracy": 4,
                "image_relevance": 3,
                "completeness": 4,
                "conciseness": 4,
                "reasoning": "回答准确，引用了参考资料中的事实。",
            })
            tool_message = MagicMock()
            tool_message.tool_calls = [tool_call]
            choice = MagicMock()
            choice.message = tool_message
            tool_response.choices = [choice]
            return tool_response

        # Streaming mode: return async generator
        if kwargs.get("stream") is True:
            answer = _find_answer(user_content)

            async def stream():
                for char in answer:
                    delta = MagicMock()
                    delta.content = char
                    choice = MagicMock()
                    choice.delta = delta
                    chunk = MagicMock()
                    chunk.choices = [choice]
                    yield chunk

            return stream()

        # Non-streaming text mode
        answer = _find_answer(user_content)
        message = MagicMock()
        message.content = answer
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    # Create mock client
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=mock_chat_create)

    # Patch both text and vision clients at the llm module level
    monkeypatch.setattr("app.services.llm._text_client", mock_client)
    monkeypatch.setattr("app.services.llm._vision_client", mock_client)
    monkeypatch.setattr("app.services.llm.get_client", lambda: mock_client)
    monkeypatch.setattr("app.services.llm.get_vision_client", lambda: mock_client)

    # Also mock the judge's tool call response
    return mock_client


@pytest.fixture
def mock_judge_response(monkeypatch):
    """Mock the LLM judge to return structured scores."""
    import json
    from unittest.mock import AsyncMock, MagicMock

    async def mock_judge_create(**kwargs):
        # Return a tool call with deterministic scores
        message = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "submit_evaluation"
        tool_call.function.arguments = json.dumps({
            "factual_accuracy": 4,
            "image_relevance": 3,
            "completeness": 4,
            "conciseness": 4,
            "reasoning": "回答准确，引用了参考资料中的事实。图文关联度中等因为问题未涉及图片。",
        })
        message.tool_calls = [tool_call]
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        return response

    mock_llm = MagicMock()
    mock_llm.chat.completions.create = AsyncMock(side_effect=mock_judge_create)
    monkeypatch.setattr("app.services.llm.get_client", lambda: mock_llm)

    return mock_llm


@pytest_asyncio.fixture
async def test_app():
    """Create a TestClient for the FastAPI app."""
    from main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
