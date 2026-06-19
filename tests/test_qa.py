"""Tests for multimodal QA endpoint (with mocked LLM)."""

import pytest


@pytest.mark.asyncio
async def test_qa_ask_sync_returns_answer(
    test_app, sample_txt_content, mock_llm_responses
):
    """Upload a doc with known facts, ask about them, verify multimodal answer."""
    # Upload
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", sample_txt_content, "text/plain")},
    )
    assert resp.status_code == 201

    # Ask via sync endpoint
    resp = await test_app.post(
        "/api/v1/qa/ask/sync",
        json={
            "question": "张三的生日是什么时候？",
            "top_k": 3,
            "top_m": 2,
            "strategy": "basic",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "生日" in data["answer"] or "1990" in data["answer"]
    assert isinstance(data["text_citations"], list)
    assert isinstance(data["image_citations"], list)
    assert isinstance(data["retrieved_texts"], list)
    assert isinstance(data["retrieved_images"], list)
    assert isinstance(data["use_vision"], bool)


@pytest.mark.asyncio
async def test_qa_invalid_strategy_rejected(test_app, sample_txt_content):
    """Invalid prompt strategy should return 400."""
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", sample_txt_content, "text/plain")},
    )
    assert resp.status_code == 201

    resp = await test_app.post(
        "/api/v1/qa/ask/sync",
        json={
            "question": "test question",
            "strategy": "invalid_strategy",
        },
    )
    assert resp.status_code == 400
    assert "无效" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_qa_empty_question_rejected(test_app):
    """Empty question should be rejected with 422."""
    resp = await test_app.post(
        "/api/v1/qa/ask/sync",
        json={"question": "", "top_k": 3},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_qa_no_documents_returns_empty_retrieval(test_app, mock_llm_responses):
    """Asking with no indexed documents should return empty retrieval."""
    resp = await test_app.post(
        "/api/v1/qa/ask/sync",
        json={"question": "什么是深度学习？", "top_k": 3},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["retrieved_texts"] == []
    assert data["retrieved_images"] == []
    assert "answer" in data


@pytest.mark.asyncio
async def test_qa_all_four_strategies_work(
    test_app, sample_txt_content, mock_llm_responses
):
    """All 4 prompt strategies should work on the same question."""
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", sample_txt_content, "text/plain")},
    )
    assert resp.status_code == 201

    for strategy in ["basic", "fewshot", "cot", "multimodal_step"]:
        resp = await test_app.post(
            "/api/v1/qa/ask/sync",
            json={
                "question": "Transformer架构的核心创新是什么？",
                "strategy": strategy,
            },
        )
        assert resp.status_code == 200, f"Strategy {strategy} failed"
        data = resp.json()
        assert len(data["answer"]) > 0, f"Strategy {strategy} returned empty answer"
