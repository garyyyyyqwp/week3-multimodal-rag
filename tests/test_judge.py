"""Tests for LLM-as-Judge scoring service."""

import asyncio
import pytest
from app.services.judge import (
    JudgeVerdict,
    JudgeResult,
    judge_answer,
    judge_batch,
    JUDGE_SYSTEM_PROMPT,
)


def test_judge_verdict_model():
    """JudgeVerdict should enforce 1-5 range on all dimensions."""
    # Valid
    v = JudgeVerdict(
        factual_accuracy=4,
        image_relevance=3,
        completeness=5,
        conciseness=2,
        reasoning="测试评分依据",
    )
    assert v.factual_accuracy == 4
    assert v.image_relevance == 3


def test_judge_verdict_invalid_range():
    """JudgeVerdict should reject values outside 1-5."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        JudgeVerdict(
            factual_accuracy=0,  # invalid: <1
            image_relevance=3,
            completeness=3,
            conciseness=3,
            reasoning="",
        )

    with pytest.raises(ValidationError):
        JudgeVerdict(
            factual_accuracy=6,  # invalid: >5
            image_relevance=3,
            completeness=3,
            conciseness=3,
            reasoning="",
        )


def test_judge_result_weighted_score():
    """JudgeResult should compute weighted total correctly."""
    result = JudgeResult(
        factual_accuracy=4,
        image_relevance=3,
        completeness=4,
        conciseness=4,
        reasoning="测试",
    )
    expected = round(4 * 0.35 + 3 * 0.25 + 4 * 0.25 + 4 * 0.15, 2)
    assert result.total_score == expected
    assert 1.0 <= result.total_score <= 5.0


@pytest.mark.asyncio
async def test_judge_answer_with_mock(mock_judge_response):
    """Judge should return structured scores from LLM tool call."""
    result = await judge_answer(
        question="测试问题",
        answer="测试回答",
        reference_answer="参考答案",
        context="标准上下文",
        has_images=False,
    )
    assert isinstance(result, JudgeResult)
    assert result.factual_accuracy == 4
    assert result.image_relevance == 3
    assert result.completeness == 4
    assert result.conciseness == 4
    assert len(result.reasoning) > 0
    assert 1.0 <= result.total_score <= 5.0


@pytest.mark.asyncio
async def test_judge_batch_concurrent(mock_judge_response):
    """Judge batch should process multiple evaluations concurrently."""
    items = [
        {
            "question": f"问题{i}",
            "answer": f"回答{i}",
            "reference_answer": f"答案{i}",
            "context": f"上下文{i}",
            "has_images": i % 2 == 0,
        }
        for i in range(5)
    ]

    results = await judge_batch(items, concurrency=3)
    assert len(results) == 5
    for r in results:
        assert isinstance(r, JudgeResult)
        assert 1 <= r.factual_accuracy <= 5
        assert 1 <= r.image_relevance <= 5
