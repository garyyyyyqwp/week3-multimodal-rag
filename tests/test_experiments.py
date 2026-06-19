"""Tests for experiment CRUD, execution, and reporting."""

import pytest


@pytest.mark.asyncio
async def test_create_experiment(test_app, sample_test_cases):
    """Create an experiment with 4 strategies and 3 test cases."""
    resp = await test_app.post(
        "/api/v1/experiments",
        json={
            "name": "测试实验 - 策略对比",
            "prompt_strategies": ["basic", "fewshot", "cot", "multimodal_step"],
            "test_cases": sample_test_cases,
            "concurrency_limit": 3,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "测试实验 - 策略对比"
    assert len(data["strategies"]) == 4
    assert data["test_case_count"] == 3
    assert data["status"] == "created"
    assert len(data["experiment_id"]) > 0


@pytest.mark.asyncio
async def test_list_experiments(test_app, sample_test_cases):
    """Create an experiment, then list all experiments."""
    resp = await test_app.post(
        "/api/v1/experiments",
        json={
            "name": "列表测试实验",
            "prompt_strategies": ["basic", "cot"],
            "test_cases": sample_test_cases[:2],
        },
    )
    assert resp.status_code == 201
    exp_id = resp.json()["experiment_id"]

    resp = await test_app.get("/api/v1/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert any(e["experiment_id"] == exp_id for e in data["experiments"])


@pytest.mark.asyncio
async def test_get_experiment_detail(test_app, sample_test_cases):
    """Get experiment detail by ID."""
    resp = await test_app.post(
        "/api/v1/experiments",
        json={
            "name": "详情测试实验",
            "prompt_strategies": ["basic"],
            "test_cases": [sample_test_cases[0]],
        },
    )
    exp_id = resp.json()["experiment_id"]

    resp = await test_app.get(f"/api/v1/experiments/{exp_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["experiment_id"] == exp_id
    assert data["name"] == "详情测试实验"


@pytest.mark.asyncio
async def test_run_experiment_and_get_report(
    test_app, sample_txt_content, sample_test_cases,
    mock_llm_responses, mock_judge_response,
):
    """Full flow: upload doc → create experiment → run → get report."""
    # 1. Upload a document first
    resp = await test_app.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", sample_txt_content, "text/plain")},
    )
    assert resp.status_code == 201

    # 2. Create experiment
    resp = await test_app.post(
        "/api/v1/experiments",
        json={
            "name": "端到端测试实验",
            "prompt_strategies": ["basic", "cot"],
            "test_cases": sample_test_cases[:2],
            "concurrency_limit": 2,
        },
    )
    assert resp.status_code == 201
    exp_id = resp.json()["experiment_id"]

    # 3. Run experiment
    resp = await test_app.post(f"/api/v1/experiments/{exp_id}/run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["total_results"] == 4  # 2 strategies × 2 test cases

    # 4. Get detail with results
    resp = await test_app.get(f"/api/v1/experiments/{exp_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 4
    for r in data["results"]:
        assert r["judge_result"] is not None
        assert 1 <= r["judge_result"]["factual_accuracy"] <= 5
        assert 1 <= r["judge_result"]["total_score"] <= 5
        assert len(r["answer"]) > 0
        assert r["error"] == ""

    # 5. Get report
    resp = await test_app.get(f"/api/v1/experiments/{exp_id}/report")
    assert resp.status_code == 200
    report_data = resp.json()
    assert report_data["format"] == "markdown"
    assert "端到端测试实验" in report_data["report"]
    assert "事实准确性" in report_data["report"]
    assert "Token 消耗" in report_data["report"]


@pytest.mark.asyncio
async def test_delete_experiment(test_app, sample_test_cases):
    """Delete an experiment and verify it's gone."""
    resp = await test_app.post(
        "/api/v1/experiments",
        json={
            "name": "待删除实验",
            "prompt_strategies": ["basic"],
            "test_cases": [sample_test_cases[0]],
        },
    )
    exp_id = resp.json()["experiment_id"]

    resp = await test_app.delete(f"/api/v1/experiments/{exp_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    resp = await test_app.get(f"/api/v1/experiments/{exp_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_nonexistent_experiment_404(test_app):
    """Accessing non-existent experiment returns 404."""
    resp = await test_app.get("/api/v1/experiments/nonexistent123")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_invalid_strategy_in_experiment(test_app, sample_test_cases):
    """Creating experiment with invalid strategy should be rejected."""
    resp = await test_app.post(
        "/api/v1/experiments",
        json={
            "name": "无效策略实验",
            "prompt_strategies": ["invalid_strategy"],
            "test_cases": sample_test_cases[:1],
        },
    )
    assert resp.status_code == 422
