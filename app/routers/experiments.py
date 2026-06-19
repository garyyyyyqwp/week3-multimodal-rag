"""Experiments Router — CRUD + Run + Report for prompt strategy evaluation."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.schemas.experiment import (
    ExperimentCreateRequest,
    ExperimentInfo,
    ExperimentListResponse,
    ExperimentDetailResponse,
    ExperimentRunResponse,
    ExperimentDeleteResponse,
    ReportResponse,
    RunResultItem,
    DimensionScores,
    StrategyCostItem,
    TokenBreakdownSchema,
)
from app.services.experiment_runner import (
    Experiment,
    TestCase,
    _save_experiment,
    _load_experiment,
    _list_experiments,
    run_experiment,
)
from app.services.reporting import generate_experiment_report

router = APIRouter(tags=["experiments"])


# ---------------------------------------------------------------------------
# Create Experiment
# ---------------------------------------------------------------------------

@router.post("", response_model=ExperimentInfo, status_code=201)
async def create_experiment(request: ExperimentCreateRequest):
    """Create a new experiment with prompt strategies and test cases."""
    experiment_id = uuid.uuid4().hex[:12]

    test_cases = [
        TestCase(
            case_id=uuid.uuid4().hex[:8],
            question=tc.question,
            reference_answer=tc.reference_answer,
            has_image=tc.has_image,
        )
        for tc in request.test_cases
    ]

    exp = Experiment(
        experiment_id=experiment_id,
        name=request.name,
        prompt_strategies=request.prompt_strategies,
        test_cases=test_cases,
        created_at=datetime.now(timezone.utc).isoformat(),
        status="created",
    )

    _save_experiment(exp)

    return ExperimentInfo(
        experiment_id=exp.experiment_id,
        name=exp.name,
        strategies=exp.prompt_strategies,
        test_case_count=len(exp.test_cases),
        status=exp.status,
        created_at=exp.created_at,
    )


# ---------------------------------------------------------------------------
# List Experiments
# ---------------------------------------------------------------------------

@router.get("", response_model=ExperimentListResponse)
async def list_experiments():
    """List all experiments with summary info."""
    experiments = _list_experiments()
    return ExperimentListResponse(
        experiments=[
            ExperimentInfo(**exp) for exp in experiments
        ],
        total=len(experiments),
    )


# ---------------------------------------------------------------------------
# Get Experiment Detail
# ---------------------------------------------------------------------------

@router.get(
    "/{experiment_id}",
    response_model=ExperimentDetailResponse,
    responses={404: {"description": "实验不存在"}},
)
async def get_experiment(experiment_id: str):
    """Get an experiment's full details, including results if completed."""
    exp = _load_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

    # Build response
    results = []
    for r in exp.results:
        judge = None
        if r.judge_result:
            judge = DimensionScores(
                factual_accuracy=r.judge_result.factual_accuracy,
                image_relevance=r.judge_result.image_relevance,
                completeness=r.judge_result.completeness,
                conciseness=r.judge_result.conciseness,
                reasoning=r.judge_result.reasoning,
                total_score=r.judge_result.total_score,
            )

        results.append(RunResultItem(
            strategy_name=r.strategy_name,
            case_id=r.case_id,
            question=r.question,
            answer=r.answer,
            reference_answer=r.reference_answer,
            latency_seconds=r.latency_seconds,
            token_breakdown=TokenBreakdownSchema(
                total_tokens=r.token_breakdown.total_tokens,
                total_input_tokens=r.token_breakdown.total_input_tokens,
                completion_tokens=r.token_breakdown.completion_tokens,
                image_count=r.token_breakdown.image_count,
                input_cost=r.token_breakdown.input_cost,
                output_cost=r.token_breakdown.output_cost,
                image_cost=r.token_breakdown.image_cost,
                total_cost=r.token_breakdown.total_cost,
            ),
            judge_result=judge,
            use_vision=r.use_vision,
            error=r.error,
        ))

    cost_reports = {}
    for strategy, report in exp.strategy_cost_reports.items():
        cost_reports[strategy] = StrategyCostItem(
            strategy_name=report.strategy_name,
            total_calls=report.total_calls,
            total_tokens=report.total_tokens,
            total_input_tokens=report.total_input_tokens,
            total_completion_tokens=report.total_completion_tokens,
            total_cost=report.total_cost,
            avg_tokens_per_call=report.avg_tokens_per_call,
            avg_cost_per_call=report.avg_cost_per_call,
        )

    return ExperimentDetailResponse(
        experiment_id=exp.experiment_id,
        name=exp.name,
        prompt_strategies=exp.prompt_strategies,
        test_case_count=len(exp.test_cases),
        created_at=exp.created_at,
        status=exp.status,
        total_duration_seconds=exp.total_duration_seconds,
        strategy_cost_reports=cost_reports,
        results=results,
    )


# ---------------------------------------------------------------------------
# Run Experiment
# ---------------------------------------------------------------------------

@router.post(
    "/{experiment_id}/run",
    response_model=ExperimentRunResponse,
    responses={
        404: {"description": "实验不存在"},
        409: {"description": "实验已完成，不能重复执行"},
    },
)
async def execute_experiment(experiment_id: str):
    """Execute an experiment — runs all prompt strategies against all test cases.

    This endpoint returns immediately after triggering execution.
    Results are persisted to disk and can be retrieved via GET /{id} or GET /{id}/report.
    """
    exp = _load_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

    if exp.status == "completed":
        raise HTTPException(status_code=409, detail="实验已完成，不能重复执行。请创建新实验。")

    if exp.status == "running":
        raise HTTPException(status_code=409, detail="实验正在运行中")

    # Reset status for re-run (failed or created)
    exp.status = "created"
    exp.results = []
    _save_experiment(exp)

    # Run experiment (this may take a while for large test sets)
    try:
        result = await run_experiment(experiment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"实验执行失败: {str(e)}")

    return ExperimentRunResponse(
        experiment_id=result.experiment_id,
        status=result.status,
        total_results=len(result.results),
        duration_seconds=result.total_duration_seconds,
    )


# ---------------------------------------------------------------------------
# Get Experiment Report
# ---------------------------------------------------------------------------

@router.get(
    "/{experiment_id}/report",
    response_model=ReportResponse,
    responses={404: {"description": "实验不存在"}},
)
async def get_experiment_report(experiment_id: str):
    """Get the generated Markdown comparison report for a completed experiment."""
    exp = _load_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

    report = generate_experiment_report(exp)

    return ReportResponse(
        experiment_id=exp.experiment_id,
        report=report,
        format="markdown",
    )


# ---------------------------------------------------------------------------
# Delete Experiment
# ---------------------------------------------------------------------------

@router.delete(
    "/{experiment_id}",
    response_model=ExperimentDeleteResponse,
    responses={404: {"description": "实验不存在"}},
)
async def delete_experiment(experiment_id: str):
    """Delete an experiment and its saved data."""
    exp = _load_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"实验不存在: {experiment_id}")

    # Remove from memory and disk
    from app.services.experiment_runner import _experiments, _SAVE_DIR

    _experiments.pop(experiment_id, None)
    path = _SAVE_DIR / f"{experiment_id}.json"
    if path.exists():
        path.unlink()

    return ExperimentDeleteResponse(
        experiment_id=experiment_id,
        deleted=True,
    )
