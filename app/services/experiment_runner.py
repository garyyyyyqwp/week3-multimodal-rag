"""
Experiment Runner — concurrent execution engine for prompt strategy evaluation.

Runs N prompt strategies × M test cases with:
  - asyncio.Semaphore for concurrency control
  - Result persistence to disk (JSON)
  - Token counting per call
  - Automatic scoring via LLM-as-Judge after all answers collected
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.services.rag_pipeline import (
    rag_pipeline,
    build_full_context,
    PROMPT_STRATEGIES,
)
from app.services.retriever import MultimodalRetrievalResult
from app.services.judge import JudgeResult, judge_batch
from app.services.token_economics import (
    count_tokens,
    estimate_text_cost,
    estimate_vision_cost,
    TokenBreakdown,
    StrategyCostReport,
    aggregate_costs,
)
from app.utils.config import (
    EXPERIMENT_MAX_CONCURRENCY,
    EXPERIMENT_DATA_DIR,
)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single test case within an experiment."""
    case_id: str
    question: str
    reference_answer: str
    has_image: bool = False


@dataclass
class RunResult:
    """Result of running one prompt strategy on one test case."""
    strategy_name: str
    case_id: str
    question: str
    answer: str
    reference_answer: str
    latency_seconds: float
    token_breakdown: TokenBreakdown
    judge_result: JudgeResult | None = None
    use_vision: bool = False
    context: str = ""
    error: str = ""


@dataclass
class Experiment:
    """Full experiment configuration and results."""
    experiment_id: str
    name: str
    prompt_strategies: list[str]   # e.g., ["basic", "fewshot", "cot", "multimodal_step"]
    test_cases: list[TestCase]
    created_at: str
    status: str = "created"         # created | running | completed | failed
    results: list[RunResult] = field(default_factory=list)
    strategy_cost_reports: dict[str, StrategyCostReport] = field(default_factory=dict)
    total_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "prompt_strategies": self.prompt_strategies,
            "test_cases": [
                {"case_id": tc.case_id, "question": tc.question,
                 "reference_answer": tc.reference_answer, "has_image": tc.has_image}
                for tc in self.test_cases
            ],
            "created_at": self.created_at,
            "status": self.status,
            "results": [
                {
                    "strategy_name": r.strategy_name,
                    "case_id": r.case_id,
                    "question": r.question,
                    "answer": r.answer,
                    "reference_answer": r.reference_answer,
                    "latency_seconds": r.latency_seconds,
                    "token_breakdown": {
                        "total_tokens": r.token_breakdown.total_tokens,
                        "total_input_tokens": r.token_breakdown.total_input_tokens,
                        "completion_tokens": r.token_breakdown.completion_tokens,
                        "image_count": r.token_breakdown.image_count,
                        "input_cost": r.token_breakdown.input_cost,
                        "output_cost": r.token_breakdown.output_cost,
                        "image_cost": r.token_breakdown.image_cost,
                        "total_cost": r.token_breakdown.total_cost,
                    },
                    "judge_result": {
                        "factual_accuracy": r.judge_result.factual_accuracy,
                        "image_relevance": r.judge_result.image_relevance,
                        "completeness": r.judge_result.completeness,
                        "conciseness": r.judge_result.conciseness,
                        "reasoning": r.judge_result.reasoning,
                        "total_score": r.judge_result.total_score,
                    } if r.judge_result else None,
                    "use_vision": r.use_vision,
                    "error": r.error,
                }
                for r in self.results
            ],
            "strategy_cost_reports": {
                k: {
                    "strategy_name": v.strategy_name,
                    "total_calls": v.total_calls,
                    "total_tokens": v.total_tokens,
                    "total_input_tokens": v.total_input_tokens,
                    "total_completion_tokens": v.total_completion_tokens,
                    "total_cost": v.total_cost,
                    "avg_tokens_per_call": v.avg_tokens_per_call,
                    "avg_cost_per_call": v.avg_cost_per_call,
                }
                for k, v in self.strategy_cost_reports.items()
            },
            "total_duration_seconds": self.total_duration_seconds,
        }


# ---------------------------------------------------------------------------
# In-memory experiment store (can be swapped for DB later)
# ---------------------------------------------------------------------------

_experiments: dict[str, Experiment] = {}

_SAVE_DIR = Path(EXPERIMENT_DATA_DIR)
_SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _save_experiment(exp: Experiment):
    """Persist experiment to disk."""
    _experiments[exp.experiment_id] = exp
    path = _SAVE_DIR / f"{exp.experiment_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(exp.to_dict(), f, ensure_ascii=False, indent=2)


def _load_experiment(experiment_id: str) -> Experiment | None:
    """Load experiment from disk."""
    if experiment_id in _experiments:
        return _experiments[experiment_id]
    path = _SAVE_DIR / f"{experiment_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Reconstruct Experiment from dict (simplified)
        exp = Experiment(
            experiment_id=data["experiment_id"],
            name=data["name"],
            prompt_strategies=data["prompt_strategies"],
            test_cases=[
                TestCase(
                    case_id=tc["case_id"],
                    question=tc["question"],
                    reference_answer=tc["reference_answer"],
                    has_image=tc.get("has_image", False),
                )
                for tc in data["test_cases"]
            ],
            created_at=data["created_at"],
            status=data["status"],
            total_duration_seconds=data.get("total_duration_seconds", 0.0),
        )
        _experiments[experiment_id] = exp
        return exp
    return None


def _list_experiments() -> list[dict]:
    """List all saved experiments with summary info."""
    results = []
    for path in sorted(_SAVE_DIR.glob("*.json"), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "experiment_id": data["experiment_id"],
                "name": data["name"],
                "strategies": data["prompt_strategies"],
                "test_case_count": len(data["test_cases"]),
                "status": data["status"],
                "created_at": data["created_at"],
            })
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_experiment(experiment_id: str) -> Experiment:
    """Execute an experiment: run all strategies × test cases concurrently.

    Algorithm:
      1. For each strategy, for each test case → rag_pipeline (answer generation)
      2. Collect all answers → judge_batch (LLM scoring)
      3. Aggregate per-strategy cost reports
      4. Persist results to disk

    Concurrency: asyncio.Semaphore(EXPERIMENT_MAX_CONCURRENCY) limits simultaneous
    LLM calls to avoid rate limiting.
    """
    exp = _load_experiment(experiment_id)
    if exp is None:
        raise ValueError(f"实验不存在: {experiment_id}")

    exp.status = "running"
    _save_experiment(exp)

    semaphore = asyncio.Semaphore(EXPERIMENT_MAX_CONCURRENCY)
    start_time = time.monotonic()

    # Phase 1: Run all (strategy, test_case) pairs
    all_results: list[RunResult] = []
    total_pairs = len(exp.prompt_strategies) * len(exp.test_cases)

    async def _run_one(strategy_name: str, case: TestCase) -> RunResult:
        async with semaphore:
            t0 = time.monotonic()
            try:
                pipeline_result = await rag_pipeline(
                    question=case.question,
                    strategy=strategy_name,
                )

                latency = round(time.monotonic() - t0, 3)

                # Build context for judge
                context = build_full_context(pipeline_result.retrieval)

                # Token breakdown
                if pipeline_result.use_vision and pipeline_result.retrieval.images:
                    breakdown = estimate_vision_cost(
                        prompt=build_full_context(pipeline_result.retrieval),
                        image_count=len(pipeline_result.retrieval.images),
                        completion_tokens=count_tokens(pipeline_result.answer),
                    )
                else:
                    breakdown = estimate_text_cost(
                        prompt=build_full_context(pipeline_result.retrieval),
                        completion_tokens=count_tokens(pipeline_result.answer),
                    )

                return RunResult(
                    strategy_name=strategy_name,
                    case_id=case.case_id,
                    question=case.question,
                    answer=pipeline_result.answer,
                    reference_answer=case.reference_answer,
                    latency_seconds=latency,
                    token_breakdown=breakdown,
                    use_vision=pipeline_result.use_vision,
                    context=context,
                    error="",
                )
            except Exception as e:
                latency = round(time.monotonic() - t0, 3)
                return RunResult(
                    strategy_name=strategy_name,
                    case_id=case.case_id,
                    question=case.question,
                    answer="",
                    reference_answer=case.reference_answer,
                    latency_seconds=latency,
                    token_breakdown=TokenBreakdown(),
                    error=str(e),
                )

    try:
        # Launch all tasks
        tasks = [
            _run_one(strategy, case)
            for strategy in exp.prompt_strategies
            for case in exp.test_cases
        ]
        all_results = await asyncio.gather(*tasks)

        # Phase 2: Judge all successful results
        judge_items = []
        for r in all_results:
            if not r.error and r.answer:
                judge_items.append({
                    "question": r.question,
                    "answer": r.answer,
                    "reference_answer": r.reference_answer,
                    "context": r.context,
                    "has_images": r.use_vision,
                })

        if judge_items:
            judge_results: list[JudgeResult] = await judge_batch(
                items=judge_items,
                concurrency=EXPERIMENT_MAX_CONCURRENCY,
            )
            # Assign judge results back
            judge_idx = 0
            for r in all_results:
                if not r.error and r.answer:
                    r.judge_result = judge_results[judge_idx]
                    judge_idx += 1

        exp.results = all_results

        # Phase 3: Aggregate per-strategy cost reports
        cost_reports: dict[str, StrategyCostReport] = {}
        for strategy in exp.prompt_strategies:
            strategy_breakdowns = [
                r.token_breakdown
                for r in all_results
                if r.strategy_name == strategy and not r.error
            ]
            report = aggregate_costs(strategy_breakdowns)
            report.strategy_name = strategy
            cost_reports[strategy] = report
        exp.strategy_cost_reports = cost_reports

        exp.total_duration_seconds = round(time.monotonic() - start_time, 2)
        exp.status = "completed"

    except Exception as e:
        exp.total_duration_seconds = round(time.monotonic() - start_time, 2)
        exp.status = "failed"
        exp.results = all_results
        _save_experiment(exp)
        raise

    _save_experiment(exp)
    return exp
