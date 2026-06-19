from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# Experiment Schemas
# ---------------------------------------------------------------------------

class TestCaseItem(BaseModel):
    """A single test case for experiment evaluation."""
    question: str = Field(..., min_length=1, max_length=2000, description="测试问题")
    reference_answer: str = Field(..., min_length=1, max_length=5000, description="参考答案")
    has_image: bool = Field(default=False, description="是否需要图片上下文")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("问题不能为空")
        return v


class ExperimentCreateRequest(BaseModel):
    """Request to create a new experiment."""
    name: str = Field(..., min_length=1, max_length=200, description="实验名称")
    prompt_strategies: list[str] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="Prompt 策略列表，支持: basic, fewshot, cot, multimodal_step",
    )
    test_cases: list[TestCaseItem] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="测试用例列表（1-100条）",
    )
    concurrency_limit: int = Field(default=5, ge=1, le=20, description="并发限制")

    @field_validator("prompt_strategies")
    @classmethod
    def validate_strategies(cls, v: list[str]) -> list[str]:
        valid = {"basic", "fewshot", "cot", "multimodal_step"}
        for strategy in v:
            if strategy not in valid:
                raise ValueError(
                    f"无效的 Prompt 策略: '{strategy}'。可选: {', '.join(sorted(valid))}"
                )
        return v

    @field_validator("test_cases")
    @classmethod
    def validate_test_cases(cls, v: list[TestCaseItem]) -> list[TestCaseItem]:
        if not v:
            raise ValueError("至少需要 1 条测试用例")
        # Deduplicate by question
        seen = set()
        unique = []
        for tc in v:
            q = tc.question.strip()
            if q not in seen:
                seen.add(q)
                unique.append(tc)
        return unique


class ExperimentInfo(BaseModel):
    experiment_id: str
    name: str
    strategies: list[str]
    test_case_count: int
    status: str
    created_at: str


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentInfo]
    total: int


class DimensionScores(BaseModel):
    factual_accuracy: int = Field(ge=1, le=5)
    image_relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    conciseness: int = Field(ge=1, le=5)
    reasoning: str = ""
    total_score: float = 0.0


class TokenBreakdownSchema(BaseModel):
    total_tokens: int = 0
    total_input_tokens: int = 0
    completion_tokens: int = 0
    image_count: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    image_cost: float = 0.0
    total_cost: float = 0.0


class RunResultItem(BaseModel):
    strategy_name: str
    case_id: str
    question: str
    answer: str
    reference_answer: str
    latency_seconds: float
    token_breakdown: TokenBreakdownSchema
    judge_result: DimensionScores | None = None
    use_vision: bool = False
    error: str = ""


class StrategyCostItem(BaseModel):
    strategy_name: str
    total_calls: int
    total_tokens: int
    total_input_tokens: int
    total_completion_tokens: int
    total_cost: float
    avg_tokens_per_call: float
    avg_cost_per_call: float


class ExperimentDetailResponse(BaseModel):
    experiment_id: str
    name: str
    prompt_strategies: list[str]
    test_case_count: int
    created_at: str
    status: str
    total_duration_seconds: float = 0.0
    strategy_cost_reports: dict[str, StrategyCostItem] = {}
    results: list[RunResultItem] = []


class ExperimentRunResponse(BaseModel):
    experiment_id: str
    status: str
    total_results: int
    duration_seconds: float


class ExperimentDeleteResponse(BaseModel):
    experiment_id: str
    deleted: bool


class ReportResponse(BaseModel):
    experiment_id: str
    report: str
    format: str = "markdown"
