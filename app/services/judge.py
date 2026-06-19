"""
LLM-as-Judge — structured 4-dimension evaluator using Pydantic + tool_choice="required".

Evaluates RAG answers across:
  1. factual_accuracy  (1-5): 与参考资料的事实是否一致
  2. image_relevance   (1-5): 是否正确利用了图片中的视觉信息
  3. completeness      (1-5): 是否覆盖了参考答案中的关键信息点
  4. conciseness       (1-5): 是否包含无关或冗余内容

Each dimension uses a strict Pydantic Schema enforced via function calling.
"""

import json
import asyncio
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured Judge Schema (tool_choice="required")
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    """Structured 4-dimension evaluation output. Enforced via tool_choice='required'."""
    factual_accuracy: int = Field(
        ge=1, le=5,
        description="事实准确性：回答与参考资料的事实是否一致。1=大量事实错误，5=完全准确无错误",
    )
    image_relevance: int = Field(
        ge=1, le=5,
        description="图文关联度：是否正确利用了图片中的视觉信息辅助回答。1=完全忽略图片信息，5=充分结合图片信息",
    )
    completeness: int = Field(
        ge=1, le=5,
        description="完整性：是否覆盖了参考答案中的关键信息点。1=遗漏大量关键信息，5=完全覆盖",
    )
    conciseness: int = Field(
        ge=1, le=5,
        description="简洁性：回答是否精炼无冗余。1=大量无关内容，5=极度精炼无冗余",
    )
    reasoning: str = Field(
        default="",
        description="评分依据：简要说明各维度评分的理由（100-200字）",
    )


@dataclass
class JudgeResult:
    """Fully resolved judge evaluation for one answer."""
    factual_accuracy: int
    image_relevance: int
    completeness: int
    conciseness: int
    reasoning: str
    total_score: float = 0.0  # weighted average

    def __post_init__(self):
        # Weighted: factual 0.35, image 0.25, completeness 0.25, conciseness 0.15
        self.total_score = round(
            self.factual_accuracy * 0.35 +
            self.image_relevance * 0.25 +
            self.completeness * 0.25 +
            self.conciseness * 0.15,
            2,
        )


# ---------------------------------------------------------------------------
# Judge System Prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """你是一个专业的问答质量评估专家。你需要根据给定的参考资料、参考答案和用户问题，对 AI 生成的回答进行 4 维度评分。

评分维度说明：
1. **事实准确性 (1-5)**：回答中的事实是否与参考资料一致。如果有任何事实错误扣分。
2. **图文关联度 (1-5)**：回答是否正确地利用了图片中的视觉信息。如果参考资料包含图片但回答未利用，评1分。如果只有文字资料（无图片），评3分（不适用）。
3. **完整性 (1-5)**：回答是否覆盖了参考答案中的关键信息点。
4. **简洁性 (1-5)**：回答是否精炼，无无关或冗余内容。

评分标准：
- 5分：优秀，无任何问题
- 4分：良好，有极少量不足
- 3分：合格，有明显不足但不影响核心信息
- 2分：较差，存在较多问题
- 1分：很差，基本不合格

请通过函数调用返回评分结果。"""


# ---------------------------------------------------------------------------
# Judge Function
# ---------------------------------------------------------------------------

async def judge_answer(
    question: str,
    answer: str,
    reference_answer: str,
    context: str,
    has_images: bool = False,
    model: str | None = None,
) -> JudgeResult:
    """Evaluate a single answer using LLM-as-Judge with structured output.

    Args:
        question: The user's question
        answer: The AI-generated answer to evaluate
        reference_answer: The expected/golden answer
        context: The retrieved context (text + image descriptions) used
        has_images: Whether the context included images
        model: LLM model to use for judging (defaults to text model)

    Returns:
        JudgeResult with 4 dimension scores and reasoning.
    """
    from app.services.llm import get_client, get_model

    user_prompt = f"""请对以下 AI 回答进行评分。

【用户问题】
{question}

【参考资料（提供给AI的上下文）】
{context}

【参考答案】
{reference_answer}

【AI 生成的回答】
{answer}

【图片信息】{'有图片资料' if has_images else '无图片资料（纯文本）'}

请返回你的评分结果。"""

    client = get_client()
    llm_model = model or get_model()

    # Use tool_choice="required" with the JudgeVerdict schema
    tools = [{
        "type": "function",
        "function": {
            "name": "submit_evaluation",
            "description": "提交对 AI 回答的 4 维度评分结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "factual_accuracy": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "事实准确性评分 (1-5)",
                    },
                    "image_relevance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "图文关联度评分 (1-5)",
                    },
                    "completeness": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "完整性评分 (1-5)",
                    },
                    "conciseness": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "简洁性评分 (1-5)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "各维度评分的简要依据",
                    },
                },
                "required": [
                    "factual_accuracy",
                    "image_relevance",
                    "completeness",
                    "conciseness",
                    "reasoning",
                ],
            },
        },
    }]

    try:
        response = await client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "submit_evaluation"}},
            temperature=0.1,  # low temp for consistent judging
        )

        # Extract tool call arguments
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        return JudgeResult(
            factual_accuracy=args["factual_accuracy"],
            image_relevance=args["image_relevance"],
            completeness=args["completeness"],
            conciseness=args["conciseness"],
            reasoning=args.get("reasoning", ""),
        )

    except (json.JSONDecodeError, AttributeError, KeyError, IndexError) as e:
        # Fallback: return minimum scores on parse failure
        return JudgeResult(
            factual_accuracy=1,
            image_relevance=1,
            completeness=1,
            conciseness=1,
            reasoning=f"评分解析失败: {str(e)[:100]}",
        )


async def judge_batch(
    items: list[dict],  # [{question, answer, reference_answer, context, has_images}, ...]
    concurrency: int = 5,
    model: str | None = None,
) -> list[JudgeResult]:
    """Evaluate multiple answers concurrently with a semaphore.

    Args:
        items: List of evaluation items
        concurrency: Max concurrent judge calls
        model: LLM model override

    Returns:
        List of JudgeResult in the same order as items.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _judge_one(item: dict) -> JudgeResult:
        async with semaphore:
            return await judge_answer(
                question=item["question"],
                answer=item["answer"],
                reference_answer=item.get("reference_answer", ""),
                context=item.get("context", ""),
                has_images=item.get("has_images", False),
                model=model,
            )

    tasks = [_judge_one(item) for item in items]
    return await asyncio.gather(*tasks)
