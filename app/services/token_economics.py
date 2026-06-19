"""
Token economics — cost analysis for different prompt strategies.

Handles:
  - Token counting via tiktoken
  - Cost estimation for Text LLM vs Vision LLM
  - Per-strategy token breakdown (system/user/completion)
  - Image cost calculation (Vision LLM charges by resolution, not tokens)
"""

from dataclasses import dataclass, field

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Pricing (CNY per 1K tokens / per image)
# ---------------------------------------------------------------------------

# Zhipu GLM-4-Flash pricing (source: bigmodel.cn)
PRICING = {
    "glm-4-flash": {
        "input_per_1k": 0.0001,     # 0.1 元/百万 tokens → 0.0001 元/千 tokens
        "output_per_1k": 0.0001,
    },
    "glm-4v": {
        "input_per_1k": 0.001,      # approximate
        "output_per_1k": 0.001,
        "image_low_res": 0.002,     # per image
        "image_high_res": 0.007,    # per image
    },
    "glm-4.6v-flash": {
        "input_per_1k": 0,          # 免费
        "output_per_1k": 0,         # 免费
        "image_low_res": 0,         # 免费
        "image_high_res": 0,        # 免费
    },
}


@dataclass
class TokenBreakdown:
    """Token counts for a single LLM call."""
    system_tokens: int = 0
    user_tokens: int = 0
    image_count: int = 0
    context_tokens: int = 0
    completion_tokens: int = 0
    total_input_tokens: int = 0
    total_tokens: int = 0

    # Costs (CNY)
    input_cost: float = 0.0
    output_cost: float = 0.0
    image_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class StrategyCostReport:
    strategy_name: str
    total_calls: int
    total_tokens: int
    total_input_tokens: int
    total_completion_tokens: int
    total_cost: float
    avg_tokens_per_call: float
    avg_cost_per_call: float
    breakdown: TokenBreakdown = field(default_factory=TokenBreakdown)


def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    if not text:
        return 0
    return len(ENC.encode(text))


def count_message_tokens(messages: list[dict]) -> int:
    """Estimate token count for chat messages.

    This is an approximation — the exact count depends on the model's tokenizer.
    Uses tiktoken cl100k_base as a close approximation for GLM models.
    """
    total = 0
    for msg in messages:
        total += 4  # message framing overhead
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            # Vision content with mixed text/image parts
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total += count_tokens(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        # Vision images: rough estimate
                        total += 85  # base tokens for low-res image
        total += 2  # end-of-message
    total += 2  # priming
    return total


def estimate_vision_cost(
    prompt: str,
    image_count: int,
    completion_tokens: int,
    model: str = "glm-4.6v-flash",
    high_res: bool = False,
) -> TokenBreakdown:
    """Estimate cost for a Vision LLM call.

    Vision LLMs charge by image resolution + token count.
    """
    pricing = PRICING.get(model, PRICING["glm-4v"])
    input_tokens = count_tokens(prompt)
    total_tokens = input_tokens + completion_tokens

    input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
    output_cost = (completion_tokens / 1000) * pricing["output_per_1k"]

    image_rate = pricing["image_high_res"] if high_res else pricing["image_low_res"]
    image_cost = image_count * image_rate

    return TokenBreakdown(
        system_tokens=0,
        user_tokens=input_tokens,
        image_count=image_count,
        context_tokens=0,
        completion_tokens=completion_tokens,
        total_input_tokens=input_tokens,
        total_tokens=total_tokens,
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        image_cost=round(image_cost, 6),
        total_cost=round(input_cost + output_cost + image_cost, 6),
    )


def estimate_text_cost(
    prompt: str,
    completion_tokens: int,
    model: str = "glm-4-flash",
) -> TokenBreakdown:
    """Estimate cost for a text-only LLM call."""
    pricing = PRICING.get(model, PRICING["glm-4-flash"])
    input_tokens = count_tokens(prompt)
    total_tokens = input_tokens + completion_tokens

    input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
    output_cost = (completion_tokens / 1000) * pricing["output_per_1k"]

    return TokenBreakdown(
        system_tokens=0,
        user_tokens=input_tokens,
        image_count=0,
        context_tokens=0,
        completion_tokens=completion_tokens,
        total_input_tokens=input_tokens,
        total_tokens=total_tokens,
        input_cost=round(input_cost, 6),
        output_cost=round(output_cost, 6),
        image_cost=0.0,
        total_cost=round(input_cost + output_cost, 6),
    )


def aggregate_costs(breakdowns: list[TokenBreakdown]) -> StrategyCostReport:
    """Aggregate multiple call breakdowns into a per-strategy report."""
    if not breakdowns:
        return StrategyCostReport(
            strategy_name="",
            total_calls=0,
            total_tokens=0,
            total_input_tokens=0,
            total_completion_tokens=0,
            total_cost=0.0,
            avg_tokens_per_call=0.0,
            avg_cost_per_call=0.0,
        )

    n = len(breakdowns)
    total_tokens = sum(b.total_tokens for b in breakdowns)
    total_input = sum(b.total_input_tokens for b in breakdowns)
    total_completion = sum(b.completion_tokens for b in breakdowns)
    total_cost = sum(b.total_cost for b in breakdowns)
    total_image_cost = sum(b.image_cost for b in breakdowns)

    return StrategyCostReport(
        strategy_name="",
        total_calls=n,
        total_tokens=total_tokens,
        total_input_tokens=total_input,
        total_completion_tokens=total_completion,
        total_cost=round(total_cost, 4),
        avg_tokens_per_call=round(total_tokens / n, 1),
        avg_cost_per_call=round(total_cost / n, 6),
        breakdown=TokenBreakdown(
            total_tokens=total_tokens,
            total_input_tokens=total_input,
            completion_tokens=total_completion,
            image_cost=round(total_image_cost, 6),
            total_cost=round(total_cost, 6),
        ),
    )


def optimize_prompt(
    prompt: str,
    target_reduction_pct: float = 0.3,
) -> tuple[str, int, int]:
    """Optimize a prompt by suggesting cuts to reduce token count.

    Returns (optimized_prompt, original_tokens, optimized_tokens).
    This is a simple heuristic — in production you'd use a smarter approach.
    """
    original_tokens = count_tokens(prompt)
    lines = prompt.split("\n")

    # Strategy: remove consecutive blank lines and trim leading whitespace
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_blank:
                cleaned_lines.append("")
                prev_blank = True
        else:
            cleaned_lines.append(stripped)
            prev_blank = False

    optimized = "\n".join(cleaned_lines)
    optimized_tokens = count_tokens(optimized)

    return optimized, original_tokens, optimized_tokens
