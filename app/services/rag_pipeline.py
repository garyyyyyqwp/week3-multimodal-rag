"""
Multimodal RAG Pipeline — upgraded from Week 2 text-only pipeline.

Key additions:
  - Multimodal context building (text + image for Vision LLM)
  - Automatic model routing: Vision LLM when images are relevant, Text LLM otherwise
  - Reference labels: [T1] for text, [F1] for images
  - 4 prompt strategies supported via template injection
"""

from dataclasses import dataclass, field
import logging
from typing import AsyncIterator, Optional

from app.services.llm import get_client, get_model, get_vision_client, get_vision_model
from app.services.vector_store import get_vector_store
from app.services.retriever import (
    retrieve_multimodal,
    TextChunk,
    ImageChunk,
    MultimodalRetrievalResult,
)
from app.utils.config import RAG_TOP_K, RAG_TOP_M

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------

def build_text_context(texts: list[TextChunk]) -> str:
    """Build context string from text chunks with reference labels."""
    parts = []
    for t in texts:
        parts.append(f"{t.ref_label} (来源: {t.filename})\n{t.content}")
    return "\n\n".join(parts)


def build_image_context(images: list[ImageChunk]) -> str:
    """Build image description context from image captions with reference labels."""
    parts = []
    for img in images:
        parts.append(f"{img.ref_label} (图片摘要, 来源: {img.filename})\n{img.caption}")
    return "\n\n".join(parts)


def build_full_context(retrieval: MultimodalRetrievalResult) -> str:
    """Build combined text + image context."""
    parts = []
    if retrieval.texts:
        parts.append("【文字资料】\n" + build_text_context(retrieval.texts))
    if retrieval.images:
        parts.append("【图片资料】\n" + build_image_context(retrieval.images))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt Templates (4 Strategies)
# ---------------------------------------------------------------------------

PROMPT_STRATEGIES = {
    "basic": (
        "你是一个知识库问答助手。基于以下参考资料回答用户问题。\n"
        "如果资料不足以回答，请如实说明。\n\n"
        "参考资料：\n{context}\n\n"
        "用户问题：{question}\n\n"
        "要求：\n"
        "1. 回答准确、简洁\n"
        "2. 引用资料时标注来源编号，如[T1]、[F1]\n"
        "3. 如果资料不足以回答，明确说明"
    ),

    "fewshot": (
        "你是一个知识库问答助手。下面是两个问答示例，然后请你回答实际的问题。\n\n"
        "--- 示例 1 ---\n"
        "资料：[T1] Transformer 架构由 Vaswani 等人在 2017 年提出，核心创新是自注意力机制。\n"
        "问题：Transformer 架构的主要创新是什么？\n"
        "回答：根据资料[T1]，Transformer 架构的主要创新是自注意力机制（Self-Attention）。\n\n"
        "--- 示例 2 ---\n"
        "资料：[T1] 反向传播算法利用链式法则计算梯度。[F1] 图片展示了前向传播和反向传播的数据流。\n"
        "问题：反向传播算法的计算过程是怎样的？\n"
        "回答：根据文字资料[T1]，反向传播算法利用链式法则计算损失函数相对于每个权重的梯度。"
        "结合图片[F1]展示的数据流，该过程包括前向传播计算输出、计算损失、反向传播误差、以及使用梯度下降更新权重四个步骤。\n\n"
        "--- 现在回答以下问题 ---\n\n"
        "参考资料：\n{context}\n\n"
        "用户问题：{question}\n\n"
        "要求：\n"
        "1. 回答准确、简洁\n"
        "2. 引用资料时标注来源编号\n"
        "3. 如果包含了图片资料[F*]，请结合图片信息回答"
    ),

    "cot": (
        "你是一个知识库问答助手。请按照以下步骤逐步推理，然后给出最终回答。\n\n"
        "步骤 1：仔细阅读以下参考资料，列出与问题相关的所有关键信息点。\n"
        "步骤 2：分析这些信息点之间的逻辑关系。\n"
        "步骤 3：基于分析，给出完整的回答。\n\n"
        "参考资料：\n{context}\n\n"
        "用户问题：{question}\n\n"
        "请在回答中明确标注你的推理过程：\n"
        "【关键信息】列出相关的事实和数据\n"
        "【分析推理】说明你的推理逻辑\n"
        "【回答】给出最终答案（需引用来源编号）"
    ),

    "multimodal_step": (
        "你是一个多模态知识库问答助手。请按照以下步骤处理图文混合资料。\n\n"
        "步骤 1：先阅读所有【文字资料】，提取与问题相关的关键文本信息。\n"
        "步骤 2：再查看所有【图片资料】的摘要描述，理解图片中的视觉信息。\n"
        "步骤 3：将文字资料与图片资料进行关联分析——图片中的内容如何补充或佐证文字信息？\n"
        "步骤 4：综合文字和图片信息，给出最终回答。\n\n"
        "参考资料：\n{context}\n\n"
        "用户问题：{question}\n\n"
        "请在回答中明确标注：\n"
        "【文字信息】从文本中提取的关键点\n"
        "【图片信息】从图片中获取的关键视觉信息\n"
        "【综合分析】将文字与图片信息关联后的完整回答（需引用来源编号）"
    ),
}


def build_prompt(
    strategy: str,
    context: str,
    question: str,
) -> str:
    """Build a prompt using the specified strategy template.

    Args:
        strategy: One of "basic", "fewshot", "cot", "multimodal_step"
        context: Formatted context string
        question: User's question

    Returns:
        Formatted prompt string.
    """
    template = PROMPT_STRATEGIES.get(strategy, PROMPT_STRATEGIES["basic"])
    return template.format(context=context, question=question)


# ---------------------------------------------------------------------------
# Model Routing
# ---------------------------------------------------------------------------

def should_use_vision(retrieval: MultimodalRetrievalResult, threshold: float = 0.3) -> bool:
    """Decide whether to use Vision LLM based on image retrieval results.

    Uses Vision LLM if:
      - At least one image was retrieved with a reasonable score
      - AND the question might benefit from visual context
    """
    if not retrieval.images:
        return False
    # At least one image with score above threshold
    return any(img.score > threshold for img in retrieval.images)


# ---------------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------------

async def generate_answer_stream(
    question: str,
    retrieval: MultimodalRetrievalResult,
    strategy: str = "basic",
    use_vision: bool | None = None,
) -> AsyncIterator[str]:
    """Generate LLM answer with context, streaming tokens.

    Automatically routes to Vision LLM when images are present and relevant.

    Args:
        question: User's question
        retrieval: Multimodal retrieval results
        strategy: Prompt strategy to use
        use_vision: Force vision mode (None = auto-detect)

    Yields:
        Token strings from the LLM response.
    """
    context = build_full_context(retrieval)
    prompt = build_prompt(strategy, context, question)

    # Auto-detect vision mode
    if use_vision is None:
        use_vision = should_use_vision(retrieval)

    if use_vision and retrieval.images:
        # Vision LLM mode: send images + text
        client = get_vision_client()
        model = get_vision_model()

        # Build multimodal message content
        content_parts: list[dict] = [
            {"type": "text", "text": prompt},
        ]
        # Add retrieved images
        for img in retrieval.images:
            if img.image_base64:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img.image_base64}"},
                })

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content_parts}],
                stream=True,
            )
        except Exception as e:
            logger.error("Vision LLM API call failed: %s", e)
            raise
    else:
        # Text LLM mode
        client = get_client()
        model = get_model()

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
        except Exception as e:
            logger.error("Text LLM API call failed: %s", e)
            raise

    try:
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error("Error while streaming LLM response: %s", e)
        raise


async def generate_bare_answer_stream(question: str) -> AsyncIterator[str]:
    """Generate pure LLM answer without any context (for comparison)."""
    client = get_client()
    model = get_model()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            stream=True,
        )
    except Exception as e:
        logger.error("Bare answer LLM API call failed: %s", e)
        raise

    try:
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error("Error while streaming bare answer response: %s", e)
        raise


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    retrieval: MultimodalRetrievalResult
    answer: str
    strategy: str
    use_vision: bool


async def rag_pipeline(
    question: str,
    top_k: int = RAG_TOP_K,
    top_m: int = RAG_TOP_M,
    strategy: str = "basic",
    force_vision: bool | None = None,
) -> PipelineResult:
    """Full multimodal RAG pipeline: retrieve → generate.

    Args:
        question: User's question
        top_k: Number of text chunks to retrieve
        top_m: Number of images to retrieve
        strategy: Prompt strategy to use
        force_vision: Force/disable Vision LLM (None = auto)

    Returns:
        PipelineResult with retrieval results and full answer text.
    """
    # Retrieve with image loading when vision might be needed
    retrieval = await retrieve_multimodal(
        query=question,
        top_k=top_k,
        top_m=top_m,
        load_images=True,  # always load for potential vision use
    )

    use_vision = force_vision if force_vision is not None else should_use_vision(retrieval)

    # Collect answer
    answer_parts = []
    try:
        async for token in generate_answer_stream(
            question, retrieval, strategy=strategy, use_vision=use_vision,
        ):
            answer_parts.append(token)
    except Exception as e:
        logger.error("RAG pipeline answer generation failed: %s", e)
        raise
    answer = "".join(answer_parts)

    return PipelineResult(
        retrieval=retrieval,
        answer=answer,
        strategy=strategy,
        use_vision=use_vision,
    )
