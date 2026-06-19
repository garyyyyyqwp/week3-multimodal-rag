"""
Report Generator — produces Markdown comparison reports for experiments.

Sections:
  1. 实验概览 (Experiment Overview)
  2. 各维度得分排名 (Dimension Score Rankings)
  3. Token 消耗与成本对比 (Token & Cost Comparison)
  4. 图文关联度专项分析 (Image-Text Relevance Analysis)
  5. 典型案例摘录 (Best & Worst Examples)
  6. 最优 Prompt 推荐 (Best Strategy Recommendation)
"""

from datetime import datetime, timezone

from app.services.experiment_runner import Experiment, RunResult


def _dimension_name(key: str) -> str:
    """Map dimension keys to Chinese names."""
    return {
        "factual_accuracy": "事实准确性",
        "image_relevance": "图文关联度",
        "completeness": "完整性",
        "conciseness": "简洁性",
        "total_score": "加权总分",
    }.get(key, key)


def generate_experiment_report(experiment: Experiment) -> str:
    """Generate a comprehensive Markdown report for a completed experiment.

    Args:
        experiment: The completed Experiment with results and scores.

    Returns:
        Complete Markdown report string.
    """
    if experiment.status != "completed":
        return f"# 实验报告 — {experiment.name}\n\n> ⚠️ 实验状态: {experiment.status}，报告尚未生成。"

    lines: list[str] = []

    # ── Header ──
    lines.append(f"# 多模态 RAG Prompt 策略对比实验报告")
    lines.append(f"**实验名称**: {experiment.name}")
    lines.append(f"**实验 ID**: `{experiment.experiment_id}`")
    lines.append(f"**创建时间**: {experiment.created_at}")
    lines.append(f"**总耗时**: {experiment.total_duration_seconds}s")
    lines.append(f"**状态**: ✅ 已完成")
    lines.append("")

    # ── 1. 实验概览 ──
    lines.append("## 1. 实验概览")
    lines.append("")
    lines.append(f"| 项目 | 值 |")
    lines.append(f"|---|---|")
    lines.append(f"| Prompt 策略数 | {len(experiment.prompt_strategies)} |")
    lines.append(f"| 测试用例数 | {len(experiment.test_cases)} |")
    lines.append(f"| 总运行次数 | {len(experiment.results)} |")
    strategies_list = "、".join(experiment.prompt_strategies)
    lines.append(f"| 策略列表 | {strategies_list} |")
    lines.append("")

    # ── 2. 各维度得分排名 ──
    lines.append("## 2. 各维度得分排名")
    lines.append("")

    # Compute per-strategy average scores across 4 dimensions
    strategy_scores: dict[str, dict[str, list[float]]] = {
        s: {"factual_accuracy": [], "image_relevance": [], "completeness": [], "conciseness": [], "total_score": []}
        for s in experiment.prompt_strategies
    }

    for r in experiment.results:
        if r.judge_result and not r.error:
            scores = strategy_scores[r.strategy_name]
            scores["factual_accuracy"].append(r.judge_result.factual_accuracy)
            scores["image_relevance"].append(r.judge_result.image_relevance)
            scores["completeness"].append(r.judge_result.completeness)
            scores["conciseness"].append(r.judge_result.conciseness)
            scores["total_score"].append(r.judge_result.total_score)

    # Build score table
    dimensions = ["factual_accuracy", "image_relevance", "completeness", "conciseness", "total_score"]
    header = "| 策略 | " + " | ".join(_dimension_name(d) for d in dimensions) + " |"
    lines.append(header)
    lines.append("|---" * (len(dimensions) + 1) + "|")

    ranked = sorted(
        strategy_scores.items(),
        key=lambda kv: sum(kv[1]["total_score"]) / len(kv[1]["total_score"]) if kv[1]["total_score"] else 0,
        reverse=True,
    )

    for rank, (strategy, dims) in enumerate(ranked):
        avg_scores = []
        for d in dimensions:
            vals = dims[d]
            avg = sum(vals) / len(vals) if vals else 0
            avg_scores.append(f"{avg:.2f}")
        medal = "🥇" if rank == 0 else ("🥈" if rank == 1 else ("🥉" if rank == 2 else "  "))
        lines.append(f"| {medal} {strategy} | " + " | ".join(avg_scores) + " |")

    lines.append("")
    lines.append(f"> **评分说明**: 每维度 1-5 分，加权总分 = 事实准确性×0.35 + 图文关联度×0.25 + 完整性×0.25 + 简洁性×0.15")
    lines.append("")

    # ── 3. Token 消耗与成本对比 ──
    lines.append("## 3. Token 消耗与成本对比")
    lines.append("")
    lines.append("| 策略 | 总调用 | 总 Token | 输入 Token | 输出 Token | 总成本(¥) | 均次成本(¥) |")
    lines.append("|---|---|---|---|---|---|---|")

    for strategy in experiment.prompt_strategies:
        report = experiment.strategy_cost_reports.get(strategy)
        if report:
            lines.append(
                f"| {strategy} | {report.total_calls} | {report.total_tokens:,} | "
                f"{report.total_input_tokens:,} | {report.total_completion_tokens:,} | "
                f"¥{report.total_cost:.4f} | ¥{report.avg_cost_per_call:.4f} |"
            )

    lines.append("")
    lines.append("> **成本模型说明**: 文本 LLM (glm-4-flash) 按 Token 计费；Vision LLM (glm-4.6v-flash) 免费使用。")
    lines.append("")

    # ── 4. 图文关联度专项分析 ──
    lines.append("## 4. 图文关联度专项分析")
    lines.append("")

    # Find results that involved images
    image_results = [r for r in experiment.results if r.use_vision and r.judge_result and not r.error]
    text_only_results = [r for r in experiment.results if not r.use_vision and r.judge_result and not r.error]

    if image_results:
        lines.append("### 含图片场景下的图文关联度对比")
        lines.append("")
        lines.append("| 策略 | 平均图文关联度 | 最高 | 最低 |")
        lines.append("|---|---|---|---|")

        for strategy in experiment.prompt_strategies:
            strategy_image = [r for r in image_results if r.strategy_name == strategy]
            if strategy_image:
                scores = [r.judge_result.image_relevance for r in strategy_image]
                lines.append(f"| {strategy} | {sum(scores)/len(scores):.2f} | {max(scores)} | {min(scores)} |")

        lines.append("")

        # Compare best multimodal strategy vs best text strategy on image relevance
        multimodal_strategies = ["multimodal_step", "fewshot"]  # ones that should use images
        text_strategies = ["basic", "cot"]

        mm_scores = [
            r.judge_result.image_relevance
            for r in image_results
            if r.strategy_name in multimodal_strategies
        ]
        txt_scores = [
            r.judge_result.image_relevance
            for r in image_results
            if r.strategy_name in text_strategies
        ]

        if mm_scores and txt_scores:
            mm_avg = sum(mm_scores) / len(mm_scores)
            txt_avg = sum(txt_scores) / len(txt_scores)
            lines.append(f"**结论**: 多模态优化策略（multimodal_step/fewshot）在图文场景下的图文关联度均分为 **{mm_avg:.2f}**，")
            lines.append(f"相比基础策略（basic/cot）的 **{txt_avg:.2f}**，" + ("显著提升" if mm_avg > txt_avg else "无显著优势") + "。")
            lines.append("")
    else:
        lines.append("> 本实验未涉及含图片的测试用例，图文关联度分析不适用。")
        lines.append("")

    # ── 5. 典型案例 ──
    lines.append("## 5. 典型案例摘录")
    lines.append("")

    # Find best and worst answers by total_score
    scored_results = [r for r in experiment.results if r.judge_result and not r.error]
    if scored_results:
        scored_results.sort(key=lambda r: r.judge_result.total_score, reverse=True)

        # Best case
        best = scored_results[0]
        lines.append("### 🌟 最佳回答")
        lines.append(f"- **策略**: {best.strategy_name}")
        lines.append(f"- **问题**: {best.question}")
        lines.append(f"- **得分**: 加权总分 {best.judge_result.total_score:.2f}")
        lines.append(f"- **评分**: 事实准确性={best.judge_result.factual_accuracy}, 图文关联度={best.judge_result.image_relevance}, 完整性={best.judge_result.completeness}, 简洁性={best.judge_result.conciseness}")
        lines.append(f"- **评审理由**: {best.judge_result.reasoning}")
        lines.append("")
        lines.append("<details><summary>查看完整回答</summary>")
        lines.append("")
        lines.append(best.answer)
        lines.append("")
        lines.append("</details>")
        lines.append("")

        # Worst case
        worst = scored_results[-1]
        lines.append("### ⚠️ 最差回答")
        lines.append(f"- **策略**: {worst.strategy_name}")
        lines.append(f"- **问题**: {worst.question}")
        lines.append(f"- **得分**: 加权总分 {worst.judge_result.total_score:.2f}")
        lines.append(f"- **评分**: 事实准确性={worst.judge_result.factual_accuracy}, 图文关联度={worst.judge_result.image_relevance}, 完整性={worst.judge_result.completeness}, 简洁性={worst.judge_result.conciseness}")
        lines.append(f"- **评审理由**: {worst.judge_result.reasoning}")
        lines.append("")
        lines.append("<details><summary>查看完整回答</summary>")
        lines.append("")
        lines.append(worst.answer)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # ── 6. 最优策略推荐 ──
    lines.append("## 6. 最优 Prompt 推荐")
    lines.append("")

    if ranked:
        best_strategy = ranked[0][0]
        best_scores = ranked[0][1]
        best_total = sum(best_scores["total_score"]) / len(best_scores["total_score"])

        lines.append(f"### 🏆 推荐策略: **{best_strategy}**")
        lines.append(f"加权总分: **{best_total:.2f}/5.00**")
        lines.append("")

        # Recommendations
        recommendations = []
        if "multimodal_step" in [s[0] for s in ranked[:2]]:
            recommendations.append("✅ **多模态场景优先使用 `multimodal_step` 策略**：该策略通过分步引导 LLM 先理解图片内容再结合文字，有效提升了图文关联度。")
        if "cot" in [s[0] for s in ranked[:2]]:
            recommendations.append("✅ **复杂推理问题使用 `cot` 策略**：Chain of Thought 在需要多步推理的问题上表现优异，建议用于分析性问题。")
        if "fewshot" in [s[0] for s in ranked[:2]]:
            recommendations.append("✅ **有高质量示例的场景使用 `fewshot` 策略**：Few-shot 示例帮助模型更好地理解输出格式和质量标准。")
        if "basic" == best_strategy:
            recommendations.append("⚠️ **基础策略表现最佳**：可能说明复杂 Prompt 在当前测试集上引入了额外噪音，建议优化 Few-shot 示例或 CoT 指令。")

        if not recommendations:
            recommendations.append("📝 **建议**: 根据具体场景选择合适的策略组合。对于图文混合场景推荐 `multimodal_step`，纯文本推理推荐 `cot`，标准化问答推荐 `basic`。")

        for rec in recommendations:
            lines.append(rec)
            lines.append("")

    # ── Footer ──
    lines.append("---")
    lines.append(f"*报告由多模态 RAG + Prompt 策略评估平台自动生成于 {datetime.now(timezone.utc).isoformat()}*")

    return "\n".join(lines)
