"""评估引擎。

职责：
1. 批量执行 test case
2. 捕获每次执行的完整数据（tool 调用链、输出、延迟、token）
3. 多维度评分：tool 选择、关键词覆盖、结构化完整性、延迟、LLM-as-judge

- 类似 C++ test harness：fixture(test case) → runner(evaluator) → assertion(scorer) → summary(report)
- 但 assertion 不是 binary pass/fail，而是 0-1 连续评分
- LLM-as-judge 用于评估无法用规则判断的维度（回答质量、逻辑连贯性）
"""

import time
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any

from evaluation.test_cases import TestCase, TestCategory

logger = logging.getLogger("bci_agent.evaluator")


# ── 评分结果 ─────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """单个维度的评分。"""
    dimension: str
    score: float          # 0.0 ~ 1.0
    max_score: float = 1.0
    details: str = ""     # 评分依据说明


@dataclass
class EvalResult:
    """单个 test case 的评估结果。"""
    test_id: str
    category: TestCategory
    query: str
    # ── 执行数据 ──
    agent_output: str = ""
    tools_called: list[str] = field(default_factory=list)
    tool_call_count: int = 0
    latency_seconds: float = 0.0
    token_usage: dict = field(default_factory=dict)
    error: str | None = None
    # ── 评分 ──
    scores: list[DimensionScore] = field(default_factory=list)
    total_score: float = 0.0
    max_total_score: float = 0.0
    weight: float = 1.0

    @property
    def weighted_score(self) -> float:
        if self.max_total_score == 0:
            return 0.0
        return (self.total_score / self.max_total_score) * self.weight

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "category": self.category.value,
            "query": self.query[:100],
            "agent_output": self.agent_output[:300],
            "tools_called": self.tools_called,
            "tool_call_count": self.tool_call_count,
            "latency_seconds": round(self.latency_seconds, 2),
            "error": self.error,
            "scores": [
                {"dimension": s.dimension, "score": s.score, "details": s.details}
                for s in self.scores
            ],
            "total_score": round(self.total_score, 2),
            "max_total_score": round(self.max_total_score, 2),
        }


# ── 评分器 ───────────────────────────────────────────────────

class Scorer:
    """多维度评分器。"""

    @staticmethod
    def score_tool_selection(
        called: list[str],
        expected: list[str],
        forbidden: list[str],
    ) -> DimensionScore:
        """评估 tool 选择是否正确。"""
        if not expected and not forbidden:
            return DimensionScore(
                dimension="tool_selection",
                score=1.0,
                details="无 tool 约束，跳过",
            )

        penalties = []
        total_checks = 0

        # 期望的 tool 是否被调用
        if expected:
            hit = sum(1 for t in expected if t in called)
            total_checks += len(expected)
            if hit < len(expected):
                missed = [t for t in expected if t not in called]
                penalties.append(f"缺失: {missed}")
        else:
            hit = 0

        # 禁止的 tool 是否被调用
        forbidden_hits = [t for t in forbidden if t in called]
        if forbidden_hits:
            total_checks += len(forbidden)
            penalties.append(f"误调用禁止 tool: {forbidden_hits}")
        elif forbidden:
            total_checks += len(forbidden)
            hit += len(forbidden)  # 正确地没调用禁止的 tool

        score = hit / total_checks if total_checks > 0 else 1.0
        details = "; ".join(penalties) if penalties else "tool 选择正确"

        return DimensionScore(
            dimension="tool_selection",
            score=round(score, 2),
            details=details,
        )

    @staticmethod
    def score_keyword_coverage(
        output: str,
        expected_keywords: list[str],
    ) -> DimensionScore:
        """评估输出是否包含期望关键词。"""
        if not expected_keywords:
            return DimensionScore(
                dimension="keyword_coverage",
                score=1.0,
                details="无关键词约束",
            )

        output_lower = output.lower()
        hits = []
        misses = []
        for kw in expected_keywords:
            if kw.lower() in output_lower:
                hits.append(kw)
            else:
                misses.append(kw)

        score = len(hits) / len(expected_keywords)
        details = f"命中: {hits}" if not misses else f"命中: {hits}, 缺失: {misses}"

        return DimensionScore(
            dimension="keyword_coverage",
            score=round(score, 2),
            details=details,
        )

    @staticmethod
    def score_field_completeness(
        output: str,
        expected_fields: list[str],
    ) -> DimensionScore:
        """评估结构化输出字段完整性。

        检查输出中是否包含期望的字段名（适用于 JSON 或自然语言中提到字段的情况）。
        """
        if not expected_fields:
            return DimensionScore(
                dimension="field_completeness",
                score=1.0,
                details="无字段约束",
            )

        # 尝试从输出中提取 JSON
        output_lower = output.lower()
        hits = []
        misses = []

        for f in expected_fields:
            # 检查字段名出现在输出中（JSON key 或自然语言提及）
            f_lower = f.lower()
            # 把 snake_case 转换为可能的中文表述也检查
            f_readable = f_lower.replace("_", " ").replace("name", "名").replace("technology", "技术")
            if f_lower in output_lower or f_readable in output_lower:
                hits.append(f)
            else:
                misses.append(f)

        score = len(hits) / len(expected_fields)
        details = f"命中: {hits}" if not misses else f"命中: {hits}, 缺失: {misses}"

        return DimensionScore(
            dimension="field_completeness",
            score=round(score, 2),
            details=details,
        )

    @staticmethod
    def score_latency(
        actual_seconds: float,
        max_seconds: float,
    ) -> DimensionScore:
        """评估延迟是否在阈值内。"""
        if actual_seconds <= max_seconds:
            # 越快分越高
            ratio = actual_seconds / max_seconds
            score = 1.0 - (ratio * 0.3)  # 最快得 1.0，刚好踩线得 0.7
        else:
            # 超时，按超出比例扣分
            overshoot = actual_seconds / max_seconds
            score = max(0.0, 0.7 - (overshoot - 1.0) * 0.5)

        return DimensionScore(
            dimension="latency",
            score=round(score, 2),
            details=f"实际 {actual_seconds:.1f}s / 阈值 {max_seconds:.1f}s",
        )

    @staticmethod
    def score_tool_call_efficiency(
        actual_calls: int,
        max_calls: int,
    ) -> DimensionScore:
        """评估 tool 调用次数是否合理（防止无限循环）。"""
        if actual_calls <= max_calls:
            score = 1.0
            details = f"{actual_calls} 次调用，在阈值 {max_calls} 内"
        else:
            score = max(0.0, 1.0 - (actual_calls - max_calls) / max_calls)
            details = f"{actual_calls} 次调用，超出阈值 {max_calls}"

        return DimensionScore(
            dimension="tool_efficiency",
            score=round(score, 2),
            details=details,
        )

    @staticmethod
    def score_error_handling(
        error: str | None,
        query: str,
    ) -> DimensionScore:
        """评估是否有未捕获的异常。"""
        if error is None:
            return DimensionScore(
                dimension="error_handling",
                score=1.0,
                details="无异常",
            )
        else:
            return DimensionScore(
                dimension="error_handling",
                score=0.0,
                details=f"异常: {error[:200]}",
            )


# ── LLM-as-Judge 评分器 ─────────────────────────────────────

class LLMJudge:
    """用 LLM 评估回答质量（无法用规则判断的维度）。

    评估维度：
    - 回答相关性：是否回答了用户的问题
    - 逻辑连贯性：是否有条理
    - 信息密度：是否有实质内容（vs 空泛套话）
    """

    JUDGE_PROMPT = """你是一个 AI Agent 回答质量评估专家。请对以下 Agent 回答进行评分。

用户问题：{query}

Agent 回答：{output}

请从以下三个维度评分（每个维度 1-5 分）：
1. 相关性：回答是否紧扣用户问题（1=完全跑题，5=精准回答）
2. 连贯性：逻辑是否通顺、有条理（1=混乱，5=清晰有条理）
3. 信息密度：是否有实质性内容（1=空泛套话，5=信息丰富有价值）

请只返回 JSON 格式，不要其他内容：
{{"relevance": <1-5>, "coherence": <1-5>, "density": <1-5>, "comment": "<一句话评价>"}}"""

    def __init__(self, llm=None):
        self.llm = llm

    def judge(self, query: str, output: str) -> DimensionScore:
        """用 LLM 评估回答质量。"""
        if not self.llm or not output.strip():
            return DimensionScore(
                dimension="llm_judge",
                score=0.5,
                details="LLM judge 未启用或输出为空",
            )

        try:
            prompt = self.JUDGE_PROMPT.format(query=query, output=output[:1500])
            response = self.llm.invoke(prompt)

            # 防御性 JSON 解析（复用项目中的策略）
            text = response.content if hasattr(response, "content") else str(response)
            score_data = self._parse_judge_response(text)

            # 三个维度取平均，归一化到 0-1
            avg = (score_data["relevance"] + score_data["coherence"] + score_data["density"]) / 15.0
            comment = score_data.get("comment", "")

            return DimensionScore(
                dimension="llm_judge",
                score=round(avg, 2),
                details=f"相关{score_data['relevance']}/连贯{score_data['coherence']}/密度{score_data['density']} | {comment}",
            )

        except Exception as e:
            logger.warning(f"LLM judge 评分失败: {e}")
            return DimensionScore(
                dimension="llm_judge",
                score=0.5,
                details=f"评分异常: {str(e)[:100]}",
            )

    def _parse_judge_response(self, text: str) -> dict:
        """防御性解析 LLM judge 的 JSON 输出。"""
        import re

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 提取 JSON 块
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # fallback：默认中等分
        logger.warning(f"无法解析 judge 输出: {text[:200]}")
        return {"relevance": 3, "coherence": 3, "density": 3, "comment": "解析失败，默认分"}


# ── 评估引擎 ─────────────────────────────────────────────────

class EvaluationEngine:
    """评估引擎：执行 test case + 评分。

    用法：
        engine = EvaluationEngine(agent, llm=get_llm())
        results = engine.run(test_cases)
    """

    def __init__(self, agent, llm=None, verbose: bool = True):
        """
        Args:
            agent: LangGraph Agent（可 invoke 的对象）
            llm: 用于 LLM-as-judge 的模型实例（可选）
            verbose: 是否打印进度
        """
        self.agent = agent
        self.scorer = Scorer()
        self.judge = LLMJudge(llm=llm)
        self.verbose = verbose

    def run(self, test_cases: list[TestCase]) -> list[EvalResult]:
        """批量执行评估。"""
        results = []

        for i, tc in enumerate(test_cases, 1):
            if self.verbose:
                print(f"\n[{i}/{len(test_cases)}] 执行: {tc.id} — {tc.query[:50]}...")

            result = self._run_single(tc)
            self._score(tc, result)
            results.append(result)

            if self.verbose:
                status = "✅" if result.total_score / max(result.max_total_score, 1) >= 0.6 else "⚠️"
                print(f"  {status} 得分: {result.total_score:.1f}/{result.max_total_score:.1f} "
                      f"| 延迟: {result.latency_seconds:.1f}s "
                      f"| tools: {result.tools_called}")

        return results

    def _run_single(self, tc: TestCase) -> EvalResult:
        """执行单个 test case。"""
        result = EvalResult(
            test_id=tc.id,
            category=tc.category,
            query=tc.query,
            weight=tc.weight,
        )

        if not tc.query.strip():
            # 空输入 edge case
            result.error = None
            result.agent_output = "(空输入测试)"
            result.latency_seconds = 0.0
            return result

        start = time.time()
        try:
            agent_result = self.agent.invoke(
                {"messages": [{"role": "user", "content": tc.query}]}
            )
            result.latency_seconds = time.time() - start

            # 提取输出
            messages = agent_result.get("messages", [])
            if messages:
                result.agent_output = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

            # 提取 tool 调用信息
            result.tools_called = self._extract_tool_calls(messages)
            result.tool_call_count = len(result.tools_called)

        except Exception as e:
            result.latency_seconds = time.time() - start
            result.error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Test {tc.id} 执行失败: {result.error}")
            logger.debug(traceback.format_exc())

        return result

    def _extract_tool_calls(self, messages: list) -> list[str]:
        """从 Agent 执行的 messages 中提取 tool 调用名称。"""
        tool_names = []
        for msg in messages:
            # LangGraph 的 AIMessage 中包含 tool_calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    if name:
                        tool_names.append(name)
            # ToolMessage 的 name 属性
            if hasattr(msg, "name") and hasattr(msg, "content") and msg.__class__.__name__ == "ToolMessage":
                if msg.name and msg.name not in tool_names:
                    tool_names.append(msg.name)
        return tool_names

    def _score(self, tc: TestCase, result: EvalResult):
        """对 test case 结果进行多维度评分。"""
        scores = []

        # 1. Tool 选择正确性
        scores.append(self.scorer.score_tool_selection(
            called=result.tools_called,
            expected=tc.expected_tools,
            forbidden=tc.forbidden_tools,
        ))

        # 2. 关键词覆盖
        scores.append(self.scorer.score_keyword_coverage(
            output=result.agent_output,
            expected_keywords=tc.expected_keywords,
        ))

        # 3. 结构化字段完整性
        scores.append(self.scorer.score_field_completeness(
            output=result.agent_output,
            expected_fields=tc.expected_fields,
        ))

        # 4. 延迟
        if result.latency_seconds > 0:
            scores.append(self.scorer.score_latency(
                actual_seconds=result.latency_seconds,
                max_seconds=tc.max_latency_seconds,
            ))

        # 5. Tool 调用效率
        scores.append(self.scorer.score_tool_call_efficiency(
            actual_calls=result.tool_call_count,
            max_calls=tc.max_tool_calls,
        ))

        # 6. 错误处理
        scores.append(self.scorer.score_error_handling(
            error=result.error,
            query=tc.query,
        ))

        # 7. LLM-as-judge（仅对有输出的 case）
        if result.agent_output and not result.error:
            scores.append(self.judge.judge(
                query=tc.query,
                output=result.agent_output,
            ))

        result.scores = scores
        result.total_score = sum(s.score for s in scores)
        result.max_total_score = len(scores) * 1.0