"""评估报告生成。

将评估结果汇总为可读报告 + JSON 数据文件。
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from evaluation.test_cases import TestCategory
from evaluation.evaluator import EvalResult


@dataclass
class CategorySummary:
    """按类别汇总的评估结果。"""
    category: str
    case_count: int
    avg_score: float
    avg_latency: float
    pass_rate: float
    worst_case: str
    details: str


class EvalReport:
    """评估报告生成器。"""

    PASS_THRESHOLD = 0.6    # 得分率 >= 60% 算通过

    def __init__(self, results: list[EvalResult]):
        self.results = results
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

    def generate_console_report(self) -> str:
        """生成控制台可读报告。"""
        lines = []
        lines.append("=" * 70)
        lines.append("  BCI Agent 质量评估报告")
        lines.append(f"  时间: {self.timestamp}")
        lines.append(f"  用例数: {len(self.results)}")
        lines.append("=" * 70)

        # 总体概览
        overall = self._calc_overall()
        lines.append(f"\n总体得分率: {overall['score_rate']:.1%}")
        lines.append(f"通过率 (>={self.PASS_THRESHOLD:.0%}): {overall['pass_rate']:.1%}")
        lines.append(f"平均延迟: {overall['avg_latency']:.1f}s")
        lines.append(f"错误数: {overall['error_count']}/{len(self.results)}")

        # 按类别汇总
        lines.append(f"\n{'─' * 70}")
        lines.append("按类别汇总:")
        lines.append(f"{'─' * 70}")

        summaries = self._summarize_by_category()
        for s in summaries:
            icon = "✅" if s.pass_rate >= 0.8 else ("⚠️" if s.pass_rate >= 0.5 else "❌")
            lines.append(
                f"  {icon} {s.category:<15} "
                f"得分率: {s.avg_score:.1%}  "
                f"通过: {s.pass_rate:.0%}  "
                f"延迟: {s.avg_latency:.1f}s  "
                f"({s.case_count} cases)"
            )
            if s.worst_case:
                lines.append(f"     └ 最弱项: {s.worst_case} — {s.details}")

        # 按维度汇总
        lines.append(f"\n{'─' * 70}")
        lines.append("按评分维度汇总:")
        lines.append(f"{'─' * 70}")

        dim_scores = self._summarize_by_dimension()
        for dim, avg_score in sorted(dim_scores.items(), key=lambda x: x[1]):
            bar = "█" * int(avg_score * 20) + "░" * (20 - int(avg_score * 20))
            lines.append(f"  {dim:<20} {bar} {avg_score:.2f}")

        # 详细结果
        lines.append(f"\n{'─' * 70}")
        lines.append("详细结果:")
        lines.append(f"{'─' * 70}")

        for r in self.results:
            score_rate = r.total_score / max(r.max_total_score, 1)
            icon = "✅" if score_rate >= self.PASS_THRESHOLD else "❌"
            lines.append(
                f"\n  {icon} [{r.test_id}] {r.query[:60]}"
            )
            lines.append(
                f"     得分: {r.total_score:.1f}/{r.max_total_score:.1f} ({score_rate:.1%}) "
                f"| 延迟: {r.latency_seconds:.1f}s "
                f"| tools: {r.tools_called}"
            )
            if r.error:
                lines.append(f"     ⚠️ 错误: {r.error[:100]}")

            for s in r.scores:
                if s.score < 1.0:
                    lines.append(f"     - {s.dimension}: {s.score:.2f} — {s.details}")

        lines.append(f"\n{'=' * 70}")
        return "\n".join(lines)

    def save_json(self, output_dir: str = "evaluation/results") -> str:
        """保存 JSON 格式的详细结果。"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = output_path / f"eval_{self.timestamp}.json"

        data = {
            "timestamp": self.timestamp,
            "total_cases": len(self.results),
            "overall": self._calc_overall(),
            "by_category": [
                {
                    "category": s.category,
                    "case_count": s.case_count,
                    "avg_score": round(s.avg_score, 3),
                    "avg_latency": round(s.avg_latency, 2),
                    "pass_rate": round(s.pass_rate, 3),
                }
                for s in self._summarize_by_category()
            ],
            "by_dimension": self._summarize_by_dimension(),
            "results": [r.to_dict() for r in self.results],
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(filename)

    def _calc_overall(self) -> dict:
        """计算总体指标。"""
        if not self.results:
            return {"score_rate": 0, "pass_rate": 0, "avg_latency": 0, "error_count": 0}

        score_rates = [
            r.total_score / max(r.max_total_score, 1)
            for r in self.results
        ]

        return {
            "score_rate": round(sum(score_rates) / len(score_rates), 3),
            "pass_rate": round(
                sum(1 for s in score_rates if s >= self.PASS_THRESHOLD) / len(score_rates), 3
            ),
            "avg_latency": round(
                sum(r.latency_seconds for r in self.results) / len(self.results), 2
            ),
            "error_count": sum(1 for r in self.results if r.error),
        }

    def _summarize_by_category(self) -> list[CategorySummary]:
        """按 TestCategory 分组汇总。"""
        from collections import defaultdict
        groups: dict[str, list[EvalResult]] = defaultdict(list)

        for r in self.results:
            groups[r.category.value].append(r)

        summaries = []
        for cat, results in groups.items():
            score_rates = [r.total_score / max(r.max_total_score, 1) for r in results]
            worst = min(results, key=lambda r: r.total_score / max(r.max_total_score, 1))
            worst_rate = worst.total_score / max(worst.max_total_score, 1)

            # 找最弱维度
            worst_dim = ""
            if worst.scores:
                lowest = min(worst.scores, key=lambda s: s.score)
                worst_dim = f"{lowest.dimension}={lowest.score:.2f}"

            summaries.append(CategorySummary(
                category=cat,
                case_count=len(results),
                avg_score=sum(score_rates) / len(score_rates),
                avg_latency=sum(r.latency_seconds for r in results) / len(results),
                pass_rate=sum(1 for s in score_rates if s >= self.PASS_THRESHOLD) / len(score_rates),
                worst_case=f"{worst.test_id} ({worst_rate:.0%})",
                details=worst_dim,
            ))

        return sorted(summaries, key=lambda s: s.avg_score)

    def _summarize_by_dimension(self) -> dict[str, float]:
        """按评分维度汇总平均分。"""
        from collections import defaultdict
        dim_scores: dict[str, list[float]] = defaultdict(list)

        for r in self.results:
            for s in r.scores:
                dim_scores[s.dimension].append(s.score)

        return {
            dim: round(sum(scores) / len(scores), 3)
            for dim, scores in dim_scores.items()
        }