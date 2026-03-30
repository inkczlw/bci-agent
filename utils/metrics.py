"""Agent 指标采集与聚合。

从 Tracer 的 trace 数据中提取关键指标：
- 延迟分位数（p50/p95/p99）
- Token 消耗统计
- Tool 调用成功率
- 每个 tool 的平均耗时

支持输出为 JSON 报告，也可以接入 Prometheus（预留接口）。
"""

import json
import time
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path


@dataclass
class ToolMetrics:
    """单个 tool 的聚合指标。"""
    name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    durations_ms: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_duration_ms(self) -> float:
        return sum(self.durations_ms) / len(self.durations_ms) if self.durations_ms else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "calls": self.call_count,
            "success_rate": round(self.success_rate, 4),
            "errors": self.error_count,
            "timeouts": self.timeout_count,
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "p50_duration_ms": self._percentile(50),
            "p95_duration_ms": self._percentile(95),
        }

    def _percentile(self, pct: int) -> float:
        if not self.durations_ms:
            return 0.0
        sorted_d = sorted(self.durations_ms)
        idx = int(len(sorted_d) * pct / 100)
        idx = min(idx, len(sorted_d) - 1)
        return round(sorted_d[idx], 2)


class MetricsCollector:
    """从 Tracer 数据中采集和聚合 Agent 指标。

    用法：
        collector = MetricsCollector()
        # 每次 trace 结束后 feed 进来
        collector.record_trace(trace)
        # 查看报告
        print(collector.report())
    """

    def __init__(self):
        self.tool_metrics: dict[str, ToolMetrics] = {}
        self.trace_durations_ms: list[float] = []
        self.trace_token_counts: list[int] = []
        self.llm_durations_ms: list[float] = []
        self.llm_token_counts: list[int] = []
        self.total_traces: int = 0
        self._start_time: float = time.time()

    def record_trace(self, trace) -> None:
        """消化一个 Trace 对象，更新所有指标。

        Args:
            trace: utils.tracer.Trace 实例
        """
        self.total_traces += 1
        self.trace_durations_ms.append(trace.total_duration_ms)
        self.trace_token_counts.append(trace.total_tokens)

        for span in trace.spans:
            if span.span_type == "tool":
                self._record_tool_span(span)
            elif span.span_type == "llm":
                self.llm_durations_ms.append(span.duration_ms)
                total = span.token_usage.get("total_tokens", 0)
                if total:
                    self.llm_token_counts.append(total)

    def _record_tool_span(self, span) -> None:
        name = span.name
        if name not in self.tool_metrics:
            self.tool_metrics[name] = ToolMetrics(name=name)
        tm = self.tool_metrics[name]
        tm.call_count += 1
        tm.durations_ms.append(span.duration_ms)
        if span.status == "ok":
            tm.success_count += 1
        elif span.status == "timeout":
            tm.timeout_count += 1
        else:
            tm.error_count += 1

    def report(self) -> str:
        """生成可读的指标报告。"""
        lines = [
            "\n" + "="*70,
            "  Agent Metrics Report",
            "="*70,
            "",
            f"  运行时长: {round((time.time() - self._start_time), 1)}s",
            f"  总 trace 数: {self.total_traces}",
            "",
            "  --- 端到端延迟 ---",
            f"  avg:  {self._avg(self.trace_durations_ms):.1f}ms",
            f"  p50:  {self._percentile(self.trace_durations_ms, 50):.1f}ms",
            f"  p95:  {self._percentile(self.trace_durations_ms, 95):.1f}ms",
            f"  p99:  {self._percentile(self.trace_durations_ms, 99):.1f}ms",
            "",
            "  --- LLM 调用 ---",
            f"  总调用次数: {len(self.llm_durations_ms)}",
            f"  avg 耗时: {self._avg(self.llm_durations_ms):.1f}ms",
            f"  avg tokens: {self._avg(self.llm_token_counts):.0f}",
            f"  总 token 消耗: {sum(self.llm_token_counts)}",
            "",
            "  --- Tool 调用 ---",
        ]
        for tm in sorted(self.tool_metrics.values(), key=lambda x: x.call_count, reverse=True):
            lines.append(
                f"  {tm.name:25s} | calls={tm.call_count:3d} | "
                f"success={tm.success_rate:.0%} | avg={tm.avg_duration_ms:.1f}ms"
            )
        lines.append("="*70 + "\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """导出为 JSON 友好的 dict，可以喂给 Prometheus/Grafana。"""
        return {
            "total_traces": self.total_traces,
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "latency": {
                "avg_ms": self._avg(self.trace_durations_ms),
                "p50_ms": self._percentile(self.trace_durations_ms, 50),
                "p95_ms": self._percentile(self.trace_durations_ms, 95),
                "p99_ms": self._percentile(self.trace_durations_ms, 99),
            },
            "llm": {
                "total_calls": len(self.llm_durations_ms),
                "avg_duration_ms": self._avg(self.llm_durations_ms),
                "total_tokens": sum(self.llm_token_counts),
            },
            "tools": {name: tm.to_dict() for name, tm in self.tool_metrics.items()},
        }

    def save_report(self, filepath: str = "metrics_report.json"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _avg(values: list) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _percentile(values: list, pct: int) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(int(len(s) * pct / 100), len(s) - 1)
        return s[idx]