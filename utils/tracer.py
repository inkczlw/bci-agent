"""Agent 执行 tracing。

捕获每次 Agent 运行的完整轨迹：LLM 调用、tool 执行、token 消耗、耗时。
不依赖外部服务（LangSmith），纯本地实现。
"""

import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class Span:
    """一个执行单元（LLM 调用 or tool 调用）。"""
    span_id: str
    span_type: str          # "llm" | "tool"
    name: str               # tool 名 or "llm_call"
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    input_data: str = ""
    output_data: str = ""
    status: str = "ok"      # "ok" | "error" | "timeout"
    token_usage: dict = field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens
    metadata: dict = field(default_factory=dict)

    def finish(self, output: str = "", status: str = "ok"):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.output_data = output[:500]  # 截断，避免日志爆炸
        self.status = status

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "type": self.span_type,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "token_usage": self.token_usage,
            "input": self.input_data[:200],
            "output": self.output_data[:200],
        }


@dataclass
class Trace:
    """一次完整的 Agent 执行轨迹。"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    total_duration_ms: float = 0.0
    spans: list[Span] = field(default_factory=list)
    total_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0

    def finish(self):
        self.end_time = time.time()
        self.total_duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.total_llm_calls = sum(1 for s in self.spans if s.span_type == "llm")
        self.total_tool_calls = sum(1 for s in self.spans if s.span_type == "tool")
        self.total_tokens = sum(
            s.token_usage.get("total_tokens", 0) for s in self.spans
        )

    def summary(self) -> str:
        """一行式摘要，适合日志输出。"""
        tool_names = [s.name for s in self.spans if s.span_type == "tool"]
        return (
            f"[Trace {self.trace_id}] "
            f"duration={self.total_duration_ms}ms | "
            f"llm_calls={self.total_llm_calls} | "
            f"tool_calls={self.total_tool_calls} ({', '.join(tool_names) or 'none'}) | "
            f"tokens={self.total_tokens}"
        )

    def detail_report(self) -> str:
        """详细报告，开发调试用。"""
        lines = [
            f"\n{'='*70}",
            f"  Trace: {self.trace_id}",
            f"  Query: {self.query}",
            f"  Total: {self.total_duration_ms}ms | "
            f"LLM×{self.total_llm_calls} | Tool×{self.total_tool_calls} | "
            f"Tokens: {self.total_tokens}",
            f"{'='*70}",
        ]
        for i, span in enumerate(self.spans):
            status_icon = "✓" if span.status == "ok" else "✗"
            lines.append(
                f"  [{i+1}] {status_icon} {span.span_type:4s} | {span.name:25s} | "
                f"{span.duration_ms:>8.1f}ms | tokens={span.token_usage.get('total_tokens', '-')}"
            )
            if span.input_data:
                lines.append(f"       in:  {span.input_data[:100]}")
            if span.output_data:
                lines.append(f"       out: {span.output_data[:100]}")
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "llm_calls": self.total_llm_calls,
            "tool_calls": self.total_tool_calls,
            "spans": [s.to_dict() for s in self.spans],
        }


class AgentTracer(BaseCallbackHandler):
    """LangChain callback handler，自动捕获 LLM 和 tool 的执行数据。

    用法：
        tracer = AgentTracer()
        tracer.start_trace("用户的问题")
        result = agent.invoke({"messages": [...]}, config={"callbacks": [tracer]})
        tracer.end_trace()
        print(tracer.current_trace.detail_report())
    """

    def __init__(self, log_dir: str | None = None):
        super().__init__()
        self.current_trace: Trace | None = None
        self.traces: list[Trace] = []       # 历史 trace 记录
        self._active_spans: dict[str, Span] = {}  # run_id → Span，追踪进行中的 span
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def start_trace(self, query: str):
        """开始一次新的 trace。在 agent.invoke() 之前调用。"""
        self.current_trace = Trace(query=query)

    def end_trace(self) -> Trace:
        """结束当前 trace 并返回。在 agent.invoke() 之后调用。"""
        if self.current_trace:
            self.current_trace.finish()
            self.traces.append(self.current_trace)
            if self.log_dir:
                self._save_trace(self.current_trace)
        return self.current_trace

    # ---- LLM callbacks ----

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id, **kwargs):
        span = Span(
            span_id=str(run_id)[:8],
            span_type="llm",
            name="llm_call",
            start_time=time.time(),
            input_data=prompts[0][:200] if prompts else "",
        )
        self._active_spans[str(run_id)] = span

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id, **kwargs):
        # ChatOpenAI 走这个回调，不走 on_llm_start
        input_preview = ""
        if messages and messages[0]:
            last_msg = messages[0][-1] if isinstance(messages[0], list) else messages[0]
            if hasattr(last_msg, "content"):
                input_preview = str(last_msg.content)[:200]
        span = Span(
            span_id=str(run_id)[:8],
            span_type="llm",
            name="llm_call",
            start_time=time.time(),
            input_data=input_preview,
        )
        self._active_spans[str(run_id)] = span

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        span = self._active_spans.pop(str(run_id), None)
        if not span:
            return
        # 提取 token usage
        if response.llm_output and "token_usage" in response.llm_output:
            span.token_usage = dict(response.llm_output["token_usage"])
        # 提取输出文本
        output = ""
        if response.generations and response.generations[0]:
            output = response.generations[0][0].text or ""
            if not output and hasattr(response.generations[0][0], "message"):
                msg = response.generations[0][0].message
                output = str(msg.content)[:500] if msg.content else ""
        span.finish(output=output)
        if self.current_trace:
            self.current_trace.spans.append(span)

    def on_llm_error(self, error: Exception, *, run_id, **kwargs):
        span = self._active_spans.pop(str(run_id), None)
        if span:
            span.finish(output=str(error)[:200], status="error")
            if self.current_trace:
                self.current_trace.spans.append(span)

    # ---- Tool callbacks ----

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        span = Span(
            span_id=str(run_id)[:8],
            span_type="tool",
            name=tool_name,
            start_time=time.time(),
            input_data=input_str[:200],
        )
        self._active_spans[str(run_id)] = span

    def on_tool_end(self, output: str, *, run_id, **kwargs):
        span = self._active_spans.pop(str(run_id), None)
        if span:
            span.finish(output=str(output)[:500])
            if self.current_trace:
                self.current_trace.spans.append(span)

    def on_tool_error(self, error: Exception, *, run_id, **kwargs):
        span = self._active_spans.pop(str(run_id), None)
        if span:
            span.finish(output=str(error)[:200], status="error")
            if self.current_trace:
                self.current_trace.spans.append(span)

    # ---- 持久化 ----

    def _save_trace(self, trace: Trace):
        if not self.log_dir:
            return
        filepath = self.log_dir / f"trace_{trace.trace_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)

    # ---- 聚合查询 ----

    def get_stats(self) -> dict:
        """聚合所有历史 trace 的统计数据。"""
        if not self.traces:
            return {"total_traces": 0}

        durations = [t.total_duration_ms for t in self.traces]
        tokens = [t.total_tokens for t in self.traces]
        durations_sorted = sorted(durations)
        n = len(durations_sorted)

        return {
            "total_traces": n,
            "avg_duration_ms": round(sum(durations) / n, 2),
            "p50_duration_ms": durations_sorted[n // 2],
            "p95_duration_ms": durations_sorted[int(n * 0.95)] if n >= 20 else durations_sorted[-1],
            "p99_duration_ms": durations_sorted[int(n * 0.99)] if n >= 100 else durations_sorted[-1],
            "avg_tokens": round(sum(tokens) / n, 2),
            "total_tokens": sum(tokens),
        }