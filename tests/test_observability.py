"""测试 Structured Logging + Metrics — 独立验证。

验证：
1. Logger 双 formatter — 控制台可读 + 文件 JSON
2. log_event 结构化字段输出
3. MetricsCollector 从 trace 数据聚合指标
4. 聚合报告输出 + JSON 持久化

依赖 tracer 已验证通过（test_tracer.py）。

用法：python -m tests.test_observability
      python -m tests.test_observability logger
      python -m tests.test_observability metrics
      python -m tests.test_observability full
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.bci_agent import create_bci_agent
from utils.tracer import AgentTracer
from utils.logger import setup_logger, log_event
from utils.metrics import MetricsCollector


# ── 1. Logger 验证 ──────────────────────────────────────────

def test_logger():
    """验证 logger 双输出：控制台彩色 + 文件 JSONL。"""
    print("\n" + "="*60)
    print("  Test: Structured Logger")
    print("="*60)

    log_dir = "logs"
    logger = setup_logger("test_logger", log_dir=log_dir, dev_mode=True)

    # 各级别日志
    log_event(logger, "agent started", mode="test")
    log_event(logger, "tool call completed",
              tool_name="search_bci_company", duration_ms=12.3, status="ok")
    log_event(logger, "llm call slow", level="warning",
              duration_ms=8500, tokens=2100)
    log_event(logger, "tool timeout", level="error",
              tool_name="rag_search", timeout_seconds=10)

    # 验证 JSONL 文件
    jsonl_path = Path(log_dir) / "test_logger.jsonl"
    assert jsonl_path.exists(), f"JSONL 文件不存在: {jsonl_path}"

    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"\n  JSONL 文件: {jsonl_path} ({len(lines)} 行)")

    # 验证每行都是 valid JSON
    for i, line in enumerate(lines[-4:]):  # 只检查刚写入的 4 行
        data = json.loads(line.strip())
        assert "timestamp" in data, f"第{i}行缺少 timestamp"
        assert "level" in data, f"第{i}行缺少 level"
        assert "message" in data, f"第{i}行缺少 message"
        print(f"  [{i+1}] {data['level']:7s} | {data['message']}")

    print("  ✓ Logger test passed")


# ── 2. Metrics 验证 ─────────────────────────────────────────

def test_metrics():
    """验证 MetricsCollector 从 trace 聚合指标。"""
    print("\n" + "="*60)
    print("  Test: MetricsCollector")
    print("="*60)

    agent = create_bci_agent()
    tracer = AgentTracer(log_dir="traces")
    metrics = MetricsCollector()

    queries = [
        "Neuralink 用的是什么技术？",
        "BrainCo 的主要产品有哪些？",
    ]

    for q in queries:
        tracer.start_trace(q)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config={"callbacks": [tracer]},
        )
        trace = tracer.end_trace()
        metrics.record_trace(trace)

    # 验证聚合数据
    assert metrics.total_traces == 2, f"应有 2 个 trace，实际: {metrics.total_traces}"
    assert len(metrics.trace_durations_ms) == 2
    assert len(metrics.llm_durations_ms) >= 2, "至少 2 次 LLM 调用"

    # 输出报告
    print(metrics.report())

    # 验证 tool 级别指标
    if metrics.tool_metrics:
        print("  Tool 级别指标:")
        for name, tm in metrics.tool_metrics.items():
            print(f"    {name}: calls={tm.call_count}, "
                  f"success={tm.success_rate:.0%}, avg={tm.avg_duration_ms:.1f}ms")

    # 持久化
    report_path = "logs/test_metrics_report.json"
    metrics.save_report(report_path)
    assert Path(report_path).exists(), f"报告文件不存在: {report_path}"

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_traces"] == 2
    print(f"\n  报告已保存: {report_path}")

    print("  ✓ Metrics test passed")


# ── 3. 全链路验证 ───────────────────────────────────────────

def test_full():
    """完整链路：tracer + logger + metrics 联合工作。"""
    print("\n" + "="*60)
    print("  Test: Full Observability Stack")
    print("="*60)

    agent = create_bci_agent()
    logger = setup_logger("full_test", log_dir="logs", dev_mode=True)
    tracer = AgentTracer(log_dir="traces")
    metrics = MetricsCollector()

    log_event(logger, "full observability test started")

    queries = [
        "介绍一下 Neuralink 的核心技术",
        "它和 BrainCo 有什么区别？",
        "对比一下两家公司的技术路线",
    ]

    for q in queries:
        log_event(logger, "query received", query=q)

        tracer.start_trace(q)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config={"callbacks": [tracer]},
        )
        trace = tracer.end_trace()
        metrics.record_trace(trace)

        # 结构化日志记录每次交互
        log_event(logger, "query completed",
                  trace_id=trace.trace_id,
                  duration_ms=trace.total_duration_ms,
                  llm_calls=trace.total_llm_calls,
                  tool_calls=trace.total_tool_calls,
                  tokens=trace.total_tokens,
                  tools_used=[s.name for s in trace.spans if s.span_type == "tool"])

        print(f"\n  {trace.summary()}")

    # 最终报告
    print(metrics.report())
    metrics.save_report("logs/full_test_metrics.json")

    log_event(logger, "test completed",
              total_traces=metrics.total_traces,
              total_tokens=sum(metrics.llm_token_counts))

    # 验证
    assert metrics.total_traces == 3
    assert Path("logs/full_test.jsonl").exists()
    assert Path("logs/full_test_metrics.json").exists()

    print("  ✓ Full stack test passed")


# ── 入口 ────────────────────────────────────────────────────

TEST_MAP = {
    "logger": test_logger,
    "metrics": test_metrics,
    "full": test_full,
}

if __name__ == "__main__":
    # test_logger()
    # test_metrics()
    test_full()
