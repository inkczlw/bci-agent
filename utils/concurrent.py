"""并发 Tool 执行器。

支持多个无依赖关系的 tool call 并行执行，降低 Agent 端到端延迟。
设计思路：检测同一轮 ReAct 循环中 LLM 返回的多个 tool_calls，
用 ThreadPoolExecutor 并发执行，替代默认的顺序执行。

- C++ 类比：std::async + std::future::wait_for 的 Python 版本
- 依赖 DAG 类比：无边相连的节点可并发调度
- 生产考量：per-tool timeout 隔离、单 tool 失败不影响其他 tool

嵌套线程池的做法能用但有代价——每个 tool call 都会创建一个额外的线程池。如果并发量大，线程数会翻倍。
对于 Agent 场景（通常一轮就几个 tool call）问题不大，但如果 scale 上去需要考虑换成 asyncio 或者用单个线程池 + Future.result(timeout) 直接做。
依赖分析目前是硬编码的（dependent_tools 集合），不是真正的 DAG 拓扑排序。如果后续 tool 间的依赖关系变复杂，需要升级成显式的依赖图。
analyze_parallelism 用超时值而不是实际耗时做估算，只能给个粗略参考。
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("bci_agent.concurrent")


# ── 每个 tool 的超时配置（秒）──────────────────────────────────
# 按预期延迟分档：本地查询 < RAG 检索 < LLM 分析 < 多步对比
TOOL_TIMEOUTS: dict[str, int] = {
    "search_bci_company": 4,
    "search_bci_news": 4,
    "rag_search": 7,
    "analyze_bci_company": 30,
    "compare_bci_companies": 60,
}
DEFAULT_TIMEOUT = 15


@dataclass
class ToolResult:
    """单个 tool 的执行结果。"""
    tool_name: str
    args: dict
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    status: str = "ok"  # "ok" | "error" | "timeout"


def get_timeout(tool_name: str) -> int:
    """获取 tool 的超时时间。"""
    return TOOL_TIMEOUTS.get(tool_name, DEFAULT_TIMEOUT)


def execute_tools_concurrent(
    tool_calls: list[dict],
    tool_map: dict[str, Callable],
    max_workers: int = 4,
) -> list[ToolResult]:
    """并发执行多个 tool call。

    Args:
        tool_calls: [{"name": "search_bci_company", "args": {"company_name": "Neuralink"}}, ...]
        tool_map: {"search_bci_company": <callable>, ...}
        max_workers: 最大并发线程数

    Returns:
        按原始顺序返回的 ToolResult 列表
    """
    if not tool_calls:
        return []

    # 只有一个 tool call 时直接执行，省去线程池开销
    if len(tool_calls) == 1:
        return [_execute_single(tool_calls[0], tool_map)]

    results: dict[int, ToolResult] = {}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(tool_calls))) as executor:
        # 提交所有任务
        future_to_index = {}
        for i, call in enumerate(tool_calls):
            future = executor.submit(_execute_single, call, tool_map)
            future_to_index[future] = i

        # 收集结果（as_completed 返回最先完成的 future）
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                # executor.submit 本身的异常（不太可能，但防御一下）
                results[idx] = ToolResult(
                    tool_name=tool_calls[idx].get("name", "unknown"),
                    args=tool_calls[idx].get("args", {}),
                    error=str(e),
                    status="error",
                )

    # 按原始顺序返回
    return [results[i] for i in range(len(tool_calls))]


def _execute_single(call: dict, tool_map: dict[str, Callable]) -> ToolResult:
    """执行单个 tool，带超时和异常捕获。"""
    name = call.get("name", "unknown")
    args = call.get("args", {})
    timeout = get_timeout(name)

    result = ToolResult(tool_name=name, args=args)
    start = time.time()

    tool_fn = tool_map.get(name)
    if tool_fn is None:
        result.error = f"Tool '{name}' not found in tool_map"
        result.status = "error"
        result.duration_ms = round((time.time() - start) * 1000, 2)
        logger.error(f"Tool not found: {name}")
        return result

    try:
        # 用嵌套 ThreadPoolExecutor 实现 per-tool timeout
        # 这样单个 tool 超时不会阻塞其他 tool
        with ThreadPoolExecutor(max_workers=1) as single:
            future = single.submit(tool_fn.invoke, args)
            result.result = future.result(timeout=timeout)
            result.status = "ok"
    except TimeoutError:
        result.error = f"Tool '{name}' timed out after {timeout}s"
        result.status = "timeout"
        logger.warning(f"Tool timeout: {name} ({timeout}s)")
    except Exception as e:
        result.error = str(e)
        result.status = "error"
        logger.error(f"Tool error: {name} - {e}")

    result.duration_ms = round((time.time() - start) * 1000, 2)
    return result


def analyze_parallelism(tool_calls: list[dict]) -> dict:
    """分析一组 tool calls 的并行度。

    Returns:
        {
            "total_calls": 3,
            "parallelizable": 2,  # 可并行的数量
            "serial_estimate_ms": 15000,
            "parallel_estimate_ms": 10000,
            "speedup_ratio": 1.5,
        }
    """
    if not tool_calls:
        return {"total_calls": 0, "parallelizable": 0}

    # 简单的依赖分析：同名 tool 的多次调用可并行
    # comparator 依赖 analyzer 的结果，不能并行
    dependent_tools = {"compare_bci_companies"}  # 需要前序结果的 tool
    parallelizable = [c for c in tool_calls if c.get("name") not in dependent_tools]
    serial_only = [c for c in tool_calls if c.get("name") in dependent_tools]

    # 估算延迟
    serial_ms = sum(get_timeout(c["name"]) * 1000 for c in tool_calls)
    parallel_group_ms = max(
        (get_timeout(c["name"]) * 1000 for c in parallelizable), default=0
    )
    serial_group_ms = sum(get_timeout(c["name"]) * 1000 for c in serial_only)
    parallel_total_ms = parallel_group_ms + serial_group_ms

    return {
        "total_calls": len(tool_calls),
        "parallelizable": len(parallelizable),
        "serial_only": len(serial_only),
        "serial_estimate_ms": serial_ms,
        "parallel_estimate_ms": parallel_total_ms,
        "speedup_ratio": round(serial_ms / parallel_total_ms, 2) if parallel_total_ms > 0 else 1.0,
    }