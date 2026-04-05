"""批量分析编排器。

将"分析 BCI 行业 top N 公司"这种高层任务拆解为 N 个独立的 Agent 调用，
通过 AsyncTaskQueue 并发执行，汇总结果生成对比报告。

"批量任务拆解成独立子任务 → 并发执行 → 结果聚合，
这跟我 C++ 里做的依赖 DAG 调度是同一个模式：
无依赖的节点并发 dispatch，最后 barrier 同步汇总。"
"""

import asyncio
import time
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from async_tasks.task_queue import AsyncTaskQueue, TaskResult, TaskStatus

logger = logging.getLogger("bci_agent.batch_analyzer")


# ── 默认 BCI 公司列表 ────────────────────────────────────────

DEFAULT_BCI_COMPANIES = [
    "Neuralink",
    "BrainCo",
    "Synchron",
    "Blackrock Neurotech",
    "Kernel",
    "Paradromics",
    "Precision Neuroscience",
    "BrainGate",
    "NextMind",
    "OpenBCI",
]


@dataclass
class BatchConfig:
    """批量分析配置。"""
    companies: list[str] = field(default_factory=lambda: DEFAULT_BCI_COMPANIES)
    max_workers: int = 3            # 并发 Agent 数
    timeout_per_task: float = 120.0 # 单任务超时
    total_timeout: float = 600.0    # 总超时
    query_template: str = "详细分析 {company} 这家 BCI 公司的技术路线、核心产品、融资阶段和竞争优势。"
    output_dir: str = "async_tasks/results"


@dataclass
class BatchResult:
    """批量分析汇总结果。"""
    total: int
    completed: int
    failed: int
    total_duration: float
    avg_duration: float
    results: list[TaskResult]

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "total_duration": round(self.total_duration, 2),
            "avg_duration": round(self.avg_duration, 2),
            "success_rate": round(self.completed / max(self.total, 1), 3),
            "results": [r.to_dict() for r in self.results],
        }


class BatchAnalyzer:
    """批量分析编排器。

    用法（同步入口）：
        analyzer = BatchAnalyzer(agent)
        batch_result = analyzer.run()

    用法（异步）：
        analyzer = BatchAnalyzer(agent)
        batch_result = await analyzer.run_async()
    """

    def __init__(self, agent, config: BatchConfig | None = None):
        self.agent = agent
        self.config = config or BatchConfig()

    def run(self, companies: list[str] | None = None) -> BatchResult:
        """同步入口，内部启动事件循环。"""
        return asyncio.run(self.run_async(companies))

    async def run_async(self, companies: list[str] | None = None) -> BatchResult:
        """异步执行批量分析。"""
        target_companies = companies or self.config.companies
        print(f"\n🚀 批量分析启动: {len(target_companies)} 家公司, "
              f"并发数: {self.config.max_workers}")

        # 创建任务队列
        queue = AsyncTaskQueue(
            agent=self.agent,
            max_workers=self.config.max_workers,
        )

        # 构造查询
        queries = [
            self.config.query_template.format(company=company)
            for company in target_companies
        ]

        # 批量提交
        start_time = time.time()
        task_ids = await queue.submit_batch(queries)

        # 打印进度
        completed_count = 0

        def on_complete(result: TaskResult):
            nonlocal completed_count
            completed_count += 1
            icon = "✅" if result.status == TaskStatus.COMPLETED else "❌"
            print(f"  {icon} [{completed_count}/{len(target_companies)}] "
                  f"{result.query[:40]}... "
                  f"({result.duration_seconds:.1f}s)")

        # 等待全部完成
        results = await queue.wait_all(
            task_ids,
            timeout=self.config.total_timeout,
            on_complete=on_complete,
        )

        total_duration = time.time() - start_time

        # 关闭队列
        await queue.shutdown()

        # 汇总
        completed = sum(1 for r in results if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == TaskStatus.FAILED)
        durations = [r.duration_seconds for r in results if r.duration_seconds > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0

        batch_result = BatchResult(
            total=len(target_companies),
            completed=completed,
            failed=failed,
            total_duration=total_duration,
            avg_duration=avg_duration,
            results=results,
        )

        # 打印汇总
        self._print_summary(batch_result, target_companies)

        # 保存结果
        self._save_results(batch_result)

        return batch_result

    def _print_summary(self, batch: BatchResult, companies: list[str]):
        """打印批量分析汇总。"""
        print(f"\n{'=' * 60}")
        print(f"  批量分析完成")
        print(f"{'=' * 60}")
        print(f"  总数: {batch.total}")
        print(f"  成功: {batch.completed}")
        print(f"  失败: {batch.failed}")
        print(f"  成功率: {batch.completed / max(batch.total, 1):.0%}")
        print(f"  总耗时: {batch.total_duration:.1f}s")
        print(f"  平均单任务: {batch.avg_duration:.1f}s")

        # 如果串行执行，预计耗时
        serial_estimate = batch.avg_duration * batch.total
        speedup = serial_estimate / max(batch.total_duration, 1)
        print(f"  串行预估: {serial_estimate:.0f}s → 并发加速比: {speedup:.1f}x")

        # 失败的任务
        failed_tasks = [r for r in batch.results if r.status == TaskStatus.FAILED]
        if failed_tasks:
            print(f"\n  ⚠️ 失败任务:")
            for r in failed_tasks:
                print(f"    - {r.query[:50]}... | 原因: {r.error}")

        print(f"{'=' * 60}")

    def _save_results(self, batch: BatchResult):
        """保存结果到 JSON 文件。"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"batch_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(batch.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"\n📄 结果已保存: {filename}")