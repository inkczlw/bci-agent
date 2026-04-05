"""异步任务队列 + 批量分析测试脚本。

用法：
    python -m tests.test_async queue       # 测试任务队列基本功能
    python -m tests.test_async batch       # 批量分析 3 家公司（快速验证）
    python -m tests.test_async batch_full  # 批量分析 10 家公司（完整测试）
    python -m tests.test_async all         # 全部测试
"""

import sys
import asyncio
import time

from agents.bci_agent import create_bci_agent
from async_tasks.task_queue import AsyncTaskQueue, TaskStatus
from async_tasks.batch_analyzer import BatchAnalyzer, BatchConfig


async def test_queue():
    """测试任务队列基本功能。"""
    print("\n=== 测试：AsyncTaskQueue 基本功能 ===\n")
    agent = create_bci_agent()
    queue = AsyncTaskQueue(agent=agent, max_workers=2)

    # 提交 2 个任务
    queries = ["介绍一下 Neuralink", "BrainCo 的主要产品是什么？"]
    task_ids = await queue.submit_batch(queries)
    print(f"提交了 {len(task_ids)} 个任务: {task_ids}")

    # 查看队列状态
    stats = queue.get_queue_stats()
    print(f"队列状态: {stats}")

    # 等待完成
    def on_done(result):
        print(f"  完成: {result.task_id} — {result.status.value} "
              f"({result.duration_seconds:.1f}s)")

    results = await queue.wait_all(task_ids, on_complete=on_done)

    # 验证
    for r in results:
        print(f"\n[{r.task_id}] status={r.status.value}")
        if r.result:
            print(f"  输出预览: {str(r.result)[:200]}...")
        if r.error:
            print(f"  错误: {r.error}")

    await queue.shutdown()
    print("\n✅ 队列测试完成")


def test_batch_quick():
    """快速批量测试（3 家公司）。"""
    print("\n=== 测试：BatchAnalyzer 快速批量（3 家）===\n")
    agent = create_bci_agent()

    config = BatchConfig(
        companies=["Neuralink", "BrainCo", "Synchron"],
        max_workers=2,
        timeout_per_task=90,
        total_timeout=300,
    )

    analyzer = BatchAnalyzer(agent=agent, config=config)
    result = analyzer.run()

    assert result.completed >= 1, "至少应有 1 个任务成功"
    print(f"\n✅ 批量快速测试完成: {result.completed}/{result.total} 成功")


def test_batch_full():
    """完整批量测试（10 家公司）。"""
    print("\n=== 测试：BatchAnalyzer 完整批量（10 家）===\n")
    agent = create_bci_agent()

    config = BatchConfig(
        max_workers=3,
        total_timeout=600,
    )

    analyzer = BatchAnalyzer(agent=agent, config=config)
    result = analyzer.run()

    print(f"\n✅ 批量完整测试完成: {result.completed}/{result.total} 成功, "
          f"加速比: {(result.avg_duration * result.total) / max(result.total_duration, 1):.1f}x")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "queue":
        asyncio.run(test_queue())
    elif command == "batch":
        test_batch_quick()
    elif command == "batch_full":
        test_batch_full()
    elif command == "all":
        asyncio.run(test_queue())
        test_batch_quick()
    else:
        print(f"❌ 未知命令: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()