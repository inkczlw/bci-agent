"""异步任务队列。

基于 asyncio + ThreadPoolExecutor 的轻量任务队列。
支持任务提交、状态查询、结果获取、并发控制。

- 单机场景：asyncio + ThreadPoolExecutor（当前实现）
- 分布式场景：Celery + Redis / RQ + Redis
- 大规模场景：Kafka producer → 独立 worker 集群
- C++ 类比：producer-consumer 模式，任务队列 = bounded queue，
  worker = consumer thread pool，Future = std::future

为什么 Agent 用 ThreadPoolExecutor 而不是纯 asyncio？
因为 LangChain/LangGraph 的 invoke 是同步阻塞调用，
不能直接 await。需要用 run_in_executor 把同步调用包进事件循环。
"""

import asyncio
import time
import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("bci_agent.task_queue")


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """任务执行结果。"""
    task_id: str
    status: TaskStatus
    query: str
    result: Any = None
    error: str | None = None
    submit_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "query": self.query[:100],
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class Task:
    """队列中的任务。"""
    task_id: str
    query: str
    status: TaskStatus = TaskStatus.PENDING
    result: TaskResult | None = None
    metadata: dict = field(default_factory=dict)


class AsyncTaskQueue:
    """异步任务队列。

    管理 Agent 任务的提交、调度、执行、结果收集。

    用法：
        queue = AsyncTaskQueue(agent=agent, max_workers=3)
        task_ids = await queue.submit_batch(["分析 Neuralink", "分析 BrainCo", ...])
        results = await queue.wait_all(task_ids)

    Args:
        agent: LangGraph Agent（可 invoke 的对象）
        max_workers: 最大并发 worker 数
        queue_size: 队列最大容量（0=无限制）
    """

    def __init__(
        self,
        agent,
        max_workers: int = 3,
        queue_size: int = 100,
    ):
        self.agent = agent
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="agent_worker",
        )
        self._tasks: dict[str, Task] = {}
        self._semaphore = asyncio.Semaphore(max_workers)
        self._queue_size = queue_size
        logger.info(f"TaskQueue 初始化: max_workers={max_workers}, queue_size={queue_size}")

    async def submit(self, query: str, metadata: dict | None = None) -> str:
        """提交单个任务，返回 task_id。"""
        if self._queue_size and len(self._tasks) >= self._queue_size:
            raise RuntimeError(f"队列已满 ({self._queue_size})")

        task_id = str(uuid.uuid4())[:8]
        task = Task(
            task_id=task_id,
            query=query,
            metadata=metadata or {},
        )
        self._tasks[task_id] = task

        # 异步启动执行（不阻塞）
        asyncio.create_task(self._execute(task))
        logger.info(f"任务已提交: {task_id} — {query[:50]}")
        return task_id

    async def submit_batch(self, queries: list[str]) -> list[str]:
        """批量提交任务，返回 task_id 列表。"""
        task_ids = []
        for i, query in enumerate(queries):
            task_id = await self.submit(query, metadata={"batch_index": i})
            task_ids.append(task_id)
        logger.info(f"批量提交 {len(queries)} 个任务")
        return task_ids

    async def get_status(self, task_id: str) -> TaskStatus:
        """查询任务状态。"""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"任务不存在: {task_id}")
        return task.status

    async def get_result(self, task_id: str) -> TaskResult | None:
        """获取任务结果（不等待）。"""
        task = self._tasks.get(task_id)
        if not task:
            raise KeyError(f"任务不存在: {task_id}")
        return task.result

    async def wait_one(self, task_id: str, timeout: float = 120.0) -> TaskResult:
        """等待单个任务完成。"""
        start = time.time()
        while time.time() - start < timeout:
            task = self._tasks.get(task_id)
            if not task:
                raise KeyError(f"任务不存在: {task_id}")

            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return task.result

            await asyncio.sleep(0.5)

        # 超时
        task = self._tasks[task_id]
        task.status = TaskStatus.FAILED
        task.result = TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            query=task.query,
            error=f"等待超时 ({timeout}s)",
        )
        return task.result

    async def wait_all(
        self,
        task_ids: list[str],
        timeout: float = 300.0,
        on_complete: Callable[[TaskResult], None] | None = None,
    ) -> list[TaskResult]:
        """等待所有任务完成，支持进度回调。"""
        results = []
        completed = set()
        start = time.time()

        while len(completed) < len(task_ids) and time.time() - start < timeout:
            for tid in task_ids:
                if tid in completed:
                    continue

                task = self._tasks.get(tid)
                if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    completed.add(tid)
                    if on_complete and task.result:
                        on_complete(task.result)

            if len(completed) < len(task_ids):
                await asyncio.sleep(0.5)

        # 收集结果（按提交顺序）
        for tid in task_ids:
            task = self._tasks.get(tid)
            if task and task.result:
                results.append(task.result)
            else:
                results.append(TaskResult(
                    task_id=tid,
                    status=TaskStatus.FAILED,
                    query=self._tasks[tid].query if tid in self._tasks else "unknown",
                    error="未完成",
                ))

        return results

    async def _execute(self, task: Task):
        """在 worker 线程中执行 Agent 调用。"""
        async with self._semaphore:  # 并发控制
            task.status = TaskStatus.RUNNING
            submit_time = time.time()

            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.RUNNING,
                query=task.query,
                submit_time=submit_time,
                start_time=time.time(),
            )

            try:
                loop = asyncio.get_event_loop()
                # 同步 Agent.invoke 放到线程池执行
                agent_result = await loop.run_in_executor(
                    self._executor,
                    self._invoke_agent,
                    task.query,
                )

                result.end_time = time.time()
                result.duration_seconds = result.end_time - result.start_time
                result.status = TaskStatus.COMPLETED
                result.result = agent_result

                task.status = TaskStatus.COMPLETED
                logger.info(
                    f"任务完成: {task.task_id} | "
                    f"{result.duration_seconds:.1f}s | "
                    f"{task.query[:30]}"
                )

            except Exception as e:
                result.end_time = time.time()
                result.duration_seconds = result.end_time - result.start_time
                result.status = TaskStatus.FAILED
                result.error = f"{type(e).__name__}: {str(e)}"

                task.status = TaskStatus.FAILED
                logger.error(f"任务失败: {task.task_id} — {result.error}")

            task.result = result

    def _invoke_agent(self, query: str) -> str:
        """同步调用 Agent（在 worker 线程中执行）。"""
        agent_result = self.agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        messages = agent_result.get("messages", [])
        if messages:
            last = messages[-1]
            return last.content if hasattr(last, "content") else str(last)
        return "(无输出)"

    def get_queue_stats(self) -> dict:
        """获取队列统计信息。"""
        status_counts = {}
        for task in self._tasks.values():
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1

        return {
            "total": len(self._tasks),
            "max_workers": self.max_workers,
            "queue_size": self._queue_size,
            "by_status": status_counts,
        }

    async def shutdown(self):
        """关闭队列，等待正在执行的任务完成。"""
        logger.info("TaskQueue 关闭中...")
        self._executor.shutdown(wait=True)
        logger.info("TaskQueue 已关闭")