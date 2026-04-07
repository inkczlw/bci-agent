"""用户交互日志采集器。

从 AgentTracer 的 trace 数据中提取结构化信息，写入 InteractionLog 表。
这是"数据飞轮"的第一步：采集 → 清洗 → 存储 → 分析 → 改进。

设计原则：
- 非侵入式：通过已有的 Tracer callback 数据采集，不修改 Agent 逻辑
- 容错：日志写入失败不影响 Agent 正常运行（fail-open）
- 可扩展：新增采集字段只需改这个文件，不动其他模块
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from storage.database import get_session, init_db
from storage.models import InteractionLog

logger = logging.getLogger(__name__)


class InteractionLogger:
    """交互日志采集器。

    典型用法（在 Agent 调用后）：
        il = InteractionLogger(session_id="user_123")
        il.log_interaction(
            user_query="分析 Neuralink",
            trace=tracer.get_current_trace(),  # 从 AgentTracer 拿
            response="Neuralink 是一家...",
        )
    """

    def __init__(self, session_id: Optional[str] = None, agent_version: str = "1.0"):
        self.session_id = session_id or str(uuid.uuid4())
        self.agent_version = agent_version
        self._ensure_db()

    def _ensure_db(self):
        try:
            init_db()
        except Exception:
            pass

    def log_interaction(
        self,
        user_query: str,
        response: str = "",
        trace: Optional[dict] = None,
        quality_score: Optional[float] = None,
        query_category: str = "unknown",
    ):
        """记录一次完整的 Agent 交互。

        Args:
            user_query: 用户原始输入
            response: Agent 最终回复
            trace: AgentTracer 输出的 trace dict（包含 spans、timing 等）
            quality_score: 评估分数（如果跑了评估 pipeline）
            query_category: 查询类别（fact/analysis/comparison/rag/edge）
        """
        try:
            # 从 trace 中提取结构化数据
            tools_called, tool_errors, timing, tokens = self._extract_from_trace(trace)

            record = InteractionLog(
                session_id=self.session_id,
                user_query=user_query,
                query_category=query_category,
                tools_called=tools_called,
                tool_call_count=len(tools_called),
                tool_errors=tool_errors,
                response_text=response[:2000],  # 截断，防止超大 response 撑爆数据库
                response_quality=quality_score,
                total_latency_ms=timing.get("total_ms", 0.0),
                llm_latency_ms=timing.get("llm_ms", 0.0),
                tool_latency_ms=timing.get("tool_ms", 0.0),
                prompt_tokens=tokens.get("prompt", 0),
                completion_tokens=tokens.get("completion", 0),
                total_tokens=tokens.get("total", 0),
                model_used=timing.get("model", ""),
                agent_version=self.agent_version,
            )

            with get_session() as session:
                session.add(record)

            logger.debug(f"[InteractionLogger] 已记录交互: {user_query[:50]}...")
        except Exception as e:
            # fail-open: 日志写入失败不影响 Agent 运行
            logger.warning(f"[InteractionLogger] 写入失败（不影响 Agent）: {e}")

    def _extract_from_trace(self, trace: Optional[dict]) -> tuple:
        """从 Trace dict 中提取工具调用、错误、耗时、token 信息。

        适配你的 AgentTracer 输出格式（Trace.to_dict()）。
        如果 trace 为 None 或格式不匹配，返回安全的默认值。
        """
        if trace is None:
            return [], [], {}, {}

        tools_called = []
        tool_errors = []
        llm_ms = 0.0
        tool_ms = 0.0
        total_tokens = {"prompt": 0, "completion": 0, "total": 0}
        model = ""

        spans = trace.get("spans", [])
        for span in spans:
            span_type = span.get("span_type", "")
            name = span.get("name", "")
            duration = span.get("duration_ms", 0.0)
            status = span.get("status", "ok")

            if span_type == "tool":
                tools_called.append(name)
                tool_ms += duration
                if status != "ok":
                    tool_errors.append(
                        {
                            "tool": name,
                            "error": status,
                            "duration_ms": duration,
                        }
                    )
            elif span_type == "llm":
                llm_ms += duration
                # 提取 token 用量
                usage = span.get("token_usage", {})
                total_tokens["prompt"] += usage.get("prompt_tokens", 0)
                total_tokens["completion"] += usage.get("completion_tokens", 0)
                total_tokens["total"] += usage.get("total_tokens", 0)
                if not model and span.get("metadata", {}).get("model"):
                    model = span["metadata"]["model"]

        timing = {
            "total_ms": trace.get("total_duration_ms", llm_ms + tool_ms),
            "llm_ms": llm_ms,
            "tool_ms": tool_ms,
            "model": model,
        }

        return tools_called, tool_errors, timing, total_tokens

    def get_session_history(self) -> list[dict]:
        """获取当前 session 的所有交互记录。"""
        try:
            with get_session() as session:
                logs = (
                    session.query(InteractionLog)
                    .filter(InteractionLog.session_id == self.session_id)
                    .order_by(InteractionLog.created_at)
                    .all()
                )
                return [
                    {
                        "query": log.user_query,
                        "tools": log.tools_called,
                        "latency_ms": log.total_latency_ms,
                        "tokens": log.total_tokens,
                        "quality": log.response_quality,
                        "time": str(log.created_at),
                    }
                    for log in logs
                ]
        except Exception as e:
            logger.warning(f"[InteractionLogger] 查询失败: {e}")
            return []


class InteractionAnalyzer:
    """交互日志分析器 —— 数据清洗 + 聚合统计。

    从 InteractionLog 表中提取有价值的洞察：
    - 哪些 tool 经常失败？
    - 哪类查询延迟最高？
    - token 消耗的分布？
    - Agent 版本之间的性能对比（为周三 A/B 测试准备）
    """

    @staticmethod
    def get_tool_error_summary(hours: int = 24) -> dict:
        """统计指定时间段内各 tool 的错误率。"""
        try:
            with get_session() as session:
                cutoff = datetime.now(timezone.utc)
                logs = (
                    session.query(InteractionLog)
                    .filter(InteractionLog.created_at >= cutoff)
                    .all()
                )

                tool_stats = {}
                for log in logs:
                    for tool_name in log.tools_called or []:
                        if tool_name not in tool_stats:
                            tool_stats[tool_name] = {"total": 0, "errors": 0}
                        tool_stats[tool_name]["total"] += 1
                    for err in log.tool_errors or []:
                        err_tool = err.get("tool", "unknown")
                        if err_tool in tool_stats:
                            tool_stats[err_tool]["errors"] += 1

                # 计算错误率
                for name, stats in tool_stats.items():
                    stats["error_rate"] = (
                        round(stats["errors"] / stats["total"], 3)
                        if stats["total"] > 0
                        else 0.0
                    )

                return tool_stats
        except Exception as e:
            logger.warning(f"[InteractionAnalyzer] 分析失败: {e}")
            return {}

    @staticmethod
    def get_latency_by_category() -> dict:
        """按查询类别统计平均延迟。"""
        try:
            with get_session() as session:
                logs = session.query(InteractionLog).all()

                categories = {}
                for log in logs:
                    cat = log.query_category or "unknown"
                    if cat not in categories:
                        categories[cat] = {
                            "count": 0,
                            "total_ms": 0.0,
                            "total_tokens": 0,
                        }
                    categories[cat]["count"] += 1
                    categories[cat]["total_ms"] += log.total_latency_ms or 0.0
                    categories[cat]["total_tokens"] += log.total_tokens or 0

                for cat, stats in categories.items():
                    stats["avg_latency_ms"] = (
                        round(stats["total_ms"] / stats["count"], 1)
                        if stats["count"] > 0
                        else 0.0
                    )
                    stats["avg_tokens"] = (
                        round(stats["total_tokens"] / stats["count"])
                        if stats["count"] > 0
                        else 0
                    )

                return categories
        except Exception as e:
            logger.warning(f"[InteractionAnalyzer] 分析失败: {e}")
            return {}

    @staticmethod
    def compare_agent_versions() -> dict:
        """对比不同 Agent 版本的性能（为周三 A/B 测试准备）。"""
        try:
            with get_session() as session:
                logs = session.query(InteractionLog).all()

                versions = {}
                for log in logs:
                    ver = log.agent_version or "unknown"
                    if ver not in versions:
                        versions[ver] = {
                            "count": 0,
                            "total_latency": 0.0,
                            "total_tokens": 0,
                            "error_count": 0,
                            "quality_scores": [],
                        }
                    v = versions[ver]
                    v["count"] += 1
                    v["total_latency"] += log.total_latency_ms or 0.0
                    v["total_tokens"] += log.total_tokens or 0
                    v["error_count"] += len(log.tool_errors or [])
                    if log.response_quality is not None:
                        v["quality_scores"].append(log.response_quality)

                # 计算平均值
                for ver, v in versions.items():
                    v["avg_latency_ms"] = (
                        round(v["total_latency"] / v["count"], 1)
                        if v["count"] > 0
                        else 0.0
                    )
                    v["avg_tokens"] = (
                        round(v["total_tokens"] / v["count"]) if v["count"] > 0 else 0
                    )
                    v["avg_quality"] = (
                        round(sum(v["quality_scores"]) / len(v["quality_scores"]), 2)
                        if v["quality_scores"]
                        else None
                    )
                    v["error_rate"] = (
                        round(v["error_count"] / v["count"], 3)
                        if v["count"] > 0
                        else 0.0
                    )
                    del v["quality_scores"]  # 清掉原始列表
                    del v["total_latency"]
                    del v["total_tokens"]

                return versions
        except Exception as e:
            logger.warning(f"[InteractionAnalyzer] 分析失败: {e}")
            return {}
