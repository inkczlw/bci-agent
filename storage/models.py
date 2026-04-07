"""ORM 模型定义。

两张核心表：
- AnalysisResult: Agent 分析结果持久化（周一）
- InteractionLog: 用户交互日志（周二）
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _utc_now():
    return datetime.now(timezone.utc)


def _gen_id():
    return str(uuid.uuid4())


class AnalysisResult(Base):
    """Agent 分析结果表。

    存储结构化的公司分析报告，避免重复 LLM 调用。
    设计思路：cache 层 —— 同一公司短期内不需要重新分析。
    """

    __tablename__ = "analysis_results"

    id = Column(String(36), primary_key=True, default=_gen_id)
    company_name = Column(String(100), nullable=False, index=True)
    analysis_type = Column(String(50), nullable=False, default="company_analysis")
    # 核心分析字段 —— 对应 schemas/bci_models.py 的 BCICompanyAnalysis
    technology_route = Column(Text, default="")
    funding_stage = Column(String(50), default="")
    core_technology = Column(Text, default="")
    competitive_advantage = Column(Text, default="")
    application_areas = Column(Text, default="")
    # 原始 JSON（完整 Pydantic model dump）
    raw_json = Column(JSON, nullable=True)
    # 元数据
    source_query = Column(Text, default="")  # 触发分析的原始 query
    model_used = Column(String(50), default="")  # 用的哪个 LLM
    latency_ms = Column(Float, default=0.0)  # 分析耗时
    token_cost = Column(Integer, default=0)  # token 消耗
    created_at = Column(DateTime, default=_utc_now)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now)

    # 复合索引：按公司名 + 时间查最新分析
    __table_args__ = (Index("ix_company_latest", "company_name", "created_at"),)

    def __repr__(self):
        return f"<AnalysisResult {self.company_name} @ {self.created_at}>"

    def is_stale(self, max_age_hours: int = 24) -> bool:
        """判断分析结果是否过期。"""
        if self.created_at is None:
            return True
        age = datetime.now(timezone.utc) - self.created_at.replace(tzinfo=timezone.utc)
        return age.total_seconds() > max_age_hours * 3600


class InteractionLog(Base):
    """用户交互日志表。

    记录每次 Agent 交互的完整链路，供后续分析和改进。
    设计思路：这是你的 "数据飞轮" 基础 ——
    收集交互数据 → 分析薄弱环节 → 优化 prompt/tool → 验证改进效果。
    """

    __tablename__ = "interaction_logs"

    id = Column(String(36), primary_key=True, default=_gen_id)
    session_id = Column(String(36), nullable=False, index=True)
    # 用户输入
    user_query = Column(Text, nullable=False)
    query_category = Column(
        String(50), default="unknown"
    )  # fact/analysis/comparison/rag/edge
    # Agent 执行过程
    tools_called = Column(
        JSON, default=list
    )  # ["search_bci_company", "analyze_bci_company"]
    tool_call_count = Column(Integer, default=0)
    tool_errors = Column(
        JSON, default=list
    )  # [{"tool": "rag_search", "error": "timeout"}]
    # Agent 输出
    response_text = Column(Text, default="")
    response_quality = Column(
        Float, nullable=True
    )  # 评估 pipeline 的评分（如果有的话）
    # 性能数据
    total_latency_ms = Column(Float, default=0.0)
    llm_latency_ms = Column(Float, default=0.0)
    tool_latency_ms = Column(Float, default=0.0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    # 元数据
    model_used = Column(String(50), default="")
    agent_version = Column(String(20), default="1.0")  # 周三 A/B 测试会用到
    created_at = Column(DateTime, default=_utc_now)

    # 按时间范围查询的索引
    __table_args__ = (
        Index("ix_interaction_time", "created_at"),
        Index("ix_session", "session_id", "created_at"),
    )

    def __repr__(self):
        return f"<InteractionLog {self.user_query[:30]}... @ {self.created_at}>"
