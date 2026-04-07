"""数据库写入 tool —— 分析结果自动入库。

Agent 可以调用这个 tool 把结构化分析结果持久化到数据库。
同时提供查询功能：先查库里有没有，有且没过期就直接返回，避免重复 LLM 调用。

设计模式：Read-Through Cache
1. Agent 要分析一个公司 → 先查 DB
2. DB 有且未过期 → 直接返回缓存结果
3. DB 没有或已过期 → Agent 走正常分析流程 → 结果写入 DB
"""

import json
import time
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy import desc

from storage.database import get_session, init_db
from storage.models import AnalysisResult


def _ensure_db():
    """确保数据库表已创建（幂等）。"""
    try:
        init_db()
    except Exception:
        pass  # 表已存在时忽略


_ensure_db()


@tool
def save_analysis_result(
    company_name: str,
    technology_route: str,
    funding_stage: str,
    core_technology: str,
    competitive_advantage: str,
    application_areas: str,
    source_query: str = "",
) -> str:
    """将 BCI 公司分析结果保存到数据库。

    当你完成了一个公司的结构化分析后，调用此工具把结果持久化。
    下次查询同一公司时可以直接从数据库获取，不需要重新分析。

    参数:
        company_name: 公司名称（如 "Neuralink"）
        technology_route: 技术路线描述
        funding_stage: 融资阶段
        core_technology: 核心技术
        competitive_advantage: 竞争优势
        application_areas: 应用领域
        source_query: 触发分析的原始用户查询（可选）

    返回:
        保存成功/失败的状态信息
    """
    try:
        with get_session() as session:
            record = AnalysisResult(
                company_name=company_name.strip(),
                technology_route=technology_route,
                funding_stage=funding_stage,
                core_technology=core_technology,
                competitive_advantage=competitive_advantage,
                application_areas=application_areas,
                source_query=source_query,
                raw_json={
                    "company_name": company_name,
                    "technology_route": technology_route,
                    "funding_stage": funding_stage,
                    "core_technology": core_technology,
                    "competitive_advantage": competitive_advantage,
                    "application_areas": application_areas,
                },
            )
            session.add(record)
            session.flush()  # 触发 id 生成
            record_id = record.id  # session 关闭前取出
        return json.dumps(
            {
                "status": "success",
                "message": f"已保存 {company_name} 的分析结果到数据库",
                "record_id": record_id,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"保存失败: {str(e)}",
            },
            ensure_ascii=False,
        )


@tool
def query_analysis_result(
    company_name: str,
    max_age_hours: int = 24,
) -> str:
    """从数据库查询已有的 BCI 公司分析结果。

    如果数据库中已有该公司的未过期分析，直接返回，无需重新分析。
    如果没有或已过期，返回提示信息，Agent 应该调用其他工具重新分析。

    参数:
        company_name: 公司名称（如 "Neuralink"）
        max_age_hours: 结果最大有效期（小时），默认 24 小时

    返回:
        已有的分析结果 JSON，或 "未找到" 提示
    """
    try:
        with get_session() as session:
            result = (
                session.query(AnalysisResult)
                .filter(AnalysisResult.company_name.ilike(f"%{company_name.strip()}%"))
                .order_by(desc(AnalysisResult.created_at))
                .first()
            )

            if result is None:
                return json.dumps(
                    {
                        "status": "not_found",
                        "message": f"数据库中没有 {company_name} 的分析记录，请使用其他工具进行分析",
                    },
                    ensure_ascii=False,
                )

            if result.is_stale(max_age_hours):
                return json.dumps(
                    {
                        "status": "stale",
                        "message": f"{company_name} 的分析结果已过期（超过 {max_age_hours} 小时），建议重新分析",
                        "last_analysis_time": str(result.created_at),
                    },
                    ensure_ascii=False,
                )

            return json.dumps(
                {
                    "status": "found",
                    "message": f"找到 {company_name} 的有效分析结果",
                    "data": result.raw_json
                    or {
                        "company_name": result.company_name,
                        "technology_route": result.technology_route,
                        "funding_stage": result.funding_stage,
                        "core_technology": result.core_technology,
                        "competitive_advantage": result.competitive_advantage,
                        "application_areas": result.application_areas,
                    },
                    "analysis_time": str(result.created_at),
                },
                ensure_ascii=False,
            )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"查询失败: {str(e)}",
            },
            ensure_ascii=False,
        )


@tool
def list_analyzed_companies() -> str:
    """列出数据库中所有已分析过的 BCI 公司。

    返回公司名称列表和最后分析时间，帮助用户了解已有数据。

    返回:
        已分析公司的列表
    """
    try:
        with get_session() as session:
            results = (
                session.query(
                    AnalysisResult.company_name,
                    AnalysisResult.created_at,
                    AnalysisResult.analysis_type,
                )
                .order_by(desc(AnalysisResult.created_at))
                .all()
            )

            if not results:
                return json.dumps(
                    {
                        "status": "empty",
                        "message": "数据库中还没有任何分析记录",
                    },
                    ensure_ascii=False,
                )

            # 去重，保留每家公司最新的记录
            seen = {}
            for name, created_at, analysis_type in results:
                if name not in seen:
                    seen[name] = {
                        "company_name": name,
                        "last_analyzed": str(created_at),
                        "analysis_type": analysis_type,
                    }

            return json.dumps(
                {
                    "status": "success",
                    "total": len(seen),
                    "companies": list(seen.values()),
                },
                ensure_ascii=False,
            )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"查询失败: {str(e)}",
            },
            ensure_ascii=False,
        )
