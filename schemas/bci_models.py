"""BCI 行业分析相关的数据模型定义。

所有 Agent 输出的结构化 schema 都在这里定义，
确保数据格式在 tools、agents、main 之间保持一致。
"""

from pydantic import BaseModel, Field


class BCICompanyAnalysis(BaseModel):
    """BCI 公司的结构化分析报告"""

    company_name: str = Field(description="公司名称")
    tech_route: str = Field(
        description="技术路线：侵入式 / 半侵入式 / 非侵入式 / 介入式"
    )
    funding_stage: str = Field(
        default="未知",
        description="融资阶段：种子轮 / A轮 / B轮 / C轮+ / 已上市",
    )
    core_technology: list[str] = Field(
        default_factory=list,
        description="核心技术列表",
    )
    competitive_advantage: list[str] = Field(
        default_factory=list,
        description="竞争优势",
    )
    application_areas: list[str] = Field(
        default_factory=list,
        description="应用领域",
    )
    key_milestones: list[str] = Field(
        default_factory=list,
        description="关键里程碑事件",
    )
    valuation_or_funding: str = Field(
        default="未知",
        description="估值或融资金额",
    )


class BCICompanyComparison(BaseModel):
    """两家 BCI 公司的对比分析报告"""

    company_a: str = Field(description="公司A名称")
    company_b: str = Field(description="公司B名称")
    tech_route_comparison: str = Field(description="技术路线对比")
    advantage_a: list[str] = Field(default_factory=list, description="公司A的优势")
    advantage_b: list[str] = Field(default_factory=list, description="公司B的优势")
    application_overlap: list[str] = Field(default_factory=list, description="重叠应用领域")
    application_diff: str = Field(default="未知", description="差异化应用方向")
    market_position: str = Field(default="未知", description="市场定位对比")
    conclusion: str = Field(default="未知", description="综合对比结论")