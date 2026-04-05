"""评估用例定义。

每个 test case 定义：
- 输入 query
- 期望调用的 tool（验证 Agent 的 tool 选择能力）
- 期望输出中的关键词（验证回答的事实覆盖）
- 期望输出格式（验证结构化输出完整性）
- 延迟阈值（验证性能）
"""

from dataclasses import dataclass, field
from enum import Enum


class TestCategory(str, Enum):
    """测试类别。"""
    FACTUAL = "factual"           # 事实查询
    ANALYSIS = "analysis"         # 结构化分析
    COMPARISON = "comparison"     # 多步对比
    RAG = "rag"                   # RAG 检索
    EDGE_CASE = "edge_case"       # 边界情况


@dataclass
class TestCase:
    """单个评估用例。"""
    id: str
    category: TestCategory
    query: str
    # ── 期望行为 ──
    expected_tools: list[str] = field(default_factory=list)      # 应该调用哪些 tool
    forbidden_tools: list[str] = field(default_factory=list)     # 不应该调用的 tool
    expected_keywords: list[str] = field(default_factory=list)   # 输出必须包含的关键词
    expected_fields: list[str] = field(default_factory=list)     # 结构化输出必须有的字段
    # ── 阈值 ──
    max_latency_seconds: float = 60.0
    max_tool_calls: int = 10                                     # 防止无限循环
    # ── 元数据 ──
    description: str = ""
    weight: float = 1.0          # 评分权重


# ── 预定义 Test Cases ──────────────────────────────────────────

EVAL_TEST_CASES: list[TestCase] = [
    # ── 事实查询 ──
    TestCase(
        id="fact_01",
        category=TestCategory.FACTUAL,
        query="介绍一下 Neuralink 这家公司",
        expected_tools=["search_bci_company"],
        expected_keywords=["Neuralink", "Elon Musk", "脑机接口"],
        max_latency_seconds=30,
        description="基础公司查询，验证静态数据 tool 选择",
    ),
    TestCase(
        id="fact_02",
        category=TestCategory.FACTUAL,
        query="BrainCo 的最新动态是什么？",
        expected_tools=["get_bci_news"],
        expected_keywords=["BrainCo"],
        max_latency_seconds=30,
        description="新闻查询，验证 news tool 选择",
    ),
    TestCase(
        id="fact_03",
        category=TestCategory.FACTUAL,
        query="目前 BCI 行业有哪些主要的技术路线？",
        expected_tools=["rag_search"],
        expected_keywords=["侵入式", "非侵入式"],
        max_latency_seconds=30,
        description="领域知识查询，验证 RAG tool 选择",
    ),

    # ── 结构化分析 ──
    TestCase(
        id="analysis_01",
        category=TestCategory.ANALYSIS,
        query="分析 Neuralink 的技术和商业前景",
        expected_tools=["analyze_bci_company"],
        expected_fields=["company_name", "core_technology", "competitive_advantage"],
        max_latency_seconds=45,
        description="结构化分析，验证 structured output 完整性",
    ),
    TestCase(
        id="analysis_02",
        category=TestCategory.ANALYSIS,
        query="详细分析 BrainCo 的产品线和市场定位",
        expected_tools=["analyze_bci_company"],
        expected_fields=["company_name", "core_technology"],
        max_latency_seconds=45,
        description="另一家公司的分析，验证泛化能力",
    ),

    # ── 多步对比 ──
    TestCase(
        id="compare_01",
        category=TestCategory.COMPARISON,
        query="对比 Neuralink 和 BrainCo 的技术路线和商业模式",
        expected_tools=["compare_bci_companies"],
        expected_keywords=["Neuralink", "BrainCo"],
        expected_fields=["company_a", "company_b"],
        max_latency_seconds=90,
        max_tool_calls=8,
        weight=1.5,
        description="多步对比，验证 comparator 链路（最复杂场景）",
    ),

    # ── RAG 检索 ──
    TestCase(
        id="rag_01",
        category=TestCategory.RAG,
        query="BCI 技术在医疗康复领域有哪些应用？",
        expected_tools=["rag_search"],
        expected_keywords=["康复", "医疗"],
        max_latency_seconds=30,
        description="RAG 检索，验证文档知识覆盖",
    ),
    TestCase(
        id="rag_02",
        category=TestCategory.RAG,
        query="脑机接口的信号采集方式有哪些？各有什么优缺点？",
        expected_tools=["rag_search"],
        expected_keywords=["信号"],
        max_latency_seconds=30,
        description="技术细节查询，验证 chunk 检索质量",
    ),

    # ── 边界情况 ──
    TestCase(
        id="edge_01",
        category=TestCategory.EDGE_CASE,
        query="特斯拉的股价是多少？",
        forbidden_tools=["analyze_bci_company", "compare_bci_companies"],
        max_latency_seconds=20,
        description="超出 BCI 领域的问题，Agent 不应强行调用分析 tool",
    ),
    TestCase(
        id="edge_02",
        category=TestCategory.EDGE_CASE,
        query="",
        max_latency_seconds=10,
        description="空输入，Agent 应优雅处理",
    ),
    TestCase(
        id="edge_03",
        category=TestCategory.EDGE_CASE,
        query="对比 Neuralink 和一家不存在的公司 XyzBrainTech",
        expected_tools=["search_bci_company"],
        max_latency_seconds=45,
        description="不存在的公司，验证 Agent 的错误处理能力",
    ),
]


def get_test_cases(
    category: TestCategory | None = None,
    ids: list[str] | None = None,
) -> list[TestCase]:
    """获取 test case，支持按类别或 ID 过滤。"""
    cases = EVAL_TEST_CASES

    if category:
        cases = [c for c in cases if c.category == category]

    if ids:
        cases = [c for c in cases if c.id in ids]

    return cases