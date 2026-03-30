"""BCI 公司分析 tool。

用 with_structured_output (function_calling) 约束输出，
结果同时存入 result_store，确保调用方能拿到原始结构化数据。
"""

from langchain_core.tools import tool
from config import get_llm
from rag.vectorstore import search
from schemas.bci_models import BCICompanyAnalysis
from utils.result_store import save_analysis

from utils.tool_registry import register


@register(timeout_seconds=15)
@tool
def analyze_bci_company(company_name: str) -> str:
    """分析 BCI 公司，返回结构化 JSON 报告。"""

    # 1. 检索
    results = search(company_name, n_results=5)
    context = "\n".join(results) if results else "未找到相关信息"

    # 2. 结构化生成
    llm = get_llm()
    structured_llm = llm.with_structured_output(
        BCICompanyAnalysis, method="function_calling"
    )

    analysis = structured_llm.invoke(
        f"根据以下资料，分析 {company_name} 这家 BCI 公司。"
        f"如果某项信息资料中没有，填'未知'。\n\n"
        f"资料：\n{context}"
    )

    # 3. 存入缓存（调用方可以直接拿结构化对象）
    save_analysis(company_name, analysis)

    # 4. 返回给 Agent 的仍然是 JSON 字符串
    return analysis.model_dump_json(ensure_ascii=False)