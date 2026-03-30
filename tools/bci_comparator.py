"""BCI 公司对比分析 tool。

演示多步规划的核心思路：
一个复杂任务被拆解成多个子步骤，每步的输出是下一步的输入。
"""

from langchain_core.tools import tool
from config import get_llm
from rag.vectorstore import search
from schemas.bci_models import BCICompanyComparison
from utils.result_store import save_analysis
from utils.tool_registry import register


@register(timeout_seconds=15)
@tool
def compare_bci_companies(company_a: str, company_b: str) -> str:
    """对比两家 BCI 公司的技术路线、竞争优势和市场定位。需要提供两个公司名称。"""

    # Step 1: 分别检索两家公司的信息
    results_a = search(company_a, n_results=5)
    results_b = search(company_b, n_results=5)

    context_a = "\n".join(results_a) if results_a else f"未找到 {company_a} 相关信息"
    context_b = "\n".join(results_b) if results_b else f"未找到 {company_b} 相关信息"

    # Step 2: 合并上下文，让 LLM 做对比分析
    llm = get_llm()
    structured_llm = llm.with_structured_output(
        BCICompanyComparison, method="function_calling"
    )

    comparison = structured_llm.invoke(
        f"根据以下资料，对比分析 {company_a} 和 {company_b} 这两家 BCI 公司。\n"
        f"从技术路线、竞争优势、应用领域、市场定位等维度进行对比。\n"
        f"如果某项信息资料中没有，填'未知'。\n\n"
        f"===== {company_a} 的资料 =====\n{context_a}\n\n"
        f"===== {company_b} 的资料 =====\n{context_b}"
    )

    return comparison.model_dump_json(ensure_ascii=False)