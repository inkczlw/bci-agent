from langchain_core.tools import tool
from rag.vectorstore import search


@tool
def search_bci_docs(query: str) -> str:
    """从 BCI 研究报告和论文中检索相关信息。用于回答脑机接口技术、公司、市场等问题。"""
    results = search(query, n_results=3)
    if not results:
        return "未找到相关文档内容"
    return "\n\n---\n\n".join(results)