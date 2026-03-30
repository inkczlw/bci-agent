from langchain_core.tools import tool
from utils.tool_registry import register


@register(timeout_seconds=15)
@tool
def search_bci_company(company_name: str) -> str:
    """搜索 BCI 公司的基本信息"""
    data = {
        "neuralink": "Neuralink：侵入式脑机接口，2016年成立，估值50亿美元，核心技术为柔性电极+手术机器人",
        "brainco": "BrainCo 强脑科技：非侵入式，2015年成立，专注教育和康复领域，核心产品为Focus脑电头环",
    }
    return data.get(company_name.lower(), f"未找到 {company_name} 的信息")