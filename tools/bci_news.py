from langchain_core.tools import tool
from utils.tool_registry import register


@register(timeout_seconds=15)
@tool
def get_bci_news(topic: str) -> str:
    """获取 BCI 脑机接口领域的最新新闻"""
    news = {
        "neuralink": "2024年1月：Neuralink 完成首例人体芯片植入，患者已能用意念控制电脑光标",
        "政策": "2024年：中国将脑机接口列入未来产业重点发展方向，多地出台支持政策",
        "融资": "2024年：BCI 领域全球融资总额超过20亿美元，非侵入式方向增长最快",
    }
    return news.get(topic.lower(), f"暂无 {topic} 相关新闻")