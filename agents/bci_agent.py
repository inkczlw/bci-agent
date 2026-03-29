"""BCI 行业分析 Agent。

使用 tool_registry 自动发现和注册所有 tool，
不再手动 import 每个 tool 模块。
"""

from langgraph.prebuilt import create_react_agent
from config import get_llm
from utils.tool_registry import get_all_tools

SYSTEM_PROMPT = """你是一个专业的 BCI（脑机接口）行业分析师。

当用户问到 BCI 公司或技术相关问题时，请使用可用的工具来获取信息。
如果某个工具不可用（超时或异常），请基于你的通用知识来回答，
并说明部分信息可能不够准确。

对于公司分析类请求：
1. 先调用搜索工具获取信息
2. 基于获取的信息，严格按以下 JSON 格式返回分析结果：

```json
{
  "company_name": "公司名称",
  "tech_route": "侵入式 / 半侵入式 / 非侵入式 / 介入式",
  "funding_stage": "种子轮 / A轮 / B轮 / C轮+ / 已上市 / 未知",
  "core_technology": ["核心技术1", "核心技术2"],
  "competitive_advantage": ["竞争优势1", "竞争优势2"],
  "application_areas": ["应用领域1", "应用领域2"],
  "key_milestones": ["里程碑1", "里程碑2"],
  "valuation_or_funding": "估值或融资金额（如有）"
}
```

只返回 JSON，不要加其他解释文字。
如果信息不足，对应字段填 "未知"。

对于其他类型的问题（非公司分析），正常回答即可。"""


def create_bci_agent(enable_fallback: bool = True):
    """创建 BCI Agent。

    Args:
        enable_fallback: 是否启用 tool 保护层。
                        开发调试时可设为 False 看原始错误。
    """
    llm = get_llm()
    tools = get_all_tools(enable_fallback=enable_fallback)
    return create_react_agent(llm, tools=tools, prompt=SYSTEM_PROMPT)