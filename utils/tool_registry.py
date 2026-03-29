"""Tool 注册中心。

集中管理所有 tool 的注册、保护层包裹、发现。
Agent 只需调 get_all_tools()，不再手动 import 每个 tool。
"""

import logging
from utils.tool_wrapper import with_fallback

logger = logging.getLogger(__name__)

# ── tool 注册表 ──────────────────────────────────────────
# 每个条目：(tool_func, timeout_seconds)
# timeout 为 None 表示使用默认值
_TOOL_CONFIGS: list[tuple] = []


def register(timeout_seconds: int | None = None):
    """装饰器：把 tool 注册到全局列表。

    用法（在 tools/bci_search.py 里）：
        from utils.tool_registry import register

        @register(timeout_seconds=15)
        @tool
        def search_bci_company(company_name: str) -> str:
            ...

    注意 @register 必须在 @tool 上面（先执行 @tool 变成 StructuredTool，
    再由 @register 收进注册表）。
    """
    def decorator(tool_func):
        _TOOL_CONFIGS.append((tool_func, timeout_seconds))
        return tool_func  # 返回原 tool，不修改
    return decorator


def get_all_tools(enable_fallback: bool = True) -> list:
    """获取所有已注册的 tool。

    Args:
        enable_fallback: True 时给每个 tool 包保护层；
                        False 时返回原始 tool（调试用）。
    """
    if not _TOOL_CONFIGS:
        _discover_tools()

    tools = []
    for tool_func, timeout in _TOOL_CONFIGS:
        if enable_fallback:
            kwargs = {}
            if timeout is not None:
                kwargs["timeout_seconds"] = timeout
            safe_tool = with_fallback(**kwargs)(tool_func)
            tools.append(safe_tool)
        else:
            tools.append(tool_func)

    logger.info(f"[tool_registry] loaded {len(tools)} tools: "
                f"{[t.name for t in tools]}")
    return tools


def _discover_tools():
    """触发 tools/ 下所有模块的 import，让 @register 生效。

    这个函数只在 _TOOL_CONFIGS 为空时被调用一次。
    """
    import importlib

    # 显式列出所有 tool 模块 —— 简单、可控、不搞黑魔法
    tool_modules = [
        "tools.bci_search",
        "tools.bci_news",
        "tools.rag_search",
        "tools.bci_analyzer",
        "tools.bci_comparator",
    ]

    for module_name in tool_modules:
        try:
            importlib.import_module(module_name)
            logger.debug(f"[tool_registry] imported {module_name}")
        except ImportError as e:
            logger.warning(f"[tool_registry] failed to import {module_name}: {e}")