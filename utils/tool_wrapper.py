"""Tool 保护层。

对 LangChain tool 添加超时控制 + 异常兜底。
不修改原始 tool 的签名和 schema。
"""

import logging
import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# 全局线程池，复用而不是每次调用都新建
_executor = ThreadPoolExecutor(max_workers=4)

# 默认配置
DEFAULT_TIMEOUT = 30
DEFAULT_FALLBACK_MSG = "Tool unavailable"


def with_fallback(timeout_seconds: int = DEFAULT_TIMEOUT,
                  fallback_prefix: str = DEFAULT_FALLBACK_MSG):
    """给一个 LangChain tool 包上超时 + 异常保护。

    用法：
        from tools.bci_search import search_bci_company
        safe_search = with_fallback(timeout_seconds=15)(search_bci_company)

    参数：
        timeout_seconds: 超时秒数。0 表示不设超时（直接调用）。
        fallback_prefix: 兜底消息前缀。

    返回：
        一个新的 StructuredTool，schema 和原 tool 完全一致。
    """
    def decorator(tool_func):
        tool_name = tool_func.name
        tool_desc = tool_func.description
        tool_schema = tool_func.args_schema

        def protected_invoke(**kwargs):
            try:
                if timeout_seconds <= 0:
                    # 不设超时，直接调用
                    return tool_func.invoke(kwargs)

                future = _executor.submit(tool_func.invoke, kwargs)
                return future.result(timeout=timeout_seconds)

            except FuturesTimeoutError:
                msg = (f"{fallback_prefix}: {tool_name} timed out "
                       f"after {timeout_seconds}s. "
                       f"Please answer based on your general knowledge.")
                logger.warning(f"[tool_timeout] {tool_name} | {timeout_seconds}s")
                return msg

            except Exception as e:
                msg = (f"{fallback_prefix}: {tool_name} encountered "
                       f"{type(e).__name__}. "
                       f"Please answer based on your general knowledge.")
                logger.error(f"[tool_error] {tool_name} | {type(e).__name__}: {e}")
                return msg

        return StructuredTool(
            name=tool_name,
            description=tool_desc,
            args_schema=tool_schema,
            func=protected_invoke,
        )

    return decorator