"""Tool 保护层的独立测试。

验证超时和异常两种 fallback 路径。
运行：python -m pytest tests/test_tool_fallback.py -v
或者：python tests/test_tool_fallback.py
"""

from langchain_core.tools import tool as lc_tool
from utils.tool_wrapper import with_fallback


# ── 测试用 tool ──────────────────────────────────────────

@lc_tool
def normal_tool(company_name: str) -> str:
    """正常返回的 tool。"""
    return f"Info about {company_name}"


@lc_tool
def slow_tool(company_name: str) -> str:
    """模拟慢 tool，超时触发。"""
    import time
    time.sleep(10)
    return "should not reach here"


@lc_tool
def broken_tool(company_name: str) -> str:
    """模拟异常 tool。"""
    raise RuntimeError("模拟内部错误")


# ── 测试 ─────────────────────────────────────────────────

def test_normal_call():
    """正常调用应该透传结果。"""
    safe = with_fallback(timeout_seconds=5)(normal_tool)
    result = safe.invoke({"company_name": "Neuralink"})
    assert "Info about Neuralink" in result
    print(f"✅ 正常调用: {result}")


def test_timeout_fallback():
    """超时应该返回 fallback 消息。"""
    safe = with_fallback(timeout_seconds=0.01)(slow_tool)
    result = safe.invoke({"company_name": "Neuralink"})
    assert "Tool unavailable" in result
    assert "timed out" in result
    print(f"✅ 超时 fallback: {result}")


def test_exception_fallback():
    """异常应该返回 fallback 消息。"""
    safe = with_fallback(timeout_seconds=5)(broken_tool)
    result = safe.invoke({"company_name": "Neuralink"})
    assert "Tool unavailable" in result
    assert "RuntimeError" in result
    print(f"✅ 异常 fallback: {result}")


def test_schema_preserved():
    """保护后的 tool 应保留原始 schema。"""
    safe = with_fallback()(normal_tool)
    assert safe.name == "normal_tool"
    assert safe.description == "正常返回的 tool。"
    assert safe.args_schema == normal_tool.args_schema
    print(f"✅ Schema 保留: name={safe.name}, schema={safe.args_schema}")


def test_custom_fallback_prefix():
    """自定义 fallback 前缀。"""
    safe = with_fallback(timeout_seconds=5, fallback_prefix="Service down")(broken_tool)
    result = safe.invoke({"company_name": "Neuralink"})
    assert "Service down" in result
    print(f"✅ 自定义前缀: {result}")


if __name__ == "__main__":
    test_normal_call()
    test_timeout_fallback()
    test_exception_fallback()
    test_schema_preserved()
    test_custom_fallback_prefix()
    print("\n🎉 所有测试通过")