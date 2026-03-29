"""LLM 输出的防御性解析工具。

提供三层能力：
1. extract_json_string — 从 LLM 原始文本中提取 JSON（三层 fallback）
2. parse_llm_output — 提取 + Pydantic schema 校验
3. parse_with_retry — 提取 + 校验 + 自动重试（适用于没有缓存层的场景）

当前项目中，结构化分析走的是 tool 缓存路径（result_store），
所以 main.py 只用到了 parse_llm_output 作为 fallback。
parse_with_retry 保留给未来不经过缓存直接消费 Agent 输出的场景。
"""
import json
import re
from typing import TypeVar
from pydantic import BaseModel, ValidationError

# 泛型：支持任意 Pydantic model
T = TypeVar("T", bound=BaseModel)


def extract_json_string(text: str) -> dict | None:
    """从 LLM 的原始文本输出中提取 JSON 对象。

    三层 fallback 策略：
    1. 整个输出就是合法 JSON → 直接解析
    2. JSON 被 markdown 代码块包裹 → 正则提取
    3. JSON 混在自由文本中 → 找第一个 { 到最后一个 }
    """

    text = text.strip()

    # Layer 1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Layer 2: 提取 ```json ... ``` 或 ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Layer 3: 花括号兜底——找最外层的 { }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_llm_output(text: str, model_class: type[T]) -> T | None:
    """解析 LLM 输出并用 Pydantic model 校验。

    先提取 JSON，再用指定的 Pydantic model 做类型校验和默认值补全。
    这个函数是泛型的——传入不同的 model_class，就能解析不同的结构。

    用法：
        result = parse_llm_output(llm_text, BCICompanyAnalysis)
        if result:
            print(result.company_name)
    """

    raw = extract_json_string(text)
    if raw is None:
        return None

    try:
        return model_class.model_validate(raw)
    except ValidationError as e:
        print(f"[Schema 校验失败] {e}")
        return None


def parse_with_retry(invoke_fn, query: str, model_class: type[T], max_retries: int = 3) -> T | None:
    """带重试的结构化输出获取。

    如果解析失败，会把错误信息反馈给 LLM 重试。
    invoke_fn 是 agent.invoke 或任何接受 messages 的调用函数。
    """

    error_msg = ""

    for attempt in range(max_retries):
        # 首次正常提问，后续追加错误反馈
        if attempt == 0:
            prompt = query
        else:
            prompt = (
                f"{query}\n\n"
                f"[格式修正提示] 上次输出解析失败：{error_msg}\n"
                f"请只返回合法 JSON，不要包含任何其他文字。"
            )

        result = invoke_fn(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        output = result["messages"][-1].content
        parsed = parse_llm_output(output, model_class)

        if parsed is not None:
            return parsed

        error_msg = f"第 {attempt + 1} 次输出无法解析为 {model_class.__name__}"
        print(f"[重试 {attempt + 1}/{max_retries}] {error_msg}")

    return None