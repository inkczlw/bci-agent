"""Tool 结果的临时存储。

解决的问题：tool 内部用 with_structured_output 生成了合法 JSON，
但 Agent 会把 JSON "翻译"成自由文本再转述给用户。
这个 store 让调用方能直接拿到 tool 的原始结构化结果。
"""

from schemas.bci_models import BCICompanyAnalysis

# 简单的内存缓存，生产环境可以换成 Redis
_analysis_cache: dict[str, BCICompanyAnalysis] = {}


def save_analysis(company: str, analysis: BCICompanyAnalysis):
    _analysis_cache[company.lower()] = analysis


def get_analysis(company: str) -> BCICompanyAnalysis | None:
    return _analysis_cache.get(company.lower())


def get_all_analyses() -> dict[str, BCICompanyAnalysis]:
    return dict(_analysis_cache)