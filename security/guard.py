"""安全加固模块。

三层防御：
1. 输入校验（InputValidator）— 在 Agent 入口处拦截恶意输入
2. Prompt Injection 检测（InjectionDetector）— 识别试图操控 Agent 行为的输入
3. Tool 调用沙箱（ToolSandbox）— 限制 tool 的执行范围和资源消耗

"安全防御是分层的，跟网络安全的纵深防御同一个思路。
输入层做格式校验和长度限制（类似 WAF），
语义层做 injection 检测（类似 SQL injection 防御），
执行层做 tool 权限控制和资源隔离（类似沙箱/容器）。"

C++ 类比：输入校验 = 参数断言（assert / contract），
injection 检测 = taint analysis，
tool 沙箱 = 进程隔离 + resource limit (ulimit)。
"""

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("bci_agent.security")


# ═══════════════════════════════════════════════════════════════
# 第一层：输入校验
# ═══════════════════════════════════════════════════════════════

class RiskLevel(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"


@dataclass
class ValidationResult:
    """输入校验结果。"""
    is_valid: bool
    risk_level: RiskLevel
    sanitized_input: str           # 清洗后的输入
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "risk_level": self.risk_level.value,
            "issues": self.issues,
        }


class InputValidator:
    """输入校验器。

    职责：格式校验、长度限制、基本 sanitization。
    这是第一道防线——在输入进入 Agent 之前拦截明显异常。
    """

    # 配置
    MAX_QUERY_LENGTH = 2000          # 最大输入长度（字符）
    MIN_QUERY_LENGTH = 1             # 最小输入长度
    MAX_REPEATED_CHARS = 50          # 最大连续重复字符数（防 DoS）

    def validate(self, query: str) -> ValidationResult:
        """校验并清洗用户输入。"""
        issues = []
        risk = RiskLevel.SAFE

        # 空输入
        if not query or not query.strip():
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.BLOCKED,
                sanitized_input="",
                issues=["空输入"],
            )

        sanitized = query.strip()

        # 长度检查
        if len(sanitized) > self.MAX_QUERY_LENGTH:
            issues.append(f"输入过长: {len(sanitized)} > {self.MAX_QUERY_LENGTH}")
            sanitized = sanitized[:self.MAX_QUERY_LENGTH]
            risk = RiskLevel.SUSPICIOUS

        # 连续重复字符检测（防 DoS / buffer overflow 类攻击）
        repeat_pattern = re.search(r'(.)\1{' + str(self.MAX_REPEATED_CHARS) + r',}', sanitized)
        if repeat_pattern:
            issues.append(f"检测到连续重复字符: '{repeat_pattern.group()[:20]}...'")
            risk = RiskLevel.SUSPICIOUS

        # 移除控制字符（保留换行和空格）
        original_len = len(sanitized)
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', sanitized)
        if len(sanitized) < original_len:
            issues.append(f"移除了 {original_len - len(sanitized)} 个控制字符")

        # Unicode 异常检测（零宽字符等）
        zero_width = re.findall(r'[\u200b-\u200f\u2028-\u202f\u2060\ufeff]', sanitized)
        if zero_width:
            issues.append(f"检测到 {len(zero_width)} 个零宽/不可见字符")
            sanitized = re.sub(r'[\u200b-\u200f\u2028-\u202f\u2060\ufeff]', '', sanitized)
            risk = RiskLevel.SUSPICIOUS

        return ValidationResult(
            is_valid=True,
            risk_level=risk,
            sanitized_input=sanitized,
            issues=issues,
        )


# ═══════════════════════════════════════════════════════════════
# 第二层：Prompt Injection 检测
# ═══════════════════════════════════════════════════════════════

@dataclass
class InjectionCheckResult:
    """Injection 检测结果。"""
    is_injection: bool
    risk_level: RiskLevel
    matched_patterns: list[str] = field(default_factory=list)
    confidence: float = 0.0        # 0.0 ~ 1.0

    def to_dict(self) -> dict:
        return {
            "is_injection": self.is_injection,
            "risk_level": self.risk_level.value,
            "matched_patterns": self.matched_patterns,
            "confidence": round(self.confidence, 2),
        }


class InjectionDetector:
    """Prompt Injection 检测器。

    检测方式：
    1. 规则匹配（pattern-based）— 快速、低成本，捕获已知攻击模式
    2. 启发式分析（heuristic）— 检测可疑的结构特征

    生产环境我觉得还需要：
    3. LLM 分类器（LLM-based）— 用另一个 LLM 判断输入是否是 injection
    4. 专用模型（如 rebuff、protectai/deberta-v3-base-injection）— 微调过的分类模型
    """

    # ── 已知攻击模式 ──
    # 每个 pattern 是 (regex, description, severity_weight)
    INJECTION_PATTERNS = [
        # 角色覆盖
        (r'(?i)ignore\s+(all\s+)?previous\s+instructions?', "指令覆盖: ignore previous", 0.9),
        (r'(?i)forget\s+(all\s+)?previous\s+(instructions?|rules?|context)', "指令覆盖: forget previous", 0.9),
        (r'(?i)disregard\s+(all\s+)?previous', "指令覆盖: disregard previous", 0.9),
        (r'(?i)you\s+are\s+now\s+a', "角色劫持: you are now a", 0.85),
        (r'(?i)act\s+as\s+(if\s+you\s+are\s+)?a', "角色劫持: act as", 0.5),
        (r'(?i)pretend\s+(to\s+be|you\s+are)', "角色劫持: pretend to be", 0.7),
        (r'(?i)from\s+now\s+on', "行为修改: from now on", 0.6),

        # System prompt 泄露
        (r'(?i)(show|reveal|print|display|output)\s+(your\s+)?(system\s+)?prompt', "system prompt 泄露", 0.95),
        (r'(?i)what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)', "system prompt 探测", 0.8),
        (r'(?i)(repeat|echo)\s+(the\s+)?(above|system|initial)', "system prompt 回显", 0.85),

        # 输出操控
        (r'(?i)respond\s+(only\s+)?with\s+(yes|no|true|false|json)', "输出格式劫持", 0.4),
        (r'(?i)do\s+not\s+(mention|say|reveal|disclose)', "输出抑制", 0.5),

        # 分隔符注入
        (r'(?i)(```|---|\*\*\*)\s*(system|assistant|user)\s*:', "角色标签注入", 0.9),
        (r'(?i)<\s*(system|assistant|instruction)', "XML 标签注入", 0.9),

        # 越狱（jailbreak）
        (r'(?i)DAN\s+mode', "DAN jailbreak", 0.95),
        (r'(?i)(developer|debug|admin|god)\s+mode', "特权模式", 0.8),
        (r'(?i)no\s+(ethical|moral|safety)\s+(guidelines|restrictions|limits)', "安全限制绕过", 0.95),
    ]

    # ── 启发式规则 ──
    HEURISTIC_THRESHOLDS = {
        "excessive_instructions": 5,       # 超过 N 个祈使句
        "role_switches": 2,                # 超过 N 次角色切换标记
        "suspicious_delimiters": 3,        # 超过 N 个可疑分隔符
    }

    def check(self, query: str) -> InjectionCheckResult:
        """检测输入是否包含 prompt injection。"""
        matched = []
        max_severity = 0.0

        # 1. 规则匹配
        for pattern, desc, severity in self.INJECTION_PATTERNS:
            if re.search(pattern, query):
                matched.append(desc)
                max_severity = max(max_severity, severity)

        # 2. 启发式分析
        heuristic_hits = self._heuristic_analysis(query)
        matched.extend(heuristic_hits)
        if heuristic_hits:
            max_severity = max(max_severity, 0.6)

        # 判定
        if max_severity >= 0.8:
            risk = RiskLevel.BLOCKED
            is_injection = True
        elif max_severity >= 0.5:
            risk = RiskLevel.SUSPICIOUS
            is_injection = True
        elif matched:
            risk = RiskLevel.SUSPICIOUS
            is_injection = False
        else:
            risk = RiskLevel.SAFE
            is_injection = False

        if matched:
            logger.warning(f"Injection 检测: risk={risk.value}, patterns={matched}")

        return InjectionCheckResult(
            is_injection=is_injection,
            risk_level=risk,
            matched_patterns=matched,
            confidence=max_severity,
        )

    def _heuristic_analysis(self, query: str) -> list[str]:
        """启发式分析可疑特征。"""
        hits = []

        # 过多祈使句（可能是在给 Agent 下一长串指令）
        imperative_count = len(re.findall(
            r'(?i)^(do|don\'t|never|always|must|should|make sure|ensure|remember)\b',
            query,
            re.MULTILINE,
        ))
        if imperative_count >= self.HEURISTIC_THRESHOLDS["excessive_instructions"]:
            hits.append(f"启发式: 过多祈使句 ({imperative_count})")

        # 角色切换标记
        role_markers = len(re.findall(
            r'(?i)(system:|assistant:|user:|human:|ai:|\[INST\]|\[/INST\])',
            query,
        ))
        if role_markers >= self.HEURISTIC_THRESHOLDS["role_switches"]:
            hits.append(f"启发式: 角色切换标记 ({role_markers})")

        # 可疑分隔符密度
        delimiters = len(re.findall(r'(={3,}|-{3,}|#{3,}|\*{3,}|`{3,})', query))
        if delimiters >= self.HEURISTIC_THRESHOLDS["suspicious_delimiters"]:
            hits.append(f"启发式: 可疑分隔符 ({delimiters})")

        return hits


# ═══════════════════════════════════════════════════════════════
# 第三层：Tool 调用沙箱
# ═══════════════════════════════════════════════════════════════

@dataclass
class ToolPermission:
    """Tool 权限配置。"""
    allowed: bool = True
    max_calls_per_session: int = 20         # 单次会话最大调用次数
    max_input_length: int = 1000            # 输入参数最大长度
    rate_limit_per_minute: int = 10         # 每分钟最大调用次数
    requires_confirmation: bool = False     # 是否需要用户确认才能执行


class ToolSandbox:
    """Tool 调用沙箱。

    职责：
    - 权限控制：某些 tool 可以限制调用频率、需要人工确认
    - 参数校验：tool 输入参数的长度和格式检查
    - 资源限制：防止单个 tool 调用消耗过多资源
    - 调用审计：记录所有 tool 调用用于事后分析

    C++ 类比：
    - 权限控制 = capability-based security
    - 参数校验 = contract / precondition check
    - 资源限制 = ulimit / cgroup
    - 调用审计 = audit log
    """

    # 默认权限配置
    DEFAULT_PERMISSIONS: dict[str, ToolPermission] = {
        "search_bci_company": ToolPermission(max_calls_per_session=30),
        "get_bci_news": ToolPermission(max_calls_per_session=20),
        "rag_search": ToolPermission(max_calls_per_session=20),
        "analyze_bci_company": ToolPermission(max_calls_per_session=10, rate_limit_per_minute=5),
        "compare_bci_companies": ToolPermission(max_calls_per_session=5, rate_limit_per_minute=3),
    }

    def __init__(self, permissions: dict[str, ToolPermission] | None = None):
        self.permissions = permissions or self.DEFAULT_PERMISSIONS
        self._call_counts: dict[str, int] = {}             # 本 session 的调用计数
        self._call_timestamps: dict[str, list[float]] = {} # 用于 rate limiting
        self._audit_log: list[dict] = []

    def check_permission(self, tool_name: str, tool_input: dict) -> ValidationResult:
        """检查 tool 调用是否被允许。"""
        issues = []
        perm = self.permissions.get(tool_name, ToolPermission())

        # 1. 是否允许
        if not perm.allowed:
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.BLOCKED,
                sanitized_input=str(tool_input),
                issues=[f"Tool '{tool_name}' 已被禁用"],
            )

        # 2. 调用次数限制
        count = self._call_counts.get(tool_name, 0)
        if count >= perm.max_calls_per_session:
            issues.append(f"超出会话调用上限: {count}/{perm.max_calls_per_session}")
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.BLOCKED,
                sanitized_input=str(tool_input),
                issues=issues,
            )

        # 3. Rate limiting
        now = time.time()
        timestamps = self._call_timestamps.get(tool_name, [])
        # 清理 1 分钟前的记录
        timestamps = [t for t in timestamps if now - t < 60]
        if len(timestamps) >= perm.rate_limit_per_minute:
            issues.append(f"超出频率限制: {len(timestamps)}/{perm.rate_limit_per_minute} per min")
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.BLOCKED,
                sanitized_input=str(tool_input),
                issues=issues,
            )

        # 4. 参数长度检查
        input_str = str(tool_input)
        if len(input_str) > perm.max_input_length:
            issues.append(f"参数过长: {len(input_str)} > {perm.max_input_length}")
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.SUSPICIOUS,
                sanitized_input=input_str[:perm.max_input_length],
                issues=issues,
            )

        # 5. 需要确认
        if perm.requires_confirmation:
            issues.append("此 tool 需要用户确认")
            return ValidationResult(
                is_valid=False,
                risk_level=RiskLevel.SUSPICIOUS,
                sanitized_input=input_str,
                issues=issues,
            )

        return ValidationResult(
            is_valid=True,
            risk_level=RiskLevel.SAFE,
            sanitized_input=input_str,
            issues=issues,
        )

    def record_call(self, tool_name: str, tool_input: dict, result: Any = None):
        """记录 tool 调用（更新计数 + 审计日志）。"""
        # 更新计数
        self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1

        # 更新时间戳
        now = time.time()
        if tool_name not in self._call_timestamps:
            self._call_timestamps[tool_name] = []
        self._call_timestamps[tool_name].append(now)

        # 审计日志
        self._audit_log.append({
            "timestamp": now,
            "tool_name": tool_name,
            "input_preview": str(tool_input)[:200],
            "result_preview": str(result)[:200] if result else None,
            "session_call_count": self._call_counts[tool_name],
        })

    def get_audit_log(self) -> list[dict]:
        """获取审计日志。"""
        return self._audit_log.copy()

    def get_stats(self) -> dict:
        """获取 sandbox 统计。"""
        return {
            "call_counts": dict(self._call_counts),
            "total_calls": sum(self._call_counts.values()),
            "audit_log_size": len(self._audit_log),
        }

    def reset(self):
        """重置 session 状态。"""
        self._call_counts.clear()
        self._call_timestamps.clear()
        self._audit_log.clear()


# ═══════════════════════════════════════════════════════════════
# 组合：SecurityGuard（统一入口）
# ═══════════════════════════════════════════════════════════════

@dataclass
class SecurityCheckResult:
    """安全检查总结果。"""
    allowed: bool
    risk_level: RiskLevel
    sanitized_query: str
    validation: ValidationResult | None = None
    injection: InjectionCheckResult | None = None
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "risk_level": self.risk_level.value,
            "issues": self.issues,
            "validation": self.validation.to_dict() if self.validation else None,
            "injection": self.injection.to_dict() if self.injection else None,
        }


class SecurityGuard:
    """安全防御统一入口。

    在 Agent 处理用户输入之前，依次通过三层检查。
    可以直接集成到 FastAPI middleware 或 Agent 入口。

    用法：
        guard = SecurityGuard()
        check = guard.check_query("用户输入...")
        if not check.allowed:
            return f"输入被拦截: {check.issues}"
        # 用 check.sanitized_query 替代原始输入
        agent.invoke({"messages": [{"role": "user", "content": check.sanitized_query}]})
    """

    def __init__(self):
        self.validator = InputValidator()
        self.detector = InjectionDetector()
        self.sandbox = ToolSandbox()

    def check_query(self, query: str) -> SecurityCheckResult:
        """检查用户输入是否安全。"""
        issues = []

        # 第一层：输入校验
        validation = self.validator.validate(query)
        if not validation.is_valid:
            return SecurityCheckResult(
                allowed=False,
                risk_level=validation.risk_level,
                sanitized_query="",
                validation=validation,
                issues=validation.issues,
            )
        issues.extend(validation.issues)

        # 第二层：Injection 检测
        injection = self.detector.check(validation.sanitized_input)
        if injection.risk_level == RiskLevel.BLOCKED:
            return SecurityCheckResult(
                allowed=False,
                risk_level=RiskLevel.BLOCKED,
                sanitized_query="",
                validation=validation,
                injection=injection,
                issues=issues + injection.matched_patterns,
            )

        # SUSPICIOUS 的 injection 记录但放行（可配置为拦截）
        if injection.is_injection:
            issues.extend(injection.matched_patterns)
            logger.warning(f"可疑输入放行: {injection.matched_patterns}")

        # 最终风险等级
        if injection.risk_level == RiskLevel.SUSPICIOUS or validation.risk_level == RiskLevel.SUSPICIOUS:
            risk = RiskLevel.SUSPICIOUS
        else:
            risk = RiskLevel.SAFE

        return SecurityCheckResult(
            allowed=True,
            risk_level=risk,
            sanitized_query=validation.sanitized_input,
            validation=validation,
            injection=injection,
            issues=issues,
        )

    def check_tool_call(self, tool_name: str, tool_input: dict) -> ValidationResult:
        """检查 tool 调用权限。"""
        return self.sandbox.check_permission(tool_name, tool_input)

    def record_tool_call(self, tool_name: str, tool_input: dict, result: Any = None):
        """记录 tool 调用。"""
        self.sandbox.record_call(tool_name, tool_input, result)