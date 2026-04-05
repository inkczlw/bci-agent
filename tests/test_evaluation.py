"""评估 pipeline 入口脚本。

用法：
    python -m tests.test_evaluation all          # 跑全部 test case
    python -m tests.test_evaluation factual      # 只跑事实查询类
    python -m tests.test_evaluation analysis     # 只跑结构化分析类
    python -m tests.test_evaluation comparison   # 只跑对比类
    python -m tests.test_evaluation rag          # 只跑 RAG 类
    python -m tests.test_evaluation edge_case    # 只跑边界情况
    python -m tests.test_evaluation single fact_01  # 跑单个 case
    python -m tests.test_evaluation quick        # 快速验证（每类各 1 个）
"""

import sys
import time

from agents.bci_agent import create_bci_agent
from config import get_llm
from evaluation.test_cases import TestCategory, get_test_cases, EVAL_TEST_CASES
from evaluation.evaluator import EvaluationEngine
from evaluation.report import EvalReport


def run_eval(test_cases, use_judge: bool = True):
    """执行评估并输出报告。"""
    print(f"\n🚀 开始评估，共 {len(test_cases)} 个用例")
    print(f"   LLM Judge: {'启用' if use_judge else '禁用'}")
    print()

    # 初始化 Agent
    agent = create_bci_agent()
    llm = get_llm() if use_judge else None

    # 创建评估引擎
    engine = EvaluationEngine(agent=agent, llm=llm, verbose=True)

    # 执行评估
    start = time.time()
    results = engine.run(test_cases)
    elapsed = time.time() - start

    # 生成报告
    report = EvalReport(results)
    console_output = report.generate_console_report()
    print(console_output)

    # 保存 JSON
    json_path = report.save_json()
    print(f"\n📄 详细结果已保存: {json_path}")
    print(f"⏱️ 总耗时: {elapsed:.1f}s")

    # 返回整体通过率（用于 CI 判断）
    overall = report._calc_overall()
    return overall["pass_rate"] >= 0.6


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    # 按类别筛选
    category_map = {
        "factual": TestCategory.FACTUAL,
        "analysis": TestCategory.ANALYSIS,
        "comparison": TestCategory.COMPARISON,
        "rag": TestCategory.RAG,
        "edge_case": TestCategory.EDGE_CASE,
    }

    if command == "all":
        cases = EVAL_TEST_CASES
    elif command == "quick":
        # 每类取第一个，快速验证
        seen = set()
        cases = []
        for tc in EVAL_TEST_CASES:
            if tc.category not in seen:
                cases.append(tc)
                seen.add(tc.category)
    elif command == "single" and len(sys.argv) >= 3:
        case_id = sys.argv[2]
        cases = get_test_cases(ids=[case_id])
        if not cases:
            print(f"❌ 找不到 test case: {case_id}")
            print(f"   可用: {[tc.id for tc in EVAL_TEST_CASES]}")
            sys.exit(1)
    elif command in category_map:
        cases = get_test_cases(category=category_map[command])
    elif command == "no-judge":
        # 不启用 LLM judge（省 token，跑得快）
        cases = EVAL_TEST_CASES
        passed = run_eval(cases, use_judge=False)
        sys.exit(0 if passed else 1)
    else:
        print(f"❌ 未知命令: {command}")
        print(__doc__)
        sys.exit(1)

    passed = run_eval(cases)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()