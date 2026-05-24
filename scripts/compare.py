"""
compare.py（v2）
===============
读取三组评估报告，生成横向对比表格，包含建议质量评分。

用法：
  python scripts/compare.py
  python scripts/compare.py --markdown
"""

import json
import argparse
from pathlib import Path


REPORT_FILES = {
    "A. 纯LLM":     "results/eval_report_A_plainllm.json",
    "B. 简化Agent":  "results/eval_report_B_liteagent.json",
    "C. MedReAct":  "results/eval_report_llm.json",
}


def load_report(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def print_comparison(show_markdown=False):
    reports = {}
    for name, path in REPORT_FILES.items():
        r = load_report(path)
        if r:
            reports[name] = r
        else:
            print(f"⚠️  未找到: {path}（跳过）")

    if not reports:
        print("没有找到任何报告，请先运行评估。")
        return

    print(f"\n{'='*72}")
    print(f"  MedReAct 消融实验对比报告")
    print(f"{'='*72}")

    col_w = 16

    # ── 核心安全指标 ──────────────────────────────────────────────────────
    print("\n【安全性与准确率】")
    safety_metrics = [
        ("overall_accuracy",  "整体准确率"),
        ("high_risk_recall",  "高风险召回率 ⚠️"),
        ("unparseable_rate",  "无法解析率"),
        ("avg_elapsed_seconds", "平均耗时(s)"),
    ]
    print(f"  {'指标':<18}", end="")
    for name in reports:
        print(f"  {name:<{col_w}}", end="")
    print()
    print(f"  {'-'*18}" + f"  {'-'*col_w}" * len(reports))

    for key, label in safety_metrics:
        print(f"  {label:<18}", end="")
        values = [r["metrics"].get(key) for r in reports.values()]
        for val in values:
            if val is None:
                cell = "N/A"
            elif key == "avg_elapsed_seconds":
                cell = f"{val:.1f}s"
            else:
                cell = f"{val:.1%}"
            numeric = [v for v in values if v is not None]
            if numeric and val is not None:
                if key in ("unparseable_rate", "avg_elapsed_seconds"):
                    is_best = val == min(numeric)
                else:
                    is_best = val == max(numeric)
                if is_best:
                    cell = f"★ {cell}"
            print(f"  {cell:<{col_w}}", end="")
        print()

    # ── 建议质量评分 ──────────────────────────────────────────────────────
    has_advice = any(
        r["metrics"].get("advice_quality") for r in reports.values()
    )
    if has_advice:
        print(f"\n【建议质量评分（满分10分，越高越好）】")
        advice_dims = ["处置方向", "关键信息", "安全性", "总评"]
        print(f"  {'维度':<18}", end="")
        for name in reports:
            print(f"  {name:<{col_w}}", end="")
        print()
        print(f"  {'-'*18}" + f"  {'-'*col_w}" * len(reports))
        for dim in advice_dims:
            print(f"  {dim:<18}", end="")
            values = [
                r["metrics"].get("advice_quality", {}).get(dim)
                for r in reports.values()
            ]
            for val in values:
                if val is None:
                    cell = "N/A"
                else:
                    cell = f"{val:.1f}/10"
                numeric = [v for v in values if v is not None]
                if numeric and val is not None and val == max(numeric):
                    cell = f"★ {cell}"
                print(f"  {cell:<{col_w}}", end="")
            print()

    # ── 分层准确率 ────────────────────────────────────────────────────────
    print(f"\n【分层准确率】")
    for level in ["高", "中", "低"]:
        print(f"  {level}风险:")
        for name, report in reports.items():
            v = report.get("by_risk_level", {}).get(level, {})
            if v and v.get("total", 0) > 0:
                acc = v.get("accuracy", 0) or 0
                bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
                print(f"    {name}: [{bar}] {acc:.1%} ({v['correct']}/{v['total']})")

    # ── 结论 ──────────────────────────────────────────────────────────────
    print(f"\n【结论】")
    if "C. MedReAct" in reports:
        medreact = reports["C. MedReAct"]["metrics"]
        for name in ["A. 纯LLM", "B. 简化Agent"]:
            if name in reports:
                other = reports[name]["metrics"]
                acc_diff = medreact["overall_accuracy"] - other["overall_accuracy"]
                direction = "提升" if acc_diff > 0 else "下降"
                print(f"  MedReAct vs {name}：整体准确率{direction} {abs(acc_diff):.1%}", end="")
                # 建议质量对比
                mq = medreact.get("advice_quality", {}).get("总评")
                oq = other.get("advice_quality", {}).get("总评")
                if mq and oq:
                    q_diff = mq - oq
                    q_dir = "高" if q_diff > 0 else "低"
                    print(f"，建议质量{q_dir} {abs(q_diff):.1f}分")
                else:
                    print()

    # ── Markdown 表格 ─────────────────────────────────────────────────────
    if show_markdown:
        print(f"\n【Markdown 表格】\n")
        print("```markdown")
        header = "| 指标 | " + " | ".join(reports.keys()) + " |"
        sep = "|------|" + "------|" * len(reports)
        print(header)
        print(sep)
        all_metrics = safety_metrics[:]
        for key, label in all_metrics:
            row = f"| {label} |"
            for r in reports.values():
                val = r["metrics"].get(key)
                if val is None:
                    row += " N/A |"
                elif key == "avg_elapsed_seconds":
                    row += f" {val:.1f}s |"
                else:
                    row += f" {val:.1%} |"
            print(row)
        if has_advice:
            for dim in ["处置方向", "关键信息", "安全性", "总评"]:
                row = f"| 建议质量-{dim} |"
                for r in reports.values():
                    val = r["metrics"].get("advice_quality", {}).get(dim)
                    row += f" {val:.1f}/10 |" if val is not None else " N/A |"
                print(row)
        print("```")

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    print_comparison(show_markdown=args.markdown)