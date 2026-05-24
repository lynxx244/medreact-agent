"""
report.py
=========
读取 evaluate.py 生成的 JSON 报告，打印可读性强的分析报告。
也可以生成用于论文/实习展示的 Markdown 表格。

用法：
  python report.py --input results/eval_report.json
  python report.py --input results/eval_report.json --markdown   # 输出 Markdown 表格
  python report.py --input results/eval_report.json --errors     # 详细展示错误案例
"""

import json
import argparse
from pathlib import Path


def print_report(report_path: str, show_markdown: bool = False, show_errors: bool = False):
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    meta = report["meta"]
    metrics = report["metrics"]
    by_level = report["by_risk_level"]

    print(f"\n{'='*65}")
    print(f"  MedReAct Agent 评估报告")
    print(f"  生成时间: {meta['timestamp'][:19]}")
    print(f"  测试集  : {meta['testset']}")
    print(f"  样本数  : {meta['total_samples']}  (错误: {meta['error_count']})")
    print(f"{'='*65}")

    # ── 核心指标 ──────────────────────────────────────────────────────────
    print("\n【核心指标】")
    acc = metrics["overall_accuracy"]
    acc_icon = "🟢" if acc >= 0.7 else ("🟡" if acc >= 0.5 else "🔴")
    print(f"  {acc_icon} 整体准确率        : {acc:.1%}")

    if metrics["high_risk_recall"] is not None:
        recall = metrics["high_risk_recall"]
        recall_icon = "🟢" if recall >= 0.8 else ("🟡" if recall >= 0.6 else "🔴")
        print(f"  {recall_icon} 高风险召回率      : {recall:.1%}   ← 最重要，漏判代价极大")
    else:
        print(f"  ⚪ 高风险召回率      : N/A（测试集中无高风险样本）")

    cov = metrics["avg_advice_coverage"]
    cov_icon = "🟢" if cov >= 0.6 else ("🟡" if cov >= 0.4 else "🔴")
    print(f"  {cov_icon} 建议关键词覆盖率  : {cov:.1%}")

    unp = metrics["unparseable_rate"]
    unp_icon = "🟢" if unp <= 0.05 else ("🟡" if unp <= 0.15 else "🔴")
    print(f"  {unp_icon} 输出无法解析率    : {unp:.1%}")
    print(f"  ⏱️  平均耗时/条        : {metrics['avg_elapsed_seconds']:.1f}s")

    # ── 分层准确率 ────────────────────────────────────────────────────────
    print("\n【分层准确率】")
    for level in ["高", "中", "低"]:
        v = by_level.get(level)
        if not v or v["total"] == 0:
            continue
        acc_l = v["accuracy"]
        bar = "█" * int(acc_l * 20) + "░" * (20 - int(acc_l * 20))
        print(f"  {level}风险: [{bar}] {acc_l:.1%}  ({v['correct']}/{v['total']})")

    # ── 混淆矩阵 ──────────────────────────────────────────────────────────
    print("\n【混淆矩阵】（行=真实，列=预测）")
    levels = ["高", "中", "低", "未知"]
    # 表头
    print(f"  {'真实\\预测':<8}", end="")
    for l in levels:
        print(f"  {l:^6}", end="")
    print()
    print(f"  {'-'*40}")
    for level in ["高", "中", "低"]:
        v = by_level.get(level)
        if not v or v["total"] == 0:
            continue
        print(f"  {level+'风险':<8}", end="")
        for pred_l in levels:
            count = v["confusion"].get(pred_l, 0)
            print(f"  {count:^6}", end="")
        print()

    # ── 错误案例分析 ──────────────────────────────────────────────────────
    if show_errors and report.get("results"):
        wrong = [r for r in report["results"] if not r["risk_correct"]]
        print(f"\n【错误案例分析】（共 {len(wrong)} 条）")
        # 按风险等级分组展示
        for level in ["高", "中", "低"]:
            level_wrong = [r for r in wrong if r["ground_truth_risk"] == level][:3]
            if not level_wrong:
                continue
            print(f"\n  ▼ 真实={level}风险，被误判的案例：")
            for r in level_wrong:
                print(f"    ID={r['id']} 预测={r['predicted_risk']}")
                print(f"    问题: {r['symptoms'][:60]}")
                print(f"    输出: {r['agent_output'][:100]}...")
                print()

    # ── Markdown 表格（用于报告/论文）────────────────────────────────────
    if show_markdown:
        print("\n【Markdown 表格（可直接粘贴到报告）】")
        print("```markdown")
        print("| 指标 | 数值 |")
        print("|------|------|")
        print(f"| 整体准确率 | {metrics['overall_accuracy']:.1%} |")
        if metrics['high_risk_recall'] is not None:
            print(f"| 高风险召回率 | {metrics['high_risk_recall']:.1%} |")
        print(f"| 建议关键词覆盖率 | {metrics['avg_advice_coverage']:.1%} |")
        print(f"| 输出无法解析率 | {metrics['unparseable_rate']:.1%} |")
        print(f"| 平均响应时间 | {metrics['avg_elapsed_seconds']:.1f}s |")
        print(f"| 测试样本数 | {meta['total_samples']} |")
        print("```")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看评估报告")
    parser.add_argument("--input",    default="results/eval_report.json", help="报告 JSON 路径")
    parser.add_argument("--markdown", action="store_true", help="输出 Markdown 格式表格")
    parser.add_argument("--errors",   action="store_true", help="显示错误案例详情")
    args = parser.parse_args()

    print_report(args.input, show_markdown=args.markdown, show_errors=args.errors)