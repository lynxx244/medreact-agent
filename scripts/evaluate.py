"""
evaluate.py
===========
批量评估 MedReAct Agent，对比 Agent 输出与 ground truth，输出评估报告。

评估维度：
  1. 风险等级准确率（Accuracy）         ← 主指标
  2. 高风险召回率（Recall@High）        ← 最重要！漏判高风险代价极大
  3. 建议关键词覆盖率（Advice Coverage） ← 软匹配
  4. 平均推理步数                        ← 效率指标
  5. 工具调用分布                        ← 行为分析

使用方式：
  python evaluate.py \
    --testset data/testset_labeled.jsonl \
    --output  results/eval_report.json  \
    --api-key YOUR_DEEPSEEK_KEY         \
    --max 50                              ← 先跑50条调试
    --mock                                 ← 加这个参数用 mock agent（不消耗 API）
"""

import json
import time
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict


# ── Mock Agent（调试用，不消耗 API）─────────────────────────────────────────

class MockAgent:
    """
    不调用真实 LLM，根据关键词模拟 Agent 输出。
    用于调试评估流程本身，确认 pipeline 跑通后再换真实 Agent。
    """
    def run(self, patient_input: str) -> str:
        # 简单规则模拟
        if any(kw in patient_input for kw in ["胸痛", "心跳", "呼吸困难", "出血", "昏迷"]):
            return "风险等级：高。建议立即拨打120或前往急诊就医。可能原因：心脏或呼吸系统急症。"
        elif any(kw in patient_input for kw in ["发烧", "咳嗽", "头痛", "腹痛", "感染"]):
            return "风险等级：中。建议尽快就医检查，注意休息，多喝水，监测体温变化。"
        else:
            return "风险等级：低。建议注意休息，清淡饮食，多喝水，症状加重时及时就医。"

    @property
    def step_count(self):
        return 3  # mock 固定步数

    @property
    def tool_calls(self):
        return {"ask_patient": 1, "risk_assess": 1, "search_symptom": 1}


# ── 真实 Agent（从 react_agent.py 导入）──────────────────────────────────────

def load_real_agent(api_key: str):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from react_agent import MedReActAgent, client as original_client
        from openai import OpenAI
        import react_agent as ra
        ra.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        ra.EVAL_MODE = True  # ← 加这一行
        print(f"[DEBUG] EVAL_MODE = {ra.EVAL_MODE}")
        return MedReActAgent(max_steps=8)
    except ImportError as e:
        raise ImportError(f"无法导入 react_agent.py，请确认文件存在: {e}")


# ── Agent 输出解析 ─────────────────────────────────────────────────────────

def extract_risk_from_output(agent_output: str) -> Optional[str]:
    """
    从 Agent 的 Final Answer 中提取风险等级。
    兼容多种输出格式：
      - "风险等级：高"
      - "风险等级：较低"
      - "建议立即就医" → 推断为高
      - "可在家观察"   → 推断为低
    """
    text = agent_output

    # 直接匹配风险等级字段
    patterns = [
        r'风险等级[：:]\s*(高|中|中等|中等偏低|较低|低)',
        r'风险[：:]\s*(高|中|中等|较低|低)',
        r'(高风险|中风险|中等风险|低风险|较低风险)',
    ]
    level_map = {
        "高": "高", "高风险": "高",
        "中": "中", "中等": "中", "中等偏低": "中", "中风险": "中", "中等风险": "中",
        "低": "低", "较低": "低", "低风险": "低", "较低风险": "低",
    }
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            raw = m.group(1)
            return level_map.get(raw, "中")  # 默认中

    # 关键词推断
    if any(kw in text for kw in ["立即就医", "急诊", "120", "马上就医", "危险"]):
        return "高"
    if any(kw in text for kw in ["建议就医", "尽快就医", "及时就医", "建议检查"]):
        return "中"
    if any(kw in text for kw in ["在家观察", "注意休息", "多喝水", "无需担心", "不用担心"]):
        return "低"
    # 补充更多中文表达兜底
    if any(kw in text for kw in ["风险较低", "风险评估较低", "目前风险低", "低风险"]):
        return "低"
    if any(kw in text for kw in ["风险较高", "风险评估较高", "高度警惕", "需要紧急", "立即前往急诊"]):
        return "高"
    if any(kw in text for kw in ["建议就医", "尽快就诊", "需要就诊", "风险中等"]):
        return "中"

    return None  # 无法解析
def extract_risk_with_llm(text: str, client) -> str:
    """
    用 LLM 解析 Agent 输出的风险等级，比正则更鲁棒。
    作为 extract_risk_from_output 的升级替代。
    """
    # 先试规则解析，能解析到就不消耗 API
    result = extract_risk_from_output(text)
    if result is not None:
        return result
    # 规则解析失败，才调用 LLM
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"""以下是一段医疗建议，请判断其风险等级。
只回答一个字：高、中、或低。不要有其他任何内容。

医疗建议：
{text[:300]}"""
            }],
            temperature=0.0,
            max_tokens=5,
        )
        result = response.choices[0].message.content.strip()
        if "高" in result:
            return "高"
        elif "中" in result:
            return "中"
        elif "低" in result:
            return "低"
    except Exception as e:
        print(f"  [LLM解析失败]: {e}")
    return None

def check_advice_coverage(agent_output: str, keywords: list) -> float:
    """
    检查 Agent 输出覆盖了多少 ground truth 建议关键词。
    返回覆盖率 0.0~1.0。
    """
    if not keywords:
        return 1.0  # 没有 ground truth 关键词，默认满分
    covered = sum(1 for kw in keywords if kw in agent_output)
    return covered / len(keywords)


# ── 评估主逻辑 ────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, agent, testset_path: str, output_path: str, max_samples: int = 100, score_client=None):
        self.agent = agent
        self.testset_path = testset_path
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples

        self.score_client = score_client
        self.results = []
        self.errors = []

    def run(self):
        print(f"\n{'='*60}")
        print(f"MedReAct 评估开始")
        print(f"测试集: {self.testset_path}")
        print(f"最大样本: {self.max_samples}")
        print(f"{'='*60}\n")

        with open(self.testset_path, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f][:self.max_samples]

        for i, sample in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] ID={sample['id']} 风险={sample['ground_truth_risk']}")
            print(f"  问题: {sample['symptoms'][:60]}...")

            result = self._evaluate_one(sample)
            self.results.append(result)

            # 实时打印单条结果
            status = "✅" if result["risk_correct"] else "❌"
            print(f"  {status} 预测={result['predicted_risk']} 真实={result['ground_truth_risk']} "
                  f"覆盖率={result['advice_coverage']:.0%}")

            # 避免 API 限速（真实 Agent 时启用）
            time.sleep(0.5)

        self._save_report()

    def _evaluate_one(self, sample: dict) -> dict:
        start_time = time.time()

        # 运行 Agent
        try:
            agent_output = self.agent.run(sample["symptoms"])
            elapsed = time.time() - start_time
            error = None
        except Exception as e:
            agent_output = ""
            elapsed = time.time() - start_time
            error = str(e)
            self.errors.append({"id": sample["id"], "error": error})
            print(f"  ⚠️  Agent 出错: {error}")

        # 解析 Agent 输出
        predicted_risk = extract_risk_with_llm(agent_output, self.score_client) \
            if (agent_output and self.score_client) \
            else extract_risk_from_output(agent_output)
        advice_coverage = check_advice_coverage(
            agent_output,
            sample.get("ground_truth_advice_keywords", [])
        )

        # 风险等级是否正确
        risk_correct = (predicted_risk == sample["ground_truth_risk"]) if predicted_risk else False

        # 获取步数和工具调用（如果 Agent 支持）
        step_count = getattr(self.agent, "step_count", None)
        tool_calls = getattr(self.agent, "tool_calls", {})

        return {
            "id": sample["id"],
            "symptoms": sample["symptoms"],
            "ground_truth_risk": sample["ground_truth_risk"],
            "predicted_risk": predicted_risk,
            "risk_correct": risk_correct,
            "advice_coverage": advice_coverage,
            "agent_output": agent_output[:500],  # 截断，避免结果文件过大
            "step_count": step_count,
            "tool_calls": tool_calls,
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
            "label_source": sample.get("label_source", "unknown"),
        }

    def _save_report(self):
        """计算汇总指标，保存完整报告。"""

        n = len(self.results)
        if n == 0:
            print("没有有效结果，报告为空")
            return

        # 基础准确率
        correct = sum(1 for r in self.results if r["risk_correct"])
        accuracy = correct / n

        # 按风险等级分层统计
        by_level = defaultdict(lambda: {"total": 0, "correct": 0, "predicted_as": defaultdict(int)})
        for r in self.results:
            gt = r["ground_truth_risk"]
            pred = r["predicted_risk"] or "未知"
            by_level[gt]["total"] += 1
            if r["risk_correct"]:
                by_level[gt]["correct"] += 1
            by_level[gt]["predicted_as"][pred] += 1

        # 高风险召回率（最重要指标！）
        high_total = by_level["高"]["total"]
        high_correct = by_level["高"]["correct"]
        high_recall = high_correct / high_total if high_total > 0 else None

        # 平均建议覆盖率
        avg_coverage = sum(r["advice_coverage"] for r in self.results) / n

        # 无法解析率
        unparseable = sum(1 for r in self.results if r["predicted_risk"] is None)

        # 平均耗时
        avg_time = sum(r["elapsed_seconds"] for r in self.results) / n

        # ── 汇总报告 ─────────────────────────────────────────────────────────
        report = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "testset": self.testset_path,
                "total_samples": n,
                "error_count": len(self.errors),
            },
            "metrics": {
                "overall_accuracy": round(accuracy, 4),
                "high_risk_recall": round(high_recall, 4) if high_recall is not None else None,
                "avg_advice_coverage": round(avg_coverage, 4),
                "unparseable_rate": round(unparseable / n, 4),
                "avg_elapsed_seconds": round(avg_time, 2),
            },
            "by_risk_level": {
                level: {
                    "total": v["total"],
                    "correct": v["correct"],
                    "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else None,
                    "confusion": dict(v["predicted_as"]),
                }
                for level, v in by_level.items()
            },
            "errors": self.errors,
            "results": self.results,
        }

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # ── 打印摘要 ─────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"📊 评估报告")
        print(f"{'='*60}")
        print(f"  样本总数      : {n}")
        print(f"  整体准确率    : {accuracy:.1%}  ({correct}/{n})")
        if high_recall is not None:
            safety_flag = "⚠️ 偏低！" if high_recall < 0.8 else "✅"
            print(f"  高风险召回率  : {high_recall:.1%}  ({high_correct}/{high_total}) {safety_flag}")
        print(f"  建议覆盖率    : {avg_coverage:.1%}")
        print(f"  无法解析率    : {unparseable/n:.1%}")
        print(f"  平均耗时      : {avg_time:.1f}s/条")
        print(f"\n  分层准确率：")
        for level in ["高", "中", "低"]:
            v = by_level.get(level, {})
            if v.get("total", 0) > 0:
                acc = v['correct'] / v['total']
                print(f"    {level}风险: {acc:.1%}  ({v['correct']}/{v['total']})")
                print(f"      预测分布: {dict(v['predicted_as'])}")
        print(f"\n  错误数        : {len(self.errors)}")
        print(f"  报告已保存至  : {self.output_path}")
        print(f"{'='*60}")


# ── 入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 MedReAct Agent")
    parser.add_argument("--testset", default="data/testset_labeled.jsonl", help="测试集路径")
    parser.add_argument("--output",  default="results/eval_report.json",   help="报告输出路径")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key")
    parser.add_argument("--max",     type=int, default=50, help="最大测试样本数")
    parser.add_argument("--mock",    action="store_true", help="使用 mock agent（不消耗 API）")
    args = parser.parse_args()

    # 初始化 Agent
    if args.mock:
        print("⚠️  使用 Mock Agent（仅用于调试 pipeline）")
        agent = MockAgent()
    else:
        if not args.api_key:
            raise ValueError("请提供 --api-key，或使用 --mock 模式")
        print("🚀 加载真实 MedReActAgent...")
        agent = load_real_agent(args.api_key)

    # 运行评估
    evaluator = Evaluator(
        agent=agent,
        testset_path=args.testset,
        output_path=args.output,
        max_samples=args.max,
    )
    evaluator.run()