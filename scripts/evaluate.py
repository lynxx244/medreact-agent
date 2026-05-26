"""
批量评估 MedReAct Agent，对比 Agent 输出与 ground truth，输出评估报告。

使用方式：
  python evaluate.py \
    --testset data/testset_labeled_llm.jsonl \
    --output  results/eval_report.json  \
    --api-key YOUR_DEEPSEEK_KEY         \
    --max 200                           \
    --workers 4
"""

import json
import time
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class MockAgent:
    def run(self, patient_input: str) -> str:
        if any(kw in patient_input for kw in ["胸痛", "心跳", "呼吸困难", "出血", "昏迷"]):
            return "风险等级：高。建议立即拨打120或前往急诊就医。"
        elif any(kw in patient_input for kw in ["发烧", "咳嗽", "头痛", "腹痛", "感染"]):
            return "风险等级：中。建议尽快就医检查。"
        else:
            return "风险等级：低。建议注意休息，多喝水。"


_thread_local = threading.local()

def get_thread_agent(api_key: str):
    """每个线程第一次调用时创建自己的 agent 实例，之后复用。"""
    if not hasattr(_thread_local, "agent"):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from react_agent import MedReActAgent
        from openai import OpenAI
        import react_agent as ra

        thread_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        ra.client = thread_client
        ra.EVAL_MODE = True

        _thread_local.agent = MedReActAgent(max_steps=8)

    return _thread_local.agent


def extract_risk_from_output(agent_output: str) -> Optional[str]:
    text = agent_output
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
            return level_map.get(m.group(1), "中")
    if any(kw in text for kw in ["立即就医", "急诊", "120", "马上就医"]):
        return "高"
    if any(kw in text for kw in ["建议就医", "尽快就医", "及时就医"]):
        return "中"
    if any(kw in text for kw in ["在家观察", "注意休息", "多喝水", "无需担心"]):
        return "低"
    return None


def check_advice_coverage(agent_output: str, keywords: list) -> float:
    if not keywords:
        return 1.0
    return sum(1 for kw in keywords if kw in agent_output) / len(keywords)


class Evaluator:
    def __init__(self, agent_or_key, testset_path, output_path,
                 max_samples=100, workers=4, use_mock=False):
        self.agent_or_key = agent_or_key
        self.testset_path = testset_path
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.workers = workers
        self.use_mock = use_mock
        self.results = []
        self.errors = []
        self._lock = threading.Lock()
        self._counter = [0]

    def _get_agent(self):
        if self.use_mock:
            return self.agent_or_key
        return get_thread_agent(self.agent_or_key)

    def run(self):
        print(f"\n{'='*60}")
        print(f"MedReAct 评估开始  并行线程: {self.workers}")
        print(f"{'='*60}\n")

        with open(self.testset_path, encoding="utf-8") as f:
            samples = [json.loads(l) for l in f][:self.max_samples]

        total = len(samples)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._evaluate_one, s, total): s for s in samples}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    sample = futures[future]
                    result = self._error_result(sample, str(e))
                with self._lock:
                    self.results.append(result)

        self._save_report()

    def _evaluate_one(self, sample: dict, total: int) -> dict:
        start_time = time.time()
        agent = self._get_agent()
        try:
            agent_output = agent.run(sample["symptoms"])
            elapsed = time.time() - start_time
            error = None
        except Exception as e:
            agent_output = ""
            elapsed = time.time() - start_time
            error = str(e)
            with self._lock:
                self.errors.append({"id": sample["id"], "error": error})

        predicted_risk = extract_risk_from_output(agent_output)
        advice_coverage = check_advice_coverage(
            agent_output, sample.get("ground_truth_advice_keywords", []))
        risk_correct = (predicted_risk == sample["ground_truth_risk"]) if predicted_risk else False

        with self._lock:
            self._counter[0] += 1
            status = "✅" if risk_correct else "❌"
            print(f"  [{self._counter[0]}/{total}] {status} ID={sample['id']} "
                  f"预测={predicted_risk} 真实={sample['ground_truth_risk']} "
                  f"耗时={elapsed:.1f}s")

        return {
            "id": sample["id"],
            "symptoms": sample["symptoms"],
            "ground_truth_risk": sample["ground_truth_risk"],
            "predicted_risk": predicted_risk,
            "risk_correct": risk_correct,
            "advice_coverage": advice_coverage,
            "agent_output": agent_output[:500],
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
            "label_source": sample.get("label_source", "unknown"),
        }

    def _error_result(self, sample, error_msg):
        with self._lock:
            self.errors.append({"id": sample["id"], "error": error_msg})
        return {
            "id": sample["id"], "symptoms": sample["symptoms"],
            "ground_truth_risk": sample["ground_truth_risk"],
            "predicted_risk": None, "risk_correct": False,
            "advice_coverage": 0.0, "agent_output": "",
            "elapsed_seconds": 0, "error": error_msg,
            "label_source": sample.get("label_source", "unknown"),
        }

    def _save_report(self):
        n = len(self.results)
        if n == 0:
            print("没有有效结果")
            return

        correct = sum(1 for r in self.results if r["risk_correct"])
        accuracy = correct / n

        by_level = defaultdict(lambda: {"total": 0, "correct": 0, "predicted_as": defaultdict(int)})
        for r in self.results:
            gt = r["ground_truth_risk"]
            pred = r["predicted_risk"] or "未知"
            by_level[gt]["total"] += 1
            if r["risk_correct"]:
                by_level[gt]["correct"] += 1
            by_level[gt]["predicted_as"][pred] += 1

        high_total = by_level["高"]["total"]
        high_correct = by_level["高"]["correct"]
        high_recall = high_correct / high_total if high_total > 0 else None
        avg_coverage = sum(r["advice_coverage"] for r in self.results) / n
        unparseable = sum(1 for r in self.results if r["predicted_risk"] is None)
        avg_time = sum(r["elapsed_seconds"] for r in self.results) / n

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
                    "total": v["total"], "correct": v["correct"],
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

        print(f"\n{'='*60}")
        print(f"📊 评估报告")
        print(f"{'='*60}")
        print(f"  样本总数      : {n}")
        print(f"  整体准确率    : {accuracy:.1%}  ({correct}/{n})")
        if high_recall is not None:
            flag = "⚠️ 偏低！" if high_recall < 0.8 else "✅"
            print(f"  高风险召回率  : {high_recall:.1%}  ({high_correct}/{high_total}) {flag}")
        print(f"  建议覆盖率    : {avg_coverage:.1%}")
        print(f"  无法解析率    : {unparseable/n:.1%}")
        print(f"  平均耗时      : {avg_time:.1f}s/条（单条，非总时间）")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 MedReAct Agent")
    parser.add_argument("--testset",  default="data/testset_labeled.jsonl")
    parser.add_argument("--output",   default="results/eval_report.json")
    parser.add_argument("--api-key",  default=None)
    parser.add_argument("--max",      type=int, default=50)
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--mock",     action="store_true")
    args = parser.parse_args()

    if args.mock:
        evaluator = Evaluator(
            agent_or_key=MockAgent(),
            testset_path=args.testset,
            output_path=args.output,
            max_samples=args.max,
            workers=args.workers,
            use_mock=True,
        )
    else:
        if not args.api_key:
            raise ValueError("请提供 --api-key")
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import react_agent as ra
        ra.EVAL_MODE = True
        evaluator = Evaluator(
            agent_or_key=args.api_key,
            testset_path=args.testset,
            output_path=args.output,
            max_samples=args.max,
            workers=args.workers,
            use_mock=False,
        )

    evaluator.run()