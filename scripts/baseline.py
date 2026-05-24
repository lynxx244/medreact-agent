"""
baseline.py（v2）
================
两个对照组，与完整 MedReAct 做消融实验对比。
新增：建议质量 LLM 打分（与 evaluate.py 保持一致）

用法：
  python scripts/baseline.py \
    --testset data/testset_labeled_llm.jsonl \
    --output  results/ \
    --api-key YOUR_KEY \
    --max 50

  # 只跑某一组
  python scripts/baseline.py ... --group A
  python scripts/baseline.py ... --group B

  # 不评估建议质量（省 API 费用）
  python scripts/baseline.py ... --no-advice-score
"""

import json
import time
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from openai import OpenAI


# ── 公共函数（与 evaluate.py 保持一致）───────────────────────────────────

def extract_risk(text: str):
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
    if any(kw in text for kw in ["立即就医", "急诊", "120", "马上就医", "紧急"]):
        return "高"
    if any(kw in text for kw in ["风险较高", "高度警惕", "需要紧急", "立即前往急诊"]):
        return "高"
    if any(kw in text for kw in ["建议就医", "尽快就诊", "需要就诊", "风险中等", "及时就医"]):
        return "中"
    if any(kw in text for kw in ["风险较低", "在家观察", "注意休息", "不用担心", "无需担心"]):
        return "低"
    return None
def extract_risk_with_llm(text: str, client) -> str:
    result = extract_risk(text)
    if result is not None:
        return result
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

def llm_advice_score(agent_output: str, doctor_answer: str, client: OpenAI) -> dict:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"""你是医疗质量评审专家。对比以下两份建议，评估AI助手的建议质量。

医生原始医嘱：
{doctor_answer[:400]}

AI助手建议：
{agent_output[:400]}

从以下三个维度打分（每项0-10分）：
1. 处置方向正确性：建议的就医/观察/用药方向是否和医生一致（10=完全一致，0=完全相反）
2. 关键信息覆盖：医生提到的重要注意事项AI是否也提到了（10=全部覆盖，0=完全没有）
3. 安全性：AI是否遗漏了医生提到的任何紧急信号或风险提示（10=无遗漏，0=严重遗漏）

只回答以下JSON格式，不要有其他内容：
{{"处置方向": 8, "关键信息": 6, "安全性": 9, "总评": 8, "点评": "建议方向一致但缺少用药指导"}}"""
            }],
            temperature=0.0,
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r'```json|```', '', text).strip()
        return json.loads(text)
    except Exception as e:
        return {"处置方向": 0, "关键信息": 0, "安全性": 0, "总评": 0, "点评": f"打分失败:{e}"}


# ── 对照组A：纯 LLM ──────────────────────────────────────────────────────

PLAIN_LLM_PROMPT = """你是一个医疗预问诊助手。根据患者描述，给出初步分诊建议。

必须严格按照以下格式输出：
风险等级：高/中/低
可能原因：xxx
建议行动：xxx
注意事项：xxx

风险等级定义：
- 高：需要立即急诊处理（生命体征异常、急性心脑血管事件、大量出血等）
- 中：建议尽快就医检查（症状持续、需要明确诊断）
- 低：可先在家观察（普通症状、无红旗征象）"""


class PlainLLMAgent:
    def __init__(self, client: OpenAI):
        self.client = client

    def run(self, patient_input: str) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": PLAIN_LLM_PROMPT},
                {"role": "user", "content": f"患者描述：{patient_input}"},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content


# ── 对照组B：简化版 Agent（只有 risk_assess）────────────────────────────

LITE_AGENT_SYSTEM = """你是一个医疗预问诊 Agent，只能使用 risk_assess 工具。

调用格式：
Action: risk_assess(["症状1", "症状2"], 持续天数)

输出格式一（调用工具）：
Thought: [推理]
Action: risk_assess(["症状列表"], 天数)

输出格式二（最终答案）：
Thought: [推理]
Final Answer:
风险等级：高/中/低
可能原因：xxx
建议行动：xxx
注意事项：xxx

注意：最多调用一次工具，从患者描述直接推断症状，不要追问。"""


class LiteAgent:
    def __init__(self, client: OpenAI):
        self.client = client

    def _risk_assess(self, symptoms: list, duration_days: int) -> str:
        symptom_str = "、".join(symptoms)
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"""你是急诊分诊专家。判断以下症状是否属于需要立即急诊处理的红旗症状。

红旗症状标准（必须满足其中之一）：
- 生命体征异常：意识丧失、休克、呼吸困难、大量出血
- 急性心脑血管：胸痛伴大汗、突发剧烈头痛、口眼歪斜、半身不遂
- 急腹症：剧烈腹痛伴板状腹
- 严重过敏：全身荨麻疹伴喉咙水肿

以下不属于红旗症状：慢性病、皮肤病、月经异常、普通消化道症状

症状列表：{symptom_str}

只回答：
危险症状：有
原因：[原因]
或：
危险症状：无
原因：无"""
            }],
            temperature=0.0,
        )
        result = response.choices[0].message.content
        if "危险症状：有" in result:
            reason = result.split("原因：")[-1].strip()
            return f"风险等级：高，{reason}，建议立即就医"
        if duration_days > 7:
            return "风险等级：中等，症状持续较长，建议尽快就医"
        elif duration_days > 3:
            return "风险等级：中等偏低，建议就医观察"
        else:
            return "风险等级：较低，可先在家观察"

    def run(self, patient_input: str) -> str:
        messages = [
            {"role": "system", "content": LITE_AGENT_SYSTEM},
            {"role": "user", "content": f"患者描述：{patient_input}"},
        ]
        response = self.client.chat.completions.create(
            model="deepseek-chat", messages=messages, temperature=0.1,
        )
        output = response.choices[0].message.content

        if "Final Answer:" in output:
            return output.split("Final Answer:")[-1].strip()

        if "Action:" in output and "risk_assess" in output:
            action_str = output.split("Action:")[1].strip()
            match = re.match(r'risk_assess\((.*)\)', action_str, re.DOTALL)
            if match:
                try:
                    observation = eval(f"self._risk_assess({match.group(1)})")
                except Exception:
                    observation = "风险评估失败，建议就医"
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": f"Observation: {observation}\n请立即给出Final Answer。"})
                response2 = self.client.chat.completions.create(
                    model="deepseek-chat", messages=messages, temperature=0.1,
                )
                output2 = response2.choices[0].message.content
                if "Final Answer:" in output2:
                    return output2.split("Final Answer:")[-1].strip()
                return output2

        # 兜底
        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": "请立即给出Final Answer，格式：风险等级：高/中/低\n可能原因：\n建议行动：\n注意事项："})
        response3 = self.client.chat.completions.create(
            model="deepseek-chat", messages=messages, temperature=0.1,
        )
        return response3.choices[0].message.content


# ── 评估单个 Agent ────────────────────────────────────────────────────────

def evaluate_agent(agent, agent_name: str, samples: list, output_path: str,
                   score_client: OpenAI = None):
    print(f"\n{'='*60}")
    print(f"评估: {agent_name}  样本数: {len(samples)}")
    print(f"建议质量打分: {'启用' if score_client else '跳过'}")
    print(f"{'='*60}")

    results = []
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] ID={sample['id']} 真实={sample['ground_truth_risk']}", end=" ")
        start = time.time()
        try:
            output = agent.run(sample["symptoms"])
            error = None
        except Exception as e:
            output = ""
            error = str(e)

        elapsed = time.time() - start
        predicted = extract_risk_with_llm(output, score_client) \
            if (output and score_client) \
            else extract_risk(output)
        correct = (predicted == sample["ground_truth_risk"]) if predicted else False

        advice_score = None
        if score_client and output and sample.get("doctor_answer"):
            advice_score = llm_advice_score(output, sample["doctor_answer"], score_client)

        status = "✅" if correct else "❌"
        score_str = f" 评分={advice_score['总评']}/10" if advice_score else ""
        print(f"→ 预测={predicted} {status} ({elapsed:.1f}s){score_str}")

        results.append({
            "id": sample["id"],
            "symptoms": sample["symptoms"],
            "ground_truth_risk": sample["ground_truth_risk"],
            "predicted_risk": predicted,
            "risk_correct": correct,
            "advice_score": advice_score,
            "agent_output": output[:300],
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
        })
        time.sleep(0.3)

    # 汇总指标
    n = len(results)
    correct_count = sum(1 for r in results if r["risk_correct"])
    accuracy = correct_count / n

    by_level = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_level[r["ground_truth_risk"]]["total"] += 1
        if r["risk_correct"]:
            by_level[r["ground_truth_risk"]]["correct"] += 1

    high_total = by_level["高"]["total"]
    high_correct = by_level["高"]["correct"]
    high_recall = high_correct / high_total if high_total > 0 else None
    unparseable = sum(1 for r in results if r["predicted_risk"] is None)
    avg_time = sum(r["elapsed_seconds"] for r in results) / n

    scored = [r for r in results if r.get("advice_score")]
    advice_metrics = {}
    if scored:
        for dim in ["处置方向", "关键信息", "安全性", "总评"]:
            advice_metrics[dim] = round(
                sum(r["advice_score"].get(dim, 0) for r in scored) / len(scored), 2
            )

    report = {
        "meta": {"agent_name": agent_name, "timestamp": datetime.now().isoformat(), "total_samples": n},
        "metrics": {
            "overall_accuracy": round(accuracy, 4),
            "high_risk_recall": round(high_recall, 4) if high_recall is not None else None,
            "unparseable_rate": round(unparseable / n, 4),
            "avg_elapsed_seconds": round(avg_time, 2),
            "advice_quality": advice_metrics,
        },
        "by_risk_level": {
            level: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else None,
            }
            for level, v in by_level.items()
        },
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n结果: 准确率={accuracy:.1%}  高风险召回={high_recall:.1%}  无法解析={unparseable/n:.1%}")
    if advice_metrics:
        print(f"建议质量: 处置方向={advice_metrics.get('处置方向',0):.1f}  "
              f"关键信息={advice_metrics.get('关键信息',0):.1f}  "
              f"总评={advice_metrics.get('总评',0):.1f}/10")
    print(f"已保存至: {output_path}")
    return report


# ── 入口 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行对照组实验")
    parser.add_argument("--testset",         default="data/testset_labeled_llm.jsonl")
    parser.add_argument("--output",          default="results/")
    parser.add_argument("--api-key",         required=True)
    parser.add_argument("--max",             type=int, default=50)
    parser.add_argument("--group",           choices=["A", "B", "all"], default="all")
    parser.add_argument("--no-advice-score", action="store_true", help="跳过建议质量打分")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")
    score_client = None if args.no_advice_score else client

    with open(args.testset, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f][:args.max]

    output_dir = Path(args.output)

    if args.group in ("A", "all"):
        evaluate_agent(PlainLLMAgent(client), "A_PlainLLM", samples,
                       str(output_dir / "eval_report_A_plainllm.json"), score_client)

    if args.group in ("B", "all"):
        evaluate_agent(LiteAgent(client), "B_LiteAgent", samples,
                       str(output_dir / "eval_report_B_liteagent.json"), score_client)

    print("\n✅ 对照组实验完成")
    print("   运行 python scripts/compare.py --markdown 查看对比结果")