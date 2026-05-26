"""
build_testset.py
================
将原始华佗数据集 (questions + answers) 转换为带 ground truth 的评估测试集。

输出格式（每行一个 JSON）：
{
  "id": 0,
  "symptoms": "患者原始描述",
  "doctor_answer": "医生原始回答",
  "ground_truth_risk": "高|中|低",
  "ground_truth_advice_keywords": ["关键词1", "关键词2"],
  "label_source": "rule|llm"   ← 标注来源，rule=关键词规则，llm=LLM判断
}

策略说明：
  1. 先用关键词规则快速标注（准确且免费）
  2. 规则覆盖不到的样本，用 LLM 补充标注（可选，需要 API key）
  3. 默认只用规则标注，确保可复现
"""

import json
import re
import argparse
from pathlib import Path
from typing import Optional

# ── 规则关键词 ────────────────────────────────────────────────────────────────

HIGH_RISK_KEYWORDS = [
    "立即就医", "马上就医", "急诊", "急救", "120",
    "手术", "住院", "紧急", "危及生命", "抢救",
    "心肌梗死", "脑梗", "脑出血", "休克", "窒息",
    "大量出血", "严重感染", "脓毒", "器官衰竭",
]

MEDIUM_RISK_KEYWORDS = [
    "尽快就医", "及时就医", "建议就医", "建议到医院",
    "去医院", "去三甲", "去门诊", "就诊", "检查治疗",
    "做检查", "做化验", "做B超", "做CT", "做血常规",
    "专科医院", "皮肤科", "妇科", "内科", "外科",
    "需要治疗", "需要用药", "需要检查",
]

LOW_RISK_KEYWORDS = [
    "注意休息", "多休息", "在家观察", "自行观察",
    "多喝水", "清淡饮食", "不用担心", "无需担心",
    "可以自行", "一般不严重", "常见情况", "无大碍",
    "正常现象", "可以自愈", "不必担心", "暂时观察",
]


def rule_label(answer: str) -> Optional[str]:
    """
    用关键词规则给答案打风险标签。
    返回 '高'/'中'/'低'，或 None（规则无法判断）。
    优先级：高 > 中 > 低
    """
    for kw in HIGH_RISK_KEYWORDS:
        if kw in answer:
            return "高"
    for kw in MEDIUM_RISK_KEYWORDS:
        if kw in answer:
            return "中"
    for kw in LOW_RISK_KEYWORDS:
        if kw in answer:
            return "低"
    return None


def extract_advice_keywords(answer: str) -> list:
    """
    从医生答案里提取关键建议词，用于后续软匹配评估。
    """
    keywords = []
    # 就医类
    for kw in ["就医", "医院", "急诊", "手术", "住院"]:
        if kw in answer:
            keywords.append(kw)
    # 生活类
    for kw in ["休息", "饮食", "多喝水", "运动", "忌口"]:
        if kw in answer:
            keywords.append(kw)
    # 用药类
    if re.search(r'(用药|服药|吃药|抗生素|消炎|止痛)', answer):
        keywords.append("用药")
    return list(set(keywords))


def build_testset(
    input_path: str,
    output_path: str,
    max_samples: int = 500,
    require_label: bool = True,
    use_llm_fallback: bool = False,
    api_key: str = None,
):
    """
    主函数：构建测试集。

    参数：
      input_path      原始 JSONL 文件路径
      output_path     输出 JSONL 文件路径
      max_samples     最多输出多少条（建议先用 100 条调试）
      require_label   True=只保留能打上标签的样本；False=没标签的也保留（标为'未知'）
      use_llm_fallback 是否对无法规则标注的样本调用 LLM 打标
      api_key         DeepSeek API key（use_llm_fallback=True 时需要）
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果需要 LLM fallback，初始化客户端
    llm_client = None
    if use_llm_fallback and api_key:
        from openai import OpenAI
        llm_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        print("LLM fallback 已启用")

    stats = {"高": 0, "中": 0, "低": 0, "未知": 0, "llm_used": 0}
    written = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin):
            if written >= max_samples:
                break

            raw = json.loads(line.strip())
            question = raw.get("questions", "").strip()
            answer = raw.get("answers", "").strip()

            if not question or not answer:
                continue

            # Step 1：规则标注
            risk_level = rule_label(answer)
            label_source = "rule"

            # Step 2：LLM 重新标注（覆盖规则结果）
            if llm_client is not None:
                risk_level = llm_label(llm_client, question, answer)
                label_source = "llm"

            # Step 3：过滤
            if require_label and risk_level is None:
                continue

            if risk_level is None:
                risk_level = "未知"

            stats[risk_level] = stats.get(risk_level, 0) + 1

            record = {
                "id": idx,
                "symptoms": question,
                "doctor_answer": answer,
                "ground_truth_risk": risk_level,
                "ground_truth_advice_keywords": extract_advice_keywords(answer),
                "label_source": label_source,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n✅ 测试集构建完成")
    print(f"   输出路径: {output_path}")
    print(f"   总样本数: {written}")
    print(f"   风险分布: 高={stats['高']} 中={stats['中']} 低={stats['低']} 未知={stats.get('未知',0)}")
    if use_llm_fallback:
        print(f"   LLM 打标: {stats['llm_used']} 条")
    return written


def llm_label(client, question: str, answer: str) -> Optional[str]:
    """
    用 LLM 对无法规则标注的样本打风险标签。
    返回 '高'/'中'/'低'，或 None。
    """
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"""你是医疗分诊专家。根据以下患者问题和医生回答，判断患者**当前**的风险等级。

注意：判断的是患者现在是否需要紧急处理，不是医生回答里提到了什么疾病。

患者问题：{question[:200]}
医生回答：{answer[:300]}

判断标准：
高（患者当前症状需要急诊或立即就医，2小时内处理）
中（建议患者尽快就医检查，但不紧急）
低（患者可在家观察或自行处理）

以下情况判为低风险：
- 症状已经好转或消失
- 慢性病咨询、复查需求
- 知识性问答（患者在问"是什么"而不是"我现在怎么办"）
- 普通皮肤病、减肥、美容问题

只回答一个词：高、中、或低"""
            }],
            temperature=0.0,
            max_tokens=10,
        )
        result = resp.choices[0].message.content.strip()
        if "高" in result:
            return "高"
        elif "中" in result:
            return "中"
        elif "低" in result:
            return "低"
    except Exception as e:
        print(f"LLM 打标失败: {e}")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 MedReAct 评估测试集")
    parser.add_argument("--input",  default="data/test_datasets.jsonl", help="原始数据路径")
    parser.add_argument("--output", default="data/testset_labeled.jsonl", help="输出路径")
    parser.add_argument("--max",    type=int, default=200, help="最大样本数")
    parser.add_argument("--keep-unlabeled", action="store_true", help="保留无法打标的样本")
    parser.add_argument("--llm-fallback",   action="store_true", help="对无法规则标注的样本用LLM打标")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key（llm-fallback时需要）")
    args = parser.parse_args()

    build_testset(
        input_path=args.input,
        output_path=args.output,
        max_samples=args.max,
        require_label=not args.keep_unlabeled,
        use_llm_fallback=args.llm_fallback,
        api_key=args.api_key,
    )