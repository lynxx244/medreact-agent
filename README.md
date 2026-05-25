# MedReAct：基于 ReAct 框架的医疗预问诊 Agent

> 从零实现的医疗分诊 Agent，核心目标：在保证整体准确率的前提下，最大化高风险患者的识别召回率。

**GitHub**: https://github.com/lynxx244/medreact-agent

---

## 项目背景

传统 LLM 直接判断患者症状风险等级存在明显缺陷：**高风险患者召回率为 0%**——模型倾向于给出保守的低/中风险结论，容易漏掉真正需要急诊的患者。

本项目基于 ReAct 论文（Yao et al., 2022）从零手写一个医疗预问诊 Agent，通过多轮推理、工具调用和 RAG 知识库检索，将高风险召回率从 0% 提升至 **63.6%**。

---

## 系统架构

```
患者描述
    ↓
┌─────────────────────────────────┐
│         MedReAct Agent          │
│                                 │
│  Thought → Action → Observation │
│       ↑________________↓        │
│                                 │
│  工具1: ask_patient（追问症状）   │
│  工具2: search_symptom（RAG检索）│
│  工具3: risk_assess（风险评估）  │
│                                 │
│  安全二次校验（低/中风险兜底）    │
└─────────────────────────────────┘
    ↓
Final Answer（风险等级 + 建议）
```

**技术栈**：DeepSeek API + FAISS + BGE-base-zh-v1.5 + 华佗26M医学数据集

---

## 核心技术挑战与解决思路

### 挑战一：高风险患者大量漏诊

**问题**：基于关键词的风险评估只能识别急性红旗症状（休克、意识丧失等），无法识别"黑便可能消化道出血"、"肩背痛可能肺癌转移"等潜在严重疾病信号。

**分析**：`risk_assess` 收到的症状列表是患者的表面描述，缺乏医学背景知识的支撑。

**解决**：
1. 构建 FAISS 向量知识库（华佗26M数据集，6000条医学问答，排除测试集避免数据泄露）
2. 将 `search_symptom` 的检索结果作为 `kb_context` 传入 `risk_assess`
3. LLM 结合患者症状 + 知识库信息综合判断，不再依赖固定关键词列表
4. 新增"潜在严重疾病信号"判断维度（便血、痰中带血、不明原因消瘦等）

**效果**：高风险召回率从 27.3% → **63.6%**（+136%）

---

### 挑战二：ground truth 标注噪声

**问题**：规则关键词标注导致大量误标——"手术"、"住院"出现在任何语境（历史手术、慢性病复诊）都会被标为高风险，导致评估结果虚高。

**发现**：规则标注下高风险样本占 62/200（31%），其中"小腿减肥"、"看书眼睛痛"被误标为高风险。

**解决**：改用 LLM 标注，高风险样本降至 35/200（17.5%），分布更符合真实医疗场景。

**启示**：v1 的 45.5% 高风险召回率是虚高的，基于准确 ground truth 的 v3 基准为 27.3%，v5 的 63.6% 才是真实的改进效果。

---

### 挑战三：医疗场景的精确率/召回率权衡

**问题**：提高高风险召回率的同时，部分中风险样本被过度升级为高风险（中风险准确率从 29.4% 降至 11.8%）。

**判断**：在医疗场景中，**假阴性（漏诊）的代价远大于假阳性（过度诊断）**——漏掉高风险患者可能危及生命，而多发一次"建议就医"只是让患者多跑一趟医院。因此主动接受这个权衡。

---

## 实验结果

### 消融实验（基于 LLM 标注的60条测试集）

| 指标 | A. 纯LLM | B. 简化Agent | C. MedReAct v3(基准) | D. MedReAct v5(最终) |
|------|----------|-------------|---------------------|---------------------|
| 整体准确率 | 60.0% | 51.7% | 50.0% | **61.7%** |
| **高风险召回率** | **0.0%** | **18.2%** | **27.3%** | **63.6%** |
| 建议覆盖率 | - | - | 66.9% | 67.8% |
| 无法解析率 | 0.0% | 0.0% | 0.0% | 0.0% |
| 平均耗时 | 1.8s | 3.5s | 12.1s | 13.7s |

**核心结论**：
- MedReAct 高风险召回率是纯 LLM 的无穷倍（0% → 63.6%）
- RAG 知识库接入是最关键的改进，单步提升召回率 +136%
- 整体准确率优于所有对照组

---

## 快速启动

### 环境准备

```bash
pip install faiss-cpu sentence-transformers openai
```

### 构建知识库

```bash
# Windows
$env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/build_kb.py

# Linux/Mac
HF_ENDPOINT="https://hf-mirror.com" python scripts/build_kb.py
```

### 运行 Agent

```python
from react_agent import MedReActAgent
agent = MedReActAgent(max_steps=8)
agent.run("我头痛发烧两天了")
```

### 运行评估

```bash
python scripts/evaluate.py \
  --testset data/testset_labeled_llm.jsonl \
  --output results/eval.json \
  --api-key YOUR_API_KEY \
  --max 60
```

---

## 项目结构

```
medreact-agent/
├── react_agent.py          # 核心 Agent（ReAct 循环 + 三个工具）
├── kb/
│   ├── kb.index            # FAISS 向量索引
│   └── answers.json        # 知识库文本
├── data/
│   ├── testset_labeled_llm.jsonl   # LLM 标注测试集（推荐）
│   └── test_datasets.jsonl         # 原始华佗数据
├── scripts/
│   ├── build_kb.py         # 构建 FAISS 知识库
│   ├── build_dataset.py    # 构建评估测试集
│   ├── evaluate.py         # 批量评估
│   ├── baseline.py         # 对照组
│   └── compare.py          # 横向对比报告
└── results/                # 评估结果
```

---

## 已知局限与未来工作

- **高风险召回率 63.6%**，距目标 80% 仍有差距，需要更大规模知识库
- **知识库覆盖度有限**，6000条问答无法覆盖所有症状类型
- **`execute_tool` 使用 `eval()`**，存在安全风险，后期改成参数解析器
- **测试集仅60条**，样本量偏少，统计显著性有限
- ground truth 基于 LLM 标注，存在一定噪声

---

## 参考文献

- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., ICLR 2023)
- 华佗26M医学问答数据集
- BGE: BAAI General Embedding (Beijing Academy of AI)
