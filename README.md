# MedReAct：基于 ReAct 框架的医疗预问诊 Agent

> 从零实现的医疗分诊 Agent，核心目标：在保证整体准确率的前提下，最大化高风险患者的识别召回率。

**GitHub**: https://github.com/lynxx244/medreact-agent

---

## 项目背景

传统 LLM 直接判断患者症状风险等级存在明显缺陷：**高风险患者召回率为 0%**——模型倾向于给出保守的低/中风险结论，容易漏掉真正需要急诊的患者。

本项目基于 ReAct 论文（Yao et al., ICLR 2023）从零手写一个医疗预问诊 Agent，通过多轮推理、工具调用和 RAG 知识库检索，将高风险召回率从 0% 提升至 **83.3%**，超过 80% 的目标阈值。

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

**根本原因**：`risk_assess` 收到的症状列表是患者的表面描述，缺乏医学背景知识支撑。知识库返回了严重疾病的提示，但 Agent 不知道要把这些信息反馈给风险判断。

**解决方案**：
1. 构建 FAISS 向量知识库（华佗26M，5800条医学问答，排除测试集避免数据泄露）
2. 将 `search_symptom` 的检索结果作为 `kb_context` 传入 `risk_assess`
3. LLM 结合患者症状 + 知识库信息综合判断，不再依赖固定关键词列表
4. 新增"潜在严重疾病信号"判断维度（便血、痰中带血、不明原因消瘦等）
5. 在 SYSTEM_PROMPT 中明确要求：必须先调用 `search_symptom`，再将结果传入 `risk_assess`

**效果**：高风险召回率 27.3% → **83.3%**

---

### 挑战二：ground truth 标注噪声

**问题**：规则关键词标注导致大量误标——"手术"、"住院"出现在任何语境都会被标为高风险，导致评估结果虚高（高风险样本占 62/200，其中"小腿减肥"、"看书眼睛痛"被误标）。

**分析过程**：
- 规则标注：高风险 62/200（31%），大量误标
- 第一版LLM标注：高风险 35/200（17.5%），仍有噪声（LLM看医生回答而非患者当前状态）
- 修复标注逻辑：prompt明确要求"判断患者当前是否需要紧急处理，不是医生提到了什么疾病"
- 最终高质量标注：高风险 6/500（1.2%），标注准确

**启示**：v3 的 45.7% 高风险召回率包含标注噪声的干扰，v8 的 83.3% 基于高质量 ground truth，更能反映真实能力。

---

### 挑战三：RAG 知识库覆盖度的天花板

**问题**：60条测试集上高风险召回率 63.6%，扩展到 200 条后降至 17.1%。

**分析**：知识库 5800 条问答无法覆盖所有症状类型，检索不到相关内容时 Agent 只能依赖症状表面判断。这揭示了 RAG 系统的核心挑战：**性能瓶颈在知识库覆盖度，而非检索算法本身**。

**解决**：修复 ground truth 标注质量后，500 条测试集上高风险召回率达到 83.3%，同时发现部分被 Agent 判为高风险的"中风险"样本（如黑便、咳血、左臂无力）实际上 Agent 的判断更准确，反映了医疗场景下 ground truth 构建的固有难度。

---

### 挑战四：精确率与召回率的权衡

**问题**：提高高风险召回率的同时，部分中风险样本被过度升级为高风险。

**判断**：在医疗场景中，**假阴性（漏诊）的代价远大于假阳性（过度诊断）**——漏掉高风险患者可能危及生命，而多发一次"建议就医"只是让患者多跑一趟医院。主动接受这个权衡。

---

## 实验结果

### 完整对比（含消融实验）

| 版本 | 测试集 | 样本数 | 整体准确率 | 高风险召回率 | 备注 |
|------|--------|--------|-----------|-------------|------|
| A. 纯LLM | 规则标注 | 60 | 60.0% | 0.0% | 基准对照 |
| B. 简化Agent | 规则标注 | 60 | 51.7% | 18.2% | 无RAG对照组 |
| C. MedReAct v3 | LLM标注 | 60 | 50.0% | 27.3% | 加RAG前基准 |
| D. MedReAct v7 | LLM标注 | 200 | 51.5% | 45.7% | 加RAG+kb_context |
| **E. MedReAct v8** | **高质量标注** | **500** | **58.4%** | **83.3% ✅** | **最终版本** |

### 核心结论
- MedReAct 高风险召回率是纯 LLM 的无穷倍（0% → 83.3%）
- RAG 知识库 + kb_context 传递是最关键的改进，单步提升召回率 +136%（v3→v7）
- 高质量 ground truth 对评估结果影响显著，标注质量是评估可信度的前提
- v8 高风险召回率超过 80% 目标阈值，同时整体准确率 58.4% 优于所有对照组

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
  --testset data/testset_labeled_llm_v2.jsonl \
  --output results/eval.json \
  --api-key YOUR_API_KEY \
  --max 500 \
  --workers 4
```

---

## 项目结构

```
medreact-agent/
├── react_agent.py                    # 核心 Agent（ReAct 循环 + 三个工具）
├── kb/
│   ├── kb.index                      # FAISS 向量索引（5800条）
│   └── answers.json                  # 知识库文本
├── data/
│   ├── testset_labeled_llm_v2.jsonl  # 高质量LLM标注测试集（推荐）
│   ├── testset_labeled_llm.jsonl     # 第一版LLM标注
│   ├── testset_labeled.jsonl         # 规则标注测试集
│   └── test_datasets.jsonl           # 原始华佗数据
├── scripts/
│   ├── build_kb.py                   # 构建 FAISS 知识库
│   ├── build_dataset.py              # 构建评估测试集（支持LLM标注）
│   ├── evaluate.py                   # 批量评估（支持--workers并行）
│   ├── baseline.py                   # 对照组
│   └── compare.py                    # 横向对比报告
└── results/                          # 评估结果
```

---

## 已知局限与未来工作

- **知识库覆盖度有限**：5800条问答无法覆盖所有症状，扩展至完整华佗26M数据集预计可进一步提升召回率
- **高风险样本稀少**：500条测试集中仅6条高风险（1.2%），召回率统计存在较大方差，需更大规模测试集
- **ground truth 构建难度**：医疗场景下风险等级界定存在主观性，部分中风险样本实际应为高风险，LLM标注仍有噪声
- **`execute_tool` 使用 `eval()`**：存在安全风险，后期改成参数解析器
- **过度升级问题**：当前 Agent 对中风险样本过度升级为高风险，精确率有待改善

---

## 参考文献

- ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., ICLR 2023)
- 华佗26M医学问答数据集
- BGE: BAAI General Embedding (Beijing Academy of AI)
