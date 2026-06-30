# MedReAct：基于 ReAct 框架的医疗预问诊 Agent

> 从零实现的中文医疗分诊 Agent，核心目标：在保证整体准确率的前提下，最大化高风险患者的识别召回率。

**GitHub**: https://github.com/lynxx244/medreact-agent

---

## 核心指标（v3 双源架构，最终版本）

| 指标 | 数值 | 说明 |
|---|---|---|
| 高风险召回率 | **81.6%** | 158条高风险样本，核心安全指标 |
| 过度分诊率 | 21.5% | 非高风险样本中被误判为高风险的比例 |
| 解析失败率 | **0.2%** | 433条中仅1条格式异常，接近生产级水平 |
| 总体准确率 | 67.8% | 高/中/低三分类 |
| 测试集规模 | 433条 | 高/中/低 = 158/141/134，分布均衡可信 |

> 高风险召回率 81.6% 超过 arXiv:2605.15680 提出的 **75% 临床可部署门槛**，优于同类英文系统 TriageAgent（78.3%），且在中文这一更困难的语言环境下实现。

---

## 项目背景

传统 LLM 直接判断患者症状风险等级存在明显缺陷：**高风险患者召回率只有 60%**——模型倾向于给出保守的低/中风险结论，容易漏掉真正需要急诊的患者。

本项目基于 ReAct 论文（Yao et al., ICLR 2023）从零手写医疗预问诊 Agent，通过多轮推理、工具调用和双源 RAG 知识库检索，将高风险召回率从接近 0% 提升至 **81.6%**，同时解析失败率控制在 0.2%。

---

## 系统架构

```
患者描述
    ↓
┌──────────────────────────────────────────────┐
│                MedReAct Agent                │
│                                              │
│    Thought → Action → Observation            │
│         ↑__________________↓                 │
│                                              │
│  工具1: ask_patient     （多轮追问症状细节）   │
│  工具2: search_guideline（规则型知识库检索）   │
│  工具3: search_symptom  （经验型知识库检索）   │
│  工具4: risk_assess     （双源融合风险评估）   │
│                                              │
│  安全二次校验（中/低风险结论单向升级兜底）      │
└──────────────────────────────────────────────┘
    ↓
Final Answer
风险等级：高/中/低
可能原因：xxx
建议行动：xxx
注意事项：xxx
```
<img width="980" height="783" alt="image" src="https://github.com/user-attachments/assets/712d3c10-3257-4e9d-8cea-eeaf6eb48281" />


### 检索 Pipeline

```
Query（患者症状描述）
  │
  ├── BM25 检索（jieba 分词，精确关键词匹配）→ top20 候选
  │
  └── 向量检索（bge-base-zh-v1.5 + FAISS）  → top20 候选
  │
  └── RRF 融合（k=60，Cormack et al. SIGIR 2009）→ top10
  │
  └── BGE Reranker v2-m3 精排（CrossEncoder）→ top3 送入 LLM
```

**技术栈**：DeepSeek API · FAISS · BGE-base-zh-v1.5 · BGE-Reranker-v2-m3 · 华佗26M医学数据集 · rank-bm25 · jieba

---

## 双源知识库

| 知识库 | 内容 | 规模 | 检索工具 |
|---|---|---|---|
| 规则型（triage_rules.json） | 医学分诊规则，含症状描述、风险等级、判断依据 | 5257 条 | search_guideline |
| 经验型（answers.json） | 华佗26M 医疗问答采样 | 26M条采样子集 | search_symptom |

两路知识在 `risk_assess` 中融合，规则型优先级高于经验型（若两者冲突，以分诊规则为主）。

设计依据：MECR-RAG（JMIR Medical Informatics 2026）双源融合架构。

---

## 核心技术细节

### 1. RRF 融合检索

v3 将线性加权融合升级为 **RRF（Reciprocal Rank Fusion）**：

$$\text{score}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d)}, \quad k=60$$

**为什么换 RRF**：BM25 分数无上界（整数），余弦相似度在 [-1,1]，两者尺度完全不同，线性加权需要手动调权重超参且跨数据集不稳定。RRF 只操作排名位置，完全绕开尺度不一致问题，k=60 是原论文推荐的跨数据集鲁棒默认值。

依据：Cormack, Clarke & Büttcher, SIGIR 2009。

### 2. BGE Reranker v2-m3 精排

RRF top10 → BGE Reranker v2-m3 → top3。

CrossEncoder 架构让 query 和 doc 互相 attend，精排质量显著优于双塔向量模型。Reranker 不改变 Recall@10（候选池不变），但显著提升 MRR（将相关文档排到更靠前的位置）。RAGalyst 实验数据：Hybrid 检索加 Reranker 后 MRR 从 0.531 提升至 0.614（+15.7%）。

依据：BAAI 官方选型建议（中英文场景推荐 bge-reranker-v2-m3）；RAGalyst（arXiv:2511.04502）消融实验。

### 3. 安全偏置设计

系统采用 **asymmetric error cost** 设计哲学：

- `risk_assess` prompt 内置保守偏置："宁可误判为高风险，也不要漏掉潜在严重疾病"
- 对中/低风险结论触发安全二次校验（单向升级，不降级）
- 代价：过度分诊率 21.5%；收益：高风险召回率 81.6%

依据：arXiv:2605.15680，"漏诊高危病例的危害远大于保守性过度分诊，评估必须专门衡量高危召回率而非仅看 F1"。

### 4. 测试集构建

- 原始平衡测试集目标：高/中/低 = 150/150/200
- 过滤 67 条非症状描述样本（如"小腿减肥"、"看书眼睛痛"），最终 433 条
- 实际分布：高风险 158（36.5%）/ 中风险 141 / 低风险 134
- 高风险标签采用 LLM 重新标注，修正了规则标注的系统性误标问题（规则标注会将任何含"手术"、"住院"字样的样本标为高风险，导致大量误标）

---

## 实验结果

### 消融实验（完整演进路径）

| 版本 | 核心改动 | 测试集 | 高风险召回率 | 总体准确率 | 备注 |
|---|---|---|---|---|---|
| A. 纯 LLM（无 Agent） | — | 规则标注，60条 | ~0% | 60.0% | 基准对照 |
| B. 简化 Agent（无 RAG） | ReAct 框架 | 规则标注，60条 | 18.2% | 51.7% | 无知识库对照 |
| C. MedReAct v3（单源） | 加 FAISS 向量检索 | LLM标注，60条 | 27.3% | 50.0% | 加 RAG 前基准 |
| D. MedReAct v7（单源） | kb_context 注入 risk_assess | LLM标注，200条 | 45.7% | 51.5% | 关键改进点 |
| E. MedReAct v8（单源） | 高质量 ground truth | 高质量标注，500条 | 83.3% | 58.4% | 单源最终版 |
| **F. MedReAct v3（双源，当前）** | **RRF + 双源知识库 + BGE Reranker** | **均衡标注，433条** | **81.6%** | **67.8%** | **架构全面升级** |

> **关于 v8（83.3%）与当前版本（81.6%）的差异**：两者测试集构成完全不同。v8 测试集高风险样本仅 6 条，统计方差极大（±一两条就能改变 15-20 个百分点）；当前版本高风险样本 158 条，统计结论更可信。架构升级（双源知识库 + RRF + Reranker）同时带来了总体准确率的显著提升（58.4% → 67.8%）。

### 与文献对比

| 系统 | 高风险召回率 | 过度分诊率 | 语言 | 测试集规模 |
|---|---|---|---|---|
| **MedReAct v3（本项目）** | **81.6%** | 21.5% | 中文 | 433条，158条高风险 |
| TriageAgent (arXiv:2605.15680) | 78.3% | 18.6% | 英文 | 1200条 |
| MECR-RAG v2 (JMIR 2026) | >80%（F1=0.79） | **12.7%** | 英文 | 500条 |
| MECR-RAG v1 (JMIR 2026) | — | 28.8% | 英文 | 500条 |
| 纯 LLM baseline (DeepSeek-Chat) | ~0% | — | 中文 | — |
| 人类护士分诊（JMIR 2026参考） | — | 8–15% | — | 金标准 |

### 混淆矩阵

```
真实\预测    高风险    中风险    低风险
─────────────────────────────────────
高风险        129       20        9
中风险         49       71       20
低风险         10       31       93
```

中风险准确率（50.7%）是主要短板，与领域内普遍结论一致：中等严重程度样本是所有分诊系统的共同难点，其临床特征与高/低风险均有重叠（MECR-RAG 论文同样指出此问题）。

---

## 快速启动

### 环境准备

```bash
pip install faiss-cpu sentence-transformers openai rank-bm25 jieba
```

### 构建经验型知识库

```bash
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
python scripts/build_kb.py

# Linux/Mac
HF_ENDPOINT="https://hf-mirror.com" python scripts/build_kb.py
```

### 构建规则型索引

```bash
python rule_retriever.py --build
```

### 交互模式

```python
from react_agent import MedReActAgent

agent = MedReActAgent(max_steps=10)
agent.run("我突然胸口疼，出了很多汗")
```

### 批量评估

```bash
python react_agent.py --eval
```

---

## 项目结构

```
medreact-agent/
├── react_agent.py                   # 核心 Agent（ReAct 循环 + 四个工具 + 评估模块）
├── rule_retriever.py                # 规则型知识库检索器（Hybrid RRF）
├── kb/
│   ├── kb.index                     # FAISS 向量索引（经验型，华佗26M采样）
│   ├── answers.json                 # 经验型知识库文本
│   └── triage_rules.json            # 规则型知识库（5257条）
├── rules.faiss                      # 规则库向量索引（预建）
├── rules_meta.pkl                   # 规则库元数据
├── rules_bm25.pkl                   # 规则库 BM25 索引
├── data/
│   ├── test_samples_filtered.json   # 最终测试集（433条，均衡分布）
│   └── test_samples.json            # 原始测试集（500条，含过滤前）
├── scripts/
│   ├── build_kb.py                  # 构建经验型 FAISS 知识库
│   ├── build_dataset.py             # 构建评估测试集
│   ├── evaluate.py                  # 批量评估
│   ├── baseline.py                  # 纯 LLM 对照组
│   └── compare.py                   # 横向对比报告
└── results/
    └── evaluation_report_final.json # 最终评估结果
```

---

## 评估指标说明

| 指标 | 定义 | 文献依据 |
|---|---|---|
| 高风险召回率 | 真实高风险样本中被正确识别为高风险的比例 | arXiv:2605.15680 |
| 过度分诊率 | 真实非高风险样本中被误判为高风险的比例 | MECR-RAG, JMIR 2026 |
| 解析失败率 | Final Answer 格式不合规的比例 | 工程鲁棒性指标 |
| Recall@10 | RRF top10 中包含相关文档的比例（检索天花板） | RAGalyst, arXiv:2511.04502 |
| MRR@3 | Reranker 精排后第一个相关文档的倒数排名均值 | RAGalyst, arXiv:2511.04502 |

---

## 已知局限与未来工作

**当前局限**：

- 过度分诊率 21.5% 偏高，根因是安全校验为单向升级设计，缺少规则约束的降级逻辑（参考 MECR-RAG v2 的双向校验策略，将过度分诊率压至 12.7%）
- 中风险准确率 50.7% 是主要短板，中等严重程度样本的高/中边界模糊
- 检索层 Recall@10 和 MRR@3 尚未计算（测试集缺少 relevant_doc_indices 标注）
- `parse_output` 依赖固定字符串格式，生产环境应改为结构化输出（structured output）

**后续改进方向**：

- 为安全校验增加双向逻辑，目标将过度分诊率降至 15% 以下
- 构建中风险 few-shot 示例集，改善高/中边界判断精度
- 对 BM25 top1 做弱监督标注，补全 Recall@10 和 MRR@3 评估
- 扩展规则库，接入 CMeKG 完整数据集

---

## 参考文献

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023
- Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009
- *MECR-RAG: Multi-Evidence Clinical Reasoning RAG for Medical Triage*, JMIR Medical Informatics 2026
- *TriageAgent: LLM-based Emergency Triage with RAG*, arXiv:2605.15680
- *RAGalyst: Benchmarking RAG Pipelines for Clinical QA*, arXiv:2511.04502
- *BRIGHT+: Beyond Relevance in Information Retrieval*, arXiv:2506.07116
- BAAI, *BGE Reranker v2-m3 Technical Report*, 2024
- 华佗26M医学问答数据集（Huatuo-26M）
