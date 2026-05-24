# MedReAct：基于 ReAct 框架的医疗预问诊 Agent

基于 [ReAct 论文](https://arxiv.org/abs/2210.03629) 从零实现的医疗预问诊 Agent，使用 DeepSeek API，配套完整评估体系与消融实验。

---

## 项目结构

```
mdreact_eval/
├── react_agent.py          # MedReAct Agent 核心实现
├── data/
│   └── testset_labeled_llm.jsonl   # 评估测试集（规则+LLM双重标注）
├── scripts/
│   ├── build_dataset.py    # 数据预处理：原始数据 → 带标签测试集
│   ├── evaluate.py         # 评估核心：批量测试 Agent，计算指标
│   ├── baseline.py         # 消融实验：纯LLM、简化Agent 两个对照组
│   ├── compare.py          # 生成三组横向对比报告
│   └── report.py           # 可视化单组评估报告
└── results/                # 评估报告输出目录（自动生成）
```

---

## Agent 设计

MedReAct 实现了标准 ReAct 循环：**Thought → Action → Observation → 循环或 Final Answer**

### 三个工具

| 工具 | 作用 |
|------|------|
| `ask_patient(question)` | 向患者追问症状细节，每次只问一个问题 |
| `risk_assess(symptoms, duration_days)` | 用 LLM 语义判断红旗症状，非关键词匹配 |
| `search_symptom(query)` | 查询症状相关医学背景知识 |

### 安全机制

Final Answer 输出前强制调用 `risk_assess`，并对低/中风险结论做二次安全校验，防止高风险漏判。

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API Key

在 `react_agent.py` 顶部填入你的 DeepSeek API key：

```python
client = OpenAI(
    api_key="your_deepseek_api_key",
    base_url="https://api.deepseek.com"
)
```

### 运行 Agent

```bash
python react_agent.py
```

---

## 评估体系

### Step 1：构建测试集

```bash
# 从原始华佗数据构建200条带标签测试集
python scripts/build_dataset.py \
    --input data/test_datasets.jsonl \
    --output data/testset_labeled_llm.jsonl \
    --max 200 \
    --llm-fallback \
    --api-key YOUR_KEY
```

### Step 2：评估 MedReAct

```bash
python scripts/evaluate.py \
    --testset data/testset_labeled_llm.jsonl \
    --output results/eval_report_llm.json \
    --api-key YOUR_KEY \
    --max 60
```

### Step 3：运行消融实验（对照组）

```bash
python scripts/baseline.py \
    --testset data/testset_labeled_llm.jsonl \
    --output results/ \
    --api-key YOUR_KEY \
    --max 60
```

### Step 4：生成对比报告

```bash
python scripts/compare.py --markdown
```

---

## 实验结果

### 消融实验对比

| 指标 | A. 纯LLM | B. 简化Agent | C. MedReAct |
|------|----------|-------------|-------------|
| 整体准确率 | 75.0% | 60.0% | 52.0% |
| **高风险召回率** | **0.0%** | **50.0%** | **50.0%** |
| 建议质量-处置方向 | 8.2/10 | 8.2/10 | 8.1/10 |
| 建议质量-安全性 | 9.1/10 | 9.0/10 | 9.0/10 |
| 建议质量-总评 | 8.1/10 | 8.1/10 | 8.1/10 |
| 无法解析率 | 0.0% | 0.0% | 0.0% |
| 平均耗时 | 1.8s | 3.7s | 12.8s |

### 核心发现

**1. ReAct 框架的价值在于安全兜底，而非内容质量提升**

三组建议质量评分相当（均为 8.1/10），说明 ReAct 的多轮推理没有提升单次建议内容质量。但纯 LLM 高风险召回率为 0%，MedReAct 通过安全二次校验达到 50%，在医疗场景中这是最关键的差异。

**2. 准确率受 ground truth 标注噪声影响**

基于关键词+LLM 的自动标注存在约 15-20% 噪声，部分案例中 Agent 判断反而比 ground truth 更合理（如术后持续出血+发热被标为中风险，Agent 判为高风险更符合临床实际）。

**3. 安全性与效率的权衡**

MedReAct 响应时间（12.8s）是纯 LLM（1.8s）的 7 倍，是安全校验和多轮推理的代价，适用于对安全性要求高于速度的场景。

### 迭代优化历程

| 版本 | 关键改动 | 无法解析率 |
|------|---------|-----------|
| v1 | 基础 ReAct 实现 | 30% |
| v2 | 强制 Final Answer 格式 | 0% |
| v3 | risk_assess 降低误触发敏感度 | 0% |
| v4 | 步数用完强制生成答案 | 0% |
| v5 | 安全二次校验机制 | 0% |
| v6 | LLM 解析器兜底 | 0% |

---

## 评估指标说明

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **高风险召回率** | 高风险案例中被正确识别的比例，漏判代价极大 | > 80% |
| 整体准确率 | 风险等级判断正确率，受标注噪声影响 | 参考值 |
| 建议质量总评 | LLM 对比医生医嘱打分（0-10），衡量建议内容质量 | > 7.0 |
| 无法解析率 | Agent 输出格式无法被解析的比例 | < 5% |

---

## 数据集

使用 [华佗26M](https://huggingface.co/datasets/FreedomIntelligence/huatuo26M-testdatasets) 测试集，包含真实医患问答对。

ground truth 标注策略：
- **规则标注**：关键词匹配医生答案，快速且可复现（覆盖约28%样本）
- **LLM标注**：对规则无法判断的样本用 DeepSeek 补充打标

---

## 技术债与后续方向

- [ ] `execute_tool` 使用了 `eval()`，存在安全风险，需改为参数解析器
- [ ] 高风险召回率仍有提升空间，可尝试更细粒度的红旗症状分类
- [ ] 当前 ground truth 质量有限，可构建人工标注的高质量小测试集
- [ ] Agent 无跨会话记忆，每次对话独立

---

## 依赖

- Python 3.8+
- openai >= 1.0.0
- DeepSeek API Key（[申请地址](https://platform.deepseek.com)）
