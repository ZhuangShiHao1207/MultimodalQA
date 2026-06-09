# MultimodalQA 实验改进方案

> 基于当前 `evaluation/results.json` 与 `EVALUATION_REPORT.md` 的全面审查，系统性列出所有需要改进、补充或修正的实验维度。每一条均包含**问题描述 → 原因分析 → 具体修改方案**。

---

## 目录

1. [🔴 P0｜循环验证（数据泄漏）问题](#1-p0循环验证数据泄漏问题)
2. [🔴 P0｜数据集规模与多样性不足](#2-p0数据集规模与多样性不足)
3. [🔴 P0｜缺少统计显著性检验](#3-p0缺少统计显著性检验)
4. [🟡 P1｜检索层从未独立评估](#4-p1检索层从未独立评估)
5. [🟡 P1｜消融实验维度单一](#5-p1消融实验维度单一)
6. [🟡 P1｜基线对比不公平](#6-p1基线对比不公平)
7. [🟢 P2｜完全缺失可视化](#7-p2完全缺失可视化)
8. [🟢 P2｜缺少延迟与成本分析](#8-p2缺少延迟与成本分析)
9. [🟢 P2｜题目难度分级缺失](#9-p2题目难度分级缺失)

---

## 1. 🔴 P0｜数据集构建流程问题（经 git 核查后的精确描述）

### 经代码考证的实际情况

通过 `git show 9e92a3e` 与 `git show 58ae3a2` 的 diff，可以精确还原此次修改的内容，分三类性质完全不同的变更：

---

#### 类型 A：Gold 事实错误（修正合理，无争议）

原始 QA 数据集（q13–q25）在首次写入时已包含多处客观事实错误，这些错误与模型输出无关，任何人读图都能核查：

| 题 | 原 gold（错误） | 修正后 | 核查依据 |
|---|---|---|---|
| q13 | `"RNN-Strong"` | `"TF-IDF+LR"` | 柱状图：LR=0.883 > RNN-Strong=0.837 |
| q17 | `"1087"` | `"1480"` | LR 混淆矩阵右上角直接读数 |
| q18 | `"positive"` | `"negative"` | Recall 计算：neg=88.4% > pos=86.2% |
| q19 | `"5765"` | `"10756"` | RNN BiLSTM 混淆矩阵右下角 |
| q20 | `"RNN-Strong"` | `"TF-IDF+LinearSVM"` | FP 对比：SVM=1456 最少 |
| q24 | `"没有过拟合"` | `"是"` | Val Loss 在 epoch 4 后单调上升 |
| q25 | `"RNN 准确率更高"` | `"传统方法"` | 文档第 7 节有明确说明 |

**这些属于原始数据集的笔误，修正是正确且必要的。** 这类修正与是否看过模型输出无关——因为这些 gold 在逻辑上就是错的。

---

#### 类型 B：扩充 Gold 候选集（无害）

q01、q11、q14、q16 的修改只是在原有正确 gold 基础上追加了更多等价表述（例如给 q01 加了"最小化簇内平方距离和"这一同义表达）。原有 gold 保持不变，仅让 ANLS 指标对长答案更宽容。**不影响正确性判断，无需处理。**

---

#### 类型 C：⚠️ 题目被整体替换（真正需要关注的问题）

**q08** 和 **q15** 的修改不是"修正 gold"，而是把原题整体换成了一道不同的题：

```diff
# q08：整题替换，且类型发生变化
- question: "从聚类可视化散点图来看，Ground Truth、Random和K-Means++的聚类效果有什么区别？"
- gold:     ["K-Means++的聚类结果更接近Ground Truth"]
- type:     figure
- requires_visual: true   ← 视觉题

+ question: "在GMM初始化方法对比中（Full协方差），Random和K-Means++哪种初始化得到的聚类结果更接近Ground Truth标签？"
+ gold:     ["Random"]
+ type:     table
+ requires_visual: false  ← 改成了非视觉题
```

```diff
# q15：问题意图改变（train loss 趋势 → val loss 最低点）
- question: "从RNN训练损失曲线来看，模型大约在第几个Epoch后Loss趋于平稳？"
- gold:     ["3", "第3个"]

+ question: "从RNN训练损失曲线来看，验证集Loss在第几个Epoch左右达到最低值？"
+ gold:     ["4", "第4个"]
```

**影响**：
- q08 原题（`requires_visual: true`）被替换为新题（`requires_visual: false`），视觉题数量从 12 道降至 11 道
- 原 q08 考察的是散点图聚类可视化理解，这类题目对多模态系统更有区分度，直接删除损失了评测覆盖面
- 原 q08 的 gold `"K-Means++"` 是否正确**从未被验证**，题目就被替换掉了

---

### 「循环验证」问题的实际范围（修正后的判断）

经过 git 核查，原来描述的"循环"范围比实际情况**更严重**——不仅是 gold 被修改，还有题目被整体替换。但同时"类型 A"的修正确实有事实依据支撑，这部分不应被归入"循环"。

**实际问题的精确描述**：

> 原始数据集（首次 commit `9e92a3e`）只有 12 道题（q01–q12），q13–q25 是在同一个"修正"commit（`58ae3a2`）里和模型答案、metrics 修改、rescore 脚本一起引入的。因此 **q13–q25 这 13 道题的 gold 从未经历过独立于模型输出的验证**——它们在被写入时就可能已经参考了模型的输出来确认正确性。

这是比单纯"改 gold"更根本的问题：**半数以上的题目（13/25）和最终评测结果在同一个 commit 里被同时引入**。

### 修改方案

#### 方案 A：回退原始 12 题，独立评估（最严格）

```bash
# 提取原始 12 题数据集
git show 9e92a3e:evaluation/datasets/self_built_qa.json > \
    evaluation/datasets/original_12_qa.json

# 基于原始 12 题单独重跑评测（不修改 gold，保留真实分数）
# 结果可能 MM < 100%，但是完全可信的
```

原始 12 题中有 2 道视觉题（q04、q11），但 q08 原题（散点图）也是视觉题，可以还原回来。

#### 方案 B：针对 q13–q25 补做独立核查（推荐，工作量小）

对 q13–q25 的每道题，不看 `results.json` 中的模型答案，直接打开原始 PDF，独立验证当前 gold 是否正确，填入核查记录表：

```markdown
| 题ID | 当前 gold | 来源 PDF 页码 | 核查截图 | 核查结论 | 核查人 |
|------|-----------|---------------|----------|----------|--------|
| q13  | TF-IDF+LR | 第3页柱状图   | [图]     | ✅ 正确   | XXX    |
| ...  |           |               |          |          |        |
```

完成后将核查记录与数据集一并存档，相当于补做了独立验证。

#### 方案 C：还原被替换的 q08 原题（建议）

将 q08 改回原始的散点图视觉题，并补充正确的 gold（打开 PDF 直接核查）：

```json
{
  "id": "q08",
  "question": "从聚类可视化散点图来看，K-Means++和Random初始化的聚类结果哪个更接近Ground Truth的分布？",
  "gold_answers": ["K-Means++", "K-Means++更接近", "kmeans++"],
  "type": "figure",
  "requires_visual": true,
  "page": 7,
  "note": "还原自原始数据集，gold 已独立核查"
}
```

> **注**：还原此题后需重新跑模型（1 次 API 调用），并如实记录结果，无论对错。

---

## 2. 🔴 P0｜数据集规模与多样性不足

### 问题描述

| 当前状态 | 问题 |
|---|---|
| 25 道题 | 置信区间极宽，±20% 以内的差异可能是噪声 |
| 2 篇 PDF | 高度同质化，都是 ML 课程实验报告 |
| 图片仅 6 张 | 视觉类型覆盖窄（折线图、柱状图、混淆矩阵）|
| 无跨领域文档 | 无财报/学术论文/产品手册/表单等 |

### 修改方案：引入外部多类别文档

以下是可直接获取、页数合适（≤20页）的文档来源，涵盖多种视觉类型：

---

#### 来源 1：DocVQA 单页文档（工业文档，已有配套 QA）

- **类型**：打印文件、备忘录、表单、信件（来自 UCSF 工业文档库）
- **视觉内容**：印章、手写注释、表格、签名
- **下载地址**：  
  - 数据集页面：https://www.docvqa.org/datasets/docvqa  
  - Hugging Face 子集（1200 样本，含 image + QA）：https://huggingface.co/datasets/nielsr/docvqa_1200_examples
- **使用方式**：每个样本即为单页图片 + QA，可直接用；若需 PDF 形式，用 `img2pdf` 打包 3–5 张同类页面

---

#### 来源 2：ChartQA 图表问答（图表理解，已有配套 QA）

- **类型**：各类统计图表（折线图、饼图、散点图、堆叠柱状图）
- **视觉内容**：图表读数、趋势、极值比较
- **下载地址**：  
  - Hugging Face：https://huggingface.co/datasets/docintel/ChartQA  
  - 原始仓库：https://github.com/vis-nlp/ChartQA  
- **使用方式**：直接加载 `split="test"` 的 2500 条，每条包含 `image`（图表）+ `question` + `answer`；用脚本按 chart_type 各取 5–8 条即可，页数≤1（纯图表）

---

#### 来源 3：MMLongBench-Doc 子集（长文档，过滤短文档使用）

- **类型**：学术论文、政府报告、财报、法律文件、教材（7 个领域）
- **视觉内容**：表格、图表、公式、跨页引用
- **下载地址**：  
  - Hugging Face：https://huggingface.co/datasets/yubo2333/MMLongBench-Doc  
  - GitHub：https://github.com/mayubo2333/MMLongBench-Doc  
- **使用方式（关键：过滤短文档）**：

```python
from datasets import load_dataset
ds = load_dataset("yubo2333/MMLongBench-Doc", split="test")

# 过滤页数 ≤ 15 页的文档
short_docs = [x for x in ds if x["num_pages"] <= 15]

# 按领域各取 5 道视觉相关题
by_domain = {}
for item in short_docs:
    domain = item["domain"]
    if domain not in by_domain:
        by_domain[domain] = []
    if len(by_domain[domain]) < 5 and item["evidence_type"] in ["figure", "table"]:
        by_domain[domain].append(item)
```

---

#### 来源 4：自建 PDF（最推荐用于本项目）

下载以下公开、页数短的 PDF 并手动构建 QA（各 5–8 道题）：

| 文档类型 | 来源 | 链接（示例） | 页数 |
|---|---|---|---|
| 上市公司年报摘要 | 腾讯/阿里巴巴财报亮点页 | 任意一家 A 股公司投关官网 → 年报摘要 | 6–10 页 |
| 学术会议 poster/论文 | arXiv | 搜索 CVPR/ICLR poster PDF | 1–2 页 |
| 政府/机构统计报告 | 国家统计局 | https://www.stats.gov.cn/sj/ → 统计摘要 PDF | 5–10 页 |
| 产品技术手册 | 各开源硬件/芯片厂商 | 如树莓派规格书，搜"Datasheet PDF" | 4–8 页 |
| 医疗/卫生报告 | WHO | https://www.who.int/publications → 摘要版 PDF | 8–15 页 |

**每类 PDF 手工出 5–8 道题，覆盖：精确数值读取 + 图表趋势判断 + 跨页推理，确保各类型约 1/3 为视觉必答题。**

---

#### 推荐最终数据集构成（≥100 道题）

| 来源 | 题数 | 文档类型 | 视觉题占比 |
|---|---|---|---|
| 原有 2 篇实验报告 | 25 | ML 报告 | 44% |
| DocVQA 子集（取 4 张图）| 20 | 工业文档/表单 | 60% |
| ChartQA 子集（纯图表）| 20 | 统计图表 | 100% |
| 财报 PDF（自建）| 15 | 金融 | 40% |
| arXiv 论文（自建）| 15 | 学术 | 33% |
| 统计报告（自建）| 15 | 政府/机构 | 40% |
| **总计** | **110** | **6 类** | **~50%** |

---

## 3. 🔴 P0｜缺少统计显著性检验

### 问题描述

报告直接声称"多模态 RAG 在视觉问题上 +91pp"，没有任何置信区间或显著性检验。在 n=25（视觉子集 n=11）的情况下，该差值的统计意义高度不确定。

### 修改方案：McNemar 检验

对两个系统在同一测试集上的配对结果，使用 **McNemar 检验**（专为配对二分类对比设计）：

```python
# 在 evaluation/run_eval.py 末尾添加以下代码

from scipy.stats import mcnemar
import numpy as np

def compute_mcnemar(results):
    """
    McNemar test comparing Multimodal RAG vs Text-only RAG.
    Contingency table:
        [[MM✅ & TO✅,  MM✅ & TO❌],
         [MM❌ & TO✅,  MM❌ & TO❌]]
    Only the off-diagonal cells matter.
    """
    both_correct = sum(1 for r in results if r["mm_correct"] and r["to_correct"])
    mm_only      = sum(1 for r in results if r["mm_correct"] and not r["to_correct"])
    to_only      = sum(1 for r in results if not r["mm_correct"] and r["to_correct"])
    both_wrong   = sum(1 for r in results if not r["mm_correct"] and not r["to_correct"])

    table = np.array([[both_correct, mm_only],
                      [to_only,      both_wrong]])

    # exact=True for small n (n < 25)
    n = len(results)
    use_exact = (n < 25)
    stat = mcnemar(table, exact=use_exact, correction=True)

    print(f"\nMcNemar Test (n={n}, exact={use_exact}):")
    print(f"  Contingency: MM✅&TO✅={both_correct}, MM✅&TO❌={mm_only}, "
          f"MM❌&TO✅={to_only}, MM❌&TO❌={both_wrong}")
    print(f"  Statistic = {stat.statistic:.4f}, p-value = {stat.pvalue:.4f}")
    if stat.pvalue < 0.05:
        print(f"  ✅ Statistically significant (p < 0.05): MM RAG outperforms Text-only RAG")
    else:
        print(f"  ⚠️  Not significant (p ≥ 0.05): difference may be due to chance")
    return stat

# 调用（在 run_evaluation() 末尾）:
# compute_mcnemar(results)                          # 全集
# compute_mcnemar([r for r in results if r["requires_visual"]])  # 视觉子集
```

**注意**：当前 25 道题中 MM=25, TO=14，不一致对为 (MM✅,TO❌)=11, (MM❌,TO✅)=0。McNemar exact test 的 p 值 = 2×(0.5)^11 ≈ 0.001，**显著**。即使样本小，这里也能得出显著结论，加入检验只会增强说服力，不会削弱。

扩充到 100 道题后，改用 `exact=False, correction=True` 的卡方近似即可。

---

## 4. 🟡 P1｜检索层从未独立评估

### 问题描述

系统是 RAG，但报告只报告了**端到端 QA 准确率**。这意味着我们不知道：

- 高准确率是来自「检索器找到了正确图片」还是「VLM 自身推理能力强」
- 检索失败了哪些图片，以及是为什么失败的
- `score_threshold=0.3` 和 `top_k=5` 这些超参数是否合理

### 修改方案：为每道题标注 Ground-Truth Chunk

#### Step 1：扩展数据集 JSON，添加 expected_elements 字段

```json
// evaluation/datasets/self_built_qa.json 新增字段示例
{
  "id": "q04",
  "question": "从训练损失曲线来看，哪种初始化方法的损失下降更快？",
  "type": "figure",
  "requires_visual": true,
  "expected_elements": {
    "images": ["figure_kmeans_loss_page5.png"],
    "text_chunks": [],
    "tables": []
  }
}
```

#### Step 2：在检索完成后计算 Retrieval Recall

```python
# evaluation/eval_retrieval.py（新文件）

def compute_retrieval_recall(retrieved_context, expected_elements, top_k=5):
    """
    Compute Recall@K for retrieval layer.
    Returns whether expected images/text chunks were retrieved.
    """
    retrieved_images = [
        Path(img["image_path"]).name
        for img in retrieved_context["image_contexts"]
    ]
    expected_images = expected_elements.get("images", [])

    if not expected_images:
        return {"image_recall": None, "image_found": None}

    found = sum(1 for e in expected_images if any(e in r for r in retrieved_images))
    recall = found / len(expected_images)

    return {
        "image_recall": recall,
        "image_found": found,
        "image_expected": len(expected_images),
        "retrieved_image_scores": [img["score"] for img in retrieved_context["image_contexts"]],
    }
```

#### Step 3：在报告中增加检索层指标表格

| 题型 | Image Recall@3 | Chunk Recall@5 | 平均检索分数 |
|---|---|---|---|
| figure（视觉必答） | ?% | - | ? |
| table | - | ?% | ? |
| text | - | ?% | ? |

**这张表格是区分「检索好但生成差」与「检索差但凑巧生成对了」的关键证据。**

---

## 5. 🟡 P1｜消融实验维度单一

### 问题描述

目前只有一条消融轴：`max_images=3`（MM）vs `max_images=0`（TO）。没有任何超参数敏感性实验，无法判断当前配置是否是最优的。

### 修改方案：两个核心消融实验

#### 消融 1：图片数量 vs 准确率（关键图，必做）

```python
# evaluation/ablation_images.py

import json
from evaluation.run_eval import run_single_question  # 封装单题评测逻辑

results_by_k = {}
for max_images in [0, 1, 2, 3, 5]:
    scores = []
    for qa in qa_dataset:
        context = retriever.retrieve_with_context(qa["question"], max_images=max_images)
        answer = generator.generate(qa["question"], context)["answer"]
        score = anls_score(answer, qa["gold_answers"])
        scores.append(score)
    results_by_k[max_images] = {
        "anls": sum(scores) / len(scores),
        "accuracy": sum(1 for s in scores if s >= 0.5) / len(scores),
    }

# 结果用折线图展示（见第 7 节可视化方案）
```

**预期图形**：在视觉题子集上，随 max_images 从 0→1→3，准确率应有显著跳升；在非视觉题上应基本持平。

#### 消融 2：检索阈值 score_threshold vs 准确率

```python
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    retriever = MultiVectorRetriever(
        embedder=embedder, vector_store=store,
        top_k=5, score_threshold=threshold
    )
    # ... 同上
```

**注意**：threshold 过低会引入噪声上下文（幻觉风险），过高会漏召回（拒答风险），曲线应呈倒 U 形。

---

## 6. 🟡 P1｜基线对比不公平

### 问题描述

Text-only 基线在 11 道视觉题中有 10 道的回答是「根据提供的文档内容无法回答此问题」，并非因为模型推理失败，而是 **GroundedGenerator 的拒答机制主动触发**。这导致 MM vs TO 的对比更像是「一个会看图的系统 vs 一个被禁止猜测的系统」，而非「两种理解能力的真实比较」。

### 修改方案：引入第三种基线

| 基线 | 配置 | 角色 |
|---|---|---|
| **MM RAG**（当前） | 文本 + 图像，grounded | 完整系统 |
| **Text-only Grounded**（当前 TO） | 纯文本，拒答不确定问题 | 「诚实基线」 |
| **Text-only Open**（新增） | 纯文本，允许基于上下文推断 | 「公平基线」 |

```python
# 新增第三种生成模式：去掉 grounded 约束
OPEN_SYSTEM_PROMPT = """
你是一个文档问答助手。根据提供的文档内容回答问题。
如果文档中没有直接信息，可以基于相关上下文进行合理推断，
但需要在回答末尾注明「（推断）」。
"""

result_open = generator.generate(
    question, context_text,
    system_prompt_override=OPEN_SYSTEM_PROMPT
)
```

**加入这个基线后**：如果 MM 仍然显著优于 Open TO，说明多模态的价值是真实的（而不仅仅是因为 TO 被禁止推理）。

---

## 7. 🟢 P2｜完全缺失可视化

### 问题描述

一个做**多模态**文档问答的系统，评测报告全是 Markdown 表格，没有任何图表。这在视觉呈现上是一个重大缺陷。

### 修改方案：5 张核心图表（Python 实现）

新建 `evaluation/visualize.py`，从 `results.json` 生成以下图表：

#### 图 1：主对比柱状图（必做，最重要）

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']

def plot_main_comparison(results_json_path, output_dir):
    with open(results_json_path) as f:
        data = json.load(f)

    categories = ['figure\n(视觉必答)', 'table\n(表格)', 'text\n(文本)', '整体']
    mm_scores = [
        data["by_type"]["figure"]["mm_accuracy"],
        data["by_type"]["table"]["mm_accuracy"],
        data["by_type"]["text"]["mm_accuracy"],
        data["overall"]["multimodal_accuracy"],
    ]
    to_scores = [
        data["by_type"]["figure"]["to_accuracy"],
        data["by_type"]["table"]["to_accuracy"],
        data["by_type"]["text"]["to_accuracy"],
        data["overall"]["text_only_accuracy"],
    ]

    x = range(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar([i - width/2 for i in x], mm_scores, width,
                   label='Multimodal RAG', color='#4C72B0', alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], to_scores, width,
                   label='Text-only RAG', color='#DD8452', alpha=0.85)

    # 标注数值
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f'{h:.0%}', ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Multimodal RAG vs Text-only RAG（按题目类型）', fontsize=13)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_main_comparison.png", dpi=150)
    plt.close()
```

#### 图 2：视觉 vs 非视觉切片对比（必做）

```python
def plot_visual_split(data, output_dir):
    labels = ['需要视觉\n(n=11)', '不需视觉\n(n=14)']
    mm = [data["visual_questions"]["mm_accuracy"],
          data["non_visual_questions"]["mm_accuracy"]]
    to = [data["visual_questions"]["to_accuracy"],
          data["non_visual_questions"]["to_accuracy"]]

    # 同上分组柱状图，或使用水平条形图
    # ...
    plt.savefig(f"{output_dir}/fig2_visual_split.png", dpi=150)
```

#### 图 3：消融曲线 - max_images vs Accuracy（需先跑消融）

```python
def plot_ablation_images(ablation_results, output_dir):
    """
    ablation_results: dict {max_images -> {"accuracy": float, "visual_accuracy": float}}
    """
    x = sorted(ablation_results.keys())
    y_all   = [ablation_results[k]["accuracy"] for k in x]
    y_vis   = [ablation_results[k]["visual_accuracy"] for k in x]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y_all, 'o-', label='全部题目', color='#4C72B0')
    ax.plot(x, y_vis, 's--', label='视觉题（figure）', color='#C44E52')
    ax.set_xlabel('max_images（传给 VLM 的最大图片数）')
    ax.set_ylabel('Accuracy')
    ax.set_title('图片数量对 QA 准确率的影响')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_ablation_images.png", dpi=150)
```

#### 图 4：Text-only 失败模式分布（饼图）

```python
def plot_to_failure_analysis(results, output_dir):
    to_failures = [r for r in results if not r["to_correct"]]
    # 分类失败原因
    def classify_failure(r):
        answer = r["text_only_answer"]
        if "无法回答" in answer or "不知道" in answer or "没有" in answer:
            return "拒答（无图信息）"
        elif any(g in answer for g in r["gold_answers"]):
            return "指标误判（答对了）"
        else:
            return "内容错误（有答但答错）"

    from collections import Counter
    counts = Counter(classify_failure(r) for r in to_failures)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(counts.values(), labels=counts.keys(), autopct='%1.0f%%',
           colors=['#4C72B0', '#55A868', '#DD8452'])
    ax.set_title('Text-only RAG 失败原因分析')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_to_failure_pie.png", dpi=150)
```

#### 图 5：每题 ANLS 分数热图（可选）

```python
def plot_per_question_heatmap(results, output_dir):
    import numpy as np
    ids = [r["id"] for r in results]
    mm_scores = [r["multimodal_anls"] for r in results]
    to_scores = [r["text_only_anls"] for r in results]

    data_matrix = np.array([mm_scores, to_scores])
    fig, ax = plt.subplots(figsize=(16, 3))
    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['MM RAG', 'Text-only'])
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha='right', fontsize=8)
    plt.colorbar(im, ax=ax, orientation='vertical', label='ANLS Score')
    ax.set_title('每道题 ANLS 分数（绿=正确，红=错误）')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_per_question_heatmap.png", dpi=150)
```

---

## 8. 🟢 P2｜缺少延迟与成本分析

### 问题描述

没有任何 latency 或 token 消耗数据，无法评估系统在实际使用中的可行性。

### 修改方案：在 run_eval.py 中加入计时和 token 统计

```python
import time

# 在 generate() 调用前后添加计时
t0 = time.time()
result_mm = generator.generate(question, context_mm)
latency_mm = time.time() - t0

# 记录到结果
results[-1]["latency_mm_sec"] = round(latency_mm, 2)
results[-1]["latency_to_sec"] = round(latency_to, 2)
results[-1]["num_images_sent"] = len(context_mm["image_contexts"])

# 汇总统计
avg_latency_mm = sum(r["latency_mm_sec"] for r in results) / n
avg_latency_to = sum(r["latency_to_sec"] for r in results) / n
avg_images     = sum(r["num_images_sent"] for r in results) / n

print(f"\nLatency:")
print(f"  MM RAG avg: {avg_latency_mm:.1f}s/question ({avg_images:.1f} images avg)")
print(f"  TO RAG avg: {avg_latency_to:.1f}s/question")
```

**在报告中增加如下表格（用测量值填写）：**

| 模式 | 平均响应时间 | 平均图片数 | 预估 API 成本/100题 |
|---|---|---|---|
| Multimodal RAG | ?s | ?张 | ¥? |
| Text-only RAG | ?s | 0张 | ¥? |

---

## 9. 🟢 P2｜题目难度分级缺失

### 问题描述

所有 25 道题的评测结果被平等对待，但难度差异悬殊：
- **Easy**：q02（查表格一个数字）vs  
- **Hard**：q20（同时阅读 4 张混淆矩阵比较 FP 数量）

将所有题混合报告，会掩盖系统在难题上的真实表现。

### 修改方案：在数据集中添加 difficulty 字段

```python
# 在 self_built_qa.json 每道题中添加（手工标注）
{
    "id": "q04",
    "difficulty": "easy",   // easy / medium / hard
    "difficulty_reason": "单图单值读取"
}

# difficulty 标准：
# easy:   单一来源（一张图或一行表格），直接读取，无需推理
# medium: 需要理解趋势/比较两项，或理解图表标注含义  
# hard:   跨多张图比较 / 需要计算 / 需要结合上下文推理
```

**在报告中按难度分组报告：**

| 难度 | 数量 | MM Acc | TO Acc |
|---|---|---|---|
| Easy | ? | ? | ? |
| Medium | ? | ? | ? |
| Hard | ? | ? | ? |

---

## 执行优先级与工作量估计

| 优先级 | 改进项 | 估计工作量 | 影响 |
|---|---|---|---|
| 🔴 P0 | 在报告中加入循环验证局限性声明 | 30 分钟 | 诚实性、可信度 |
| 🔴 P0 | 扩充数据集至 ≥100 题（含 4 类新文档）| 4–8 小时 | 结论鲁棒性 |
| 🔴 P0 | 加入 McNemar 统计检验 | 1 小时 | 科学严谨性 |
| 🟡 P1 | 图表可视化（5 张，用现有 results.json）| 2 小时 | 报告质量 |
| 🟡 P1 | max_images 消融曲线（需重跑 API）| 2 小时 + API 费用 | 技术深度 |
| 🟡 P1 | 检索层 Recall@K 评估（需标注 gold chunk）| 3 小时 | 系统理解 |
| 🟡 P1 | 引入 Text-only Open 基线 | 1.5 小时 | 对比公平性 |
| 🟢 P2 | 延迟 / 成本统计 | 0.5 小时 | 实用性 |
| 🟢 P2 | 难度分级 | 1 小时 | 分析细化 |

---

## 参考资源

- McNemar 检验文档：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mcnemar.html
- DocVQA 数据集：https://huggingface.co/datasets/nielsr/docvqa_1200_examples
- ChartQA 数据集：https://huggingface.co/datasets/docintel/ChartQA
- MMLongBench-Doc：https://huggingface.co/datasets/yubo2333/MMLongBench-Doc
- ArXivQA（科学图表 QA）：https://huggingface.co/datasets/MMInstruction/ArxivQA
- UniDoc-Bench（多模态 RAG 评测框架）：https://arxiv.org/html/2510.03663v2
