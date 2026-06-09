# 评测报告：多模态 RAG vs 纯文本 RAG 对比实验

> **系统**：MultimodalQA — 文档智能助手（多模态 RAG）
> **最新评测时间**：2026 年 5 月（初版）；2026 年 6 月（扩充数据集 + 补充实验）
> **VLM 模型**：智谱 AI GLM-4.6V（max_tokens=4096）
> **嵌入模型**：BAAI/bge-m3（1024 维，CUDA + FP16）
> **向量库**：ChromaDB（持久化）
> **评测指标**：ANLS（扩展版，含子串匹配/数值转换/有序比较）+ 准确率（阈值 0.5）

---

## 1. 实验目的

定量验证多模态 RAG 相比纯文本 RAG 的优势，特别关注：
- 图表/混淆矩阵/曲线等视觉信息场景下多模态系统是否有显著优势
- 引入「Text-only Open」第三基线，区分"多模态提升"来自视觉理解还是纯粹因为 grounded 基线拒答
- 跨领域（5 类文档）的泛化能力

**三路消融对比**：

| 模式 | 配置 | 角色 |
|---|---|---|
| **Multimodal RAG** | 文本 + 图像 → VLM | 完整系统 |
| **Text-only Grounded** | 纯文本，无证据则拒答 | 诚实基线（原 TO） |
| **Text-only Open** | 纯文本，允许基于上下文推断 | 公平基线（新增）|

---

## 2. 评测数据集

### 2.1 数据集构成（124 题，5 类文档）

| 文档 | 领域 | 题数 | 视觉题 |
|---|---|---|---|
| 测试数据（K-Means/GMM 实验报告）.pdf | ML 实验报告 | 25 | 10 |
| 大数据报告（IMDB 情感分类）.pdf | ML 实验报告 | 27 | 12 |
| 2404.07143v2.pdf（Infini-attention） | CS / NLP | 20 | 7 |
| IPCC_AR6_SYR_SPM.pdf | 气候 / 环境科学 | 29 | 12 |
| 战事给全球经济前景...pdf（IMF Blog） | 宏观经济 | 23 | 11 |
| **合计** | **5 类** | **124** | **62（50%）** |

| 题目类型 | 数量 |
|---|---|
| figure（图表，需视觉/视觉有帮助） | 62 |
| table（表格数值） | 38 |
| text（正文段落理解） | 24 |

| 难度 | 数量 |
|---|---|
| easy（单来源直接读取）| 63 |
| medium（趋势/比较/2 信息整合）| 53 |
| hard（跨图比较/计算/推理）| 8 |

### 2.2 数据集构建说明

- **gold 标注流程**：先基于 PDF 原文独立标注 gold，再运行模型评测，不允许以模型输出为依据修改 gold
- **多答案设计**：每题 2–5 个等价表述，覆盖中英文、百分比/小数等表达形式
- **视觉题定义**：`requires_visual=true` 包含两类：(a) 仅靠图片才能回答；(b) 图片有明显帮助（趋势判断、颜色读取、图表比较等）

---

## 3. 实验结果（2026-06-09 实测，124 题，5 类文档）

### 3.1 总体指标

| 模式 | ANLS | 准确率 |
|---|---|---|
| **Multimodal RAG** | **0.6774** | **67.74%** |
| Text-only Grounded | 0.3710 | 37.10% |
| Text-only Open（新增）| 0.5202 | 52.42% |
| MM vs TO Grounded | +0.3064 | **+30.6 pp** |
| MM vs TO Open | +0.1572 | +15.3 pp |

### 3.2 按问题类型

| 类型 | 数量 | MM Acc | TO Grounded | TO Open |
|---|---|---|---|---|
| figure（视觉）| 62 | **66%** | 13% | 35% |
| table | 31 | 68% | 61% | 74% |
| text | 31 | 71% | 61% | 65% |

### 3.3 按难度

| 难度 | 数量 | MM Acc | TO Grounded | TO Open |
|---|---|---|---|---|
| easy | 63 | 73% | 56% | 67% |
| medium | 53 | 62% | 21% | 42% |
| hard | 8 | 62% | 0% | 12% |

### 3.4 视觉题 vs 非视觉题

| 子集 | 数量 | MM Acc | TO Grounded | TO Open |
|---|---|---|---|---|
| 视觉题 | 62 | **66%** | 13% | 35% |
| 非视觉题 | 62 | 69% | 61% | 69% |

### 3.5 统计显著性（McNemar 检验，MM vs TO Grounded）

| 子集 | b（MM对TO错）| c（MM错TO对）| χ² | p |
|---|---|---|---|---|
| 全部（n=124）| 38 | 0 | 36.03 | **<0.0001**（高度显著）|
| 视觉题（n=62）| 33 | 0 | 31.03 | **<0.0001**（高度显著）|
| 非视觉题（n=62）| 5 | 0 | 3.20 | 0.074（**不显著**，符合预期）|

### 3.6 延迟分析

| 模式 | 平均延迟（秒/题）|
|---|---|
| Multimodal RAG | 5.1s |
| Text-only Grounded | 4.6s |
| Text-only Open | 11.9s |
| 图像附加开销（MM−TO）| +0.6s |
| 平均图片数/题 | 1.46 |

### 3.7 TO Grounded 失败模式（视觉题 62 道）

| 失败类型 | 数量 | 占比 |
|---|---|---|
| 拒答（"根据文档内容无法回答"）| 46 | 85% |
| 推理错误（有答案但答错）| 8 | 15% |
| 答对 | 8 | 13% |

### 3.8 可视化（evaluation/figures/）

| 图表 | 文件 | 状态 |
|---|---|---|
| MM vs TO 按题型分组柱状图 | fig1_main_comparison.png | ✅ 已更新（2026-06-09）|
| 视觉题 vs 非视觉题对比 | fig2_visual_split.png | ✅ 已更新 |
| 每道题 ANLS 热图 | fig3_per_question_heatmap.png | ✅ 已更新 |
| Text-only 失败原因饼图 | fig4_to_failure_pie.png | ✅ 已更新 |
| McNemar 检验结果 | fig5_mcnemar_summary.png | ✅ 已更新 |
| 按难度分组柱状图 | fig7_difficulty.png | ✅ 已更新 |

---

## 4. 旧版 25 题历史结果（2026-05，仅供参考）

> 以下为数据集未扩充前的结果，仅供对比参考，**不代表当前系统在完整数据集上的性能**。

| 子集 | MM Acc | TO Acc | McNemar p |
|---|---|---|---|
| 全部（n=25） | 100% | 56% | 0.001 |
| 视觉题（n=11） | 100% | 9% | 0.002 |
| 非视觉题（n=14） | 100% | 93% | 1.000 |

**旧版 TO 失败模式**：11 道视觉题中 10 道为拒答（91%），1 道推理错误（9%）。

---

## 6. 系统配置

```yaml
# 检索
embedder: BAAI/bge-m3 (1024-dim, fp16, cuda)
vector_store: ChromaDB (persistent, cosine similarity)
top_k: 5
score_threshold: 0.3

# 生成
vlm_model: glm-4.6v (Zhipu AI)
max_tokens: 4096
max_images_per_query: 3
generation_modes:
  - auto (multimodal when images available)
  - grounded (text-only, refuse without evidence)
  - open (text-only, allow inference)

# 索引
collection_naming: "doc_{md5(filename)[:8]}"
```

---

## 7. 复现方法

```bash
# 1. 准备环境
conda activate multimodalQA

# 2. 启动后端（构建索引）
cd backend && uvicorn main:app --port 8001

# 3. 主评测（3路对比，含延迟统计）
python -m evaluation.run_eval

# 4. 消融实验（max_images sweep）
python -m evaluation.ablation_images

# 5. 检索层评估（先填 expected_elements.images）
python -m evaluation.eval_retrieval

# 6. 生成可视化图表
python -m evaluation.visualize

# 7. 用已有 results.json 重新打分（不消耗 API）
python -m evaluation.rescore
```

---

## 8. 关键结论（初版）

| 维度 | 结论 |
|---|---|
| 视觉题 MM vs TO Grounded | +91pp，p=0.002，高度显著 |
| 非视觉题 MM vs TO Grounded | +7pp，p=1.0，不显著（符合预期）|
| TO Grounded 失败模式 | 91% 为拒答，9% 为推理错误 |
| 跨领域泛化（124题版）| 待补充 |
| 消融（图片数量影响）| 待补充 |
