## 作业 4 ：电影评论情感分类实验报告（改进 版）

姓名：庄仕豪

学号：

23336355

## 一、实验任务与要求

本实验使用 IMDB 电影评论数据集完成二分类任务（ positive/negative ），并对以下模型进行对比：

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. RNN

## 二、数据集与预处理

## 2.1 数据来源

数据集为 Hugging Face 平台 IMDB 数据集，脚本首次运行自动下载并缓存到 data 目录。

## 数据规模如下：

1. 训练集：

25000 条

2. 测试集：

25000 条

- 3.

- 标签：

0 （负样本）、 1 （正样本）

## 2.2 预处理流程

1. 全部文本转小写
2. 去除非字母数字字符
3. 空格分词

## 对于 TF-IDF 模型：

1. 采用 unigram + bigram （ 1-gram, 2-gram ）
2. 使用英文停用词过滤
3. 词特征上限设置为 30000

## 对于 RNN 模型：

1. 在训练集上构建词表（ max vocab 60000 起）
2. 句子截断 / 补齐到固定长度（ max len 220 起）
3. 加入 、 特殊符号

## 三、模型方法与改进策略

## 3.1 TF-IDF + Logistic Regression

经典文本分类基线。该方法在高维稀疏空间中训练稳定、速度快，对情感任务通常有较强表现。

## 3.2 TF-IDF + Linear SVM

将线性分类器替换为 LinearSVC 。 SVM 对稀疏特征表现稳定，是文本分类常见强基线。

## 3.3 RNN （ BiLSTM ）改进

与普通 RNN 相比，本次进行了系统增强：

1. 从单向 LSTM 改为双向 BiLSTM
2. 隐层增大（ hidden dim 256 ）、多层结构（ 2 layers ）
3. 使用 dropout 抑制过拟合
4. 增加梯度裁剪（ gradient clipping ）
5. 引入验证集（训练集 9:1 划分）
6. 使用 ReduceLROnPlateau 学习率调度
7. 采用 early stopping （ patience ）

## 四、评估指标

为避免只看 Accuracy 带来的片面性，本实验使用以下指标：

1. Accuracy ：整体正确率

2. Precision ：正类预测的准确性

3. Recall ：正类召回能力

4. F1 ：

Precision 与 Recall 的调和平均

5. ROC-AUC ：概率排序能力，越接近

1 越好

其中：

## 五、实验设置与运行

环境：

conda activate pytorch

核心运行命令：

```
cd 作业 4\code conda activate pytorch python .\sentiment_classification.py --data_dir ..\data --result_dir ..\result --subset_size -1 --rnn_epochs 6 --batch_size 128 --max_len 220 --max_vocab 60000 --rnn_lr 0.001 --patience 2
```

说明： subset\_size=-1 表示使用完整 25000 训练集与完整 25000 测试集。

## 六、实验结果

## [来自 ../result/performance.csv 的实测结果如下：](file:///C:/Users/12077/Desktop/college/%E5%A4%A7%E4%B8%89%E4%B8%8B/%E5%A4%A7%E6%95%B0%E6%8D%AE/%E4%BD%9C%E4%B8%9A4/result/performance.csv)

| Model                       |   Accuracy |   Precision |   Recall |     F1 |   ROC_AUC |
|-----------------------------|------------|-------------|----------|--------|-----------|
| TF-IDF + LogisticRegression |     0.8832 |      0.8820 |   0.8849 | 0.8834 |    0.9526 |
| TF-IDF + LinearSVM          |     0.8729 |      0.8810 |   0.8622 | 0.8715 |    0.9467 |
| RNN (BiLSTM, PyTorch)       |     0.8281 |      0.8081 |   0.8605 | 0.8335 |    0.8985 |
| RNN-Strong (BiLSTM)         |     0.8374 |      0.8561 |   0.8112 | 0.8330 |    0.9063 |

## 6.1 指标总览图

<!-- image -->

## 6.2 RNN 训练过程曲线

IMDB Model Comparison(Accuracy vs F1)

<!-- image -->

## 6.3 混淆矩阵

## TF-IDF + LogisticRegression ：

<!-- image -->

TF-IDF + LinearSVM ：

<!-- image -->

<!-- image -->

## RNN-Strong (BiLSTM) ：

<!-- image -->

## 七、结果分析

## 7.1 综合性能对比

1. LogisticRegression 在五项指标上整体最优（ F1=0.8834 ）
2. LinearSVM 紧随其后（ F1=0.8715 ）
3. 改进后的 BiLSTM 相比早期版本显著提升，但仍略低于 TF-IDF 线性模型

说明在 IMDB 任务中，经过良好特征工程的线性模型依然是非常强的基线。

## 7.2 RNN 效果

1. 与简单的 RNN 相比，改进版 RNN 已从接近随机水平提升到 F1 0.83+ 区间
2. 二阶段更高参数训练提高了 Accuracy 与 ROC-AUC ，但 F1 与第一阶段接近
3. 这表明模型存在一定的阈值 / 偏置变化，继续提升可能需要更系统的结构升级而非仅增加训练轮数

## 7.3 为什么 RNN 仍未超过 TF-IDF 线性模型

## 可能原因包括：

1. 未使用预训练词向量或预训练语言模型
2. 纯 BiLSTM 对长文本关键句提取能力有限
3. 文本分类任务中，情感关键词对线性模型已足够友好

## 八、结论与后续优化方向

## 8.1 结论

本次实验在完整 IMDB 数据上完成了三类模型与多指标评估，结果显示：

1. 当前最佳模型为 TF-IDF + Logistic Regression
2. SVM 表现稳定且接近 LR
3. 改进后 BiLSTM 已明显提升，但仍未超过 TF-IDF 线性基线

## 8.2 后续优化方向

1. 尝试 GRU + Attention 或 TextCNN-LSTM 混合结构
2. 使用 GloVe/FastText 预训练词向量
3. 尝试 BERT 类预训练模型进行对照
4. 在验证集上搜索最佳分类阈值以提升 F1

## 九、实验产出文件

- 1.

- 代码主文件： ../code/sentiment\_classification.py

2. 指标结果： ../result/performance.csv

3. 指标图：

../result/accuracy\_comparison.png

4. 训练曲线： ../result/rnn\_training\_curve.png

5. 混淆矩

阵： ../result/confusion\_matrix\_tf\_idf\_logisticregression.png 、 ../result/confusion\_matrix\_tf\_idf \_linearsvm.png 、 ../result/confusion\_matrix\_rnn\_bilstm\_pytorch.png 、 ../result/confusion\_mat rix\_rnn\_strong\_bilstm.png