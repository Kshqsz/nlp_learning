# NLP Learning

这是一个用于学习自然语言处理 (NLP) 基础概念和实现的仓库，包含从零开始的简单 demo 代码，以及使用预训练模型进行微调的实战项目。

## 📁 项目结构

```
nlp_learning/
├── README.md
├── simple_demo/                                # 从零实现的基础 demo
│   ├── minimal_word_embedding_train.py         # 词向量训练
│   ├── simple_sentiment_classifier.py          # 情感分类器
│   ├── rnn_text_classifier.py                  # RNN 文本分类
│   ├── transformer_sentiment_classification.py # Transformer 情感分类（手写实现）
│   └── transformer_encoder_classifier.py       # Transformer 情感分类（使用 PyTorch 模块）
├── pipeline/                                   # Hugging Face Pipeline 快速应用
│   ├── sentiment_analysis_demo.py              # 英文情感分析
│   └── sentiment_pipeline_chinese.py           # 中文情感分析
├── sentiment_classifier/                       # BERT 二分类微调
│   └── bert_finetune_sentiment.py              # 中文情感分类（ChnSentiCorp）
└── news_classifier/                            # BERT 多分类微调
    └── bert_news_multiclass.py                 # 中文新闻分类（CLUE tnews 15分类）
```

## 🚀 simple_demo

包含 NLP 基础概念演示，使用 PyTorch 从零实现。

### 1. 极简词向量训练 (`minimal_word_embedding_train.py`)

**功能**：从零开始训练词向量 (Word Embedding)

**核心概念**：

- 使用简单的上下文预测任务：根据当前词预测下一个词
- 通过 `nn.Embedding` 层学习词的向量表示
- 词向量维度：8 维

**模型结构**：

- Embedding 层：将词 ID 映射到 8 维向量
- Linear 层：将词向量映射回词汇表大小，用于预测下一个词

**演示效果**：

- 训练 1000 轮后，计算 "cats" 和 "dogs" 的余弦相似度
- 验证语义相近的词在向量空间中距离更近

### 2. 简单情感分类器 (`simple_sentiment_classifier.py`)

**功能**：实现基于词向量平均的文本情感分类

**核心概念**：

- Mean Pooling：对句子中所有词向量求平均，得到句子表示
- 二分类任务：正面情感 vs 负面情感

**模型结构**：

```

SimpleSentimentNet:

  - Embedding 层 (vocab_size → 8)

  - Mean Pooling (对所有词向量求平均)

  - 全连接层 1 (8 → 16)

  - ReLU 激活

  - 全连接层 2 (16 → 2)

  - 输出：正面 / 负面

```

**演示效果**：

- 训练 300 轮
- 测试句子 "i love this" 的情感预测

---

### 3. RNN 文本分类器 (`rnn_text_classifier.py`)

**功能**：使用循环神经网络 (RNN) 进行文本情感分类

**核心概念**：

- RNN：处理序列数据，捕获词序信息
- Padding：将不同长度的句子填充到相同长度
- 使用最后一个隐藏状态进行分类

**模型结构**：

```

SimpleRNNClassifier:

  - Embedding 层 (vocab_size → 8, padding_idx=0)

  - RNN 层 (8 → 16, batch_first=True)

  - 全连接层 (16 → 2)

  - 输出：正面 / 负面

```

**关键技术**：

- **序列填充 (Padding)**：统一句子长度为 6，不足部分用 `<PAD>` 填充
- **Hidden State**：使用 RNN 最后的隐藏状态作为句子表示
- **批量训练**：一次处理所有样本，提升训练效率

**演示效果**：

- 训练 300 轮，Loss 从 0.69 降至 0.0001
- 对测试句子进行情感预测

**运行方式**：

```bash

cd simple_demo

python rnn_text_classifier.py

```

---

### 4. Transformer 情感分类器（手写实现）(`transformer_sentiment_classification.py`)

**功能**：从零实现 Transformer 编码器进行文本情感分类

**核心概念**：

- Self-Attention：自注意力机制，捕获词之间的全局关系
- Position Encoding：位置编码，为模型提供序列位置信息
- Residual Connection：残差连接，帮助深层网络训练
- Layer Normalization：层归一化，稳定训练过程

**模型结构**：

```
MiniTransformerEncoder:
  - Embedding 层 + 位置编码 (vocab_size → 8)
  - Self-Attention 层 (Q, K, V 变换) ← 手写实现
  - LayerNorm + 残差连接
  - FeedForward 层 (8 → 16 → 8)
  - LayerNorm + 残差连接
  - 分类层 (8 → 2)
  - 输出：正面 / 负面
```

**关键技术**：

- **Self-Attention**：$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- **位置编码**：使用可学习的位置参数
- **残差连接**：$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$
- **CLS Token**：使用第一个位置的输出作为句子表示

**演示效果**：

- 训练 50 轮，快速收敛
- 可视化注意力权重矩阵，理解模型关注的词

**运行方式**：

```bash
cd simple_demo
python transformer_sentiment_classification.py
```

---

### 5. Transformer 情感分类器（PyTorch 模块）(`transformer_encoder_classifier.py`)

**功能**：使用 PyTorch 内置的 `nn.TransformerEncoder` 进行文本情感分类

**核心概念**：

- 使用 PyTorch 提供的 `nn.TransformerEncoderLayer` 和 `nn.TransformerEncoder`
- 多头注意力机制 (Multi-Head Attention)
- 支持多层 Transformer 堆叠

**模型结构**：

```
TransformerClassifier:
  - Embedding 层 + 位置编码 (vocab_size → 16)
  - TransformerEncoderLayer × 2 (2 heads, FFN = dim × 4)
  - 分类层 (16 → 2)
  - 输出：正面 / 负面
```

**与手写版本的对比**：

| 特性     | 手写版本   | PyTorch 模块版本      |
| -------- | ---------- | --------------------- |
| 注意力   | 单头       | 多头 (2 heads)        |
| 层数     | 1 层       | 2 层                  |
| 实现     | 手写 Q/K/V | nn.TransformerEncoder |
| 学习目的 | 理解原理   | 工程应用              |

**运行方式**：

```bash
cd simple_demo
python transformer_encoder_classifier.py
```

---

## 🔌 pipeline

使用 Hugging Face Pipeline API，快速应用预训练模型，无需训练。

### 1. 英文情感分析 (`sentiment_analysis_demo.py`)

**功能**：使用 Hugging Face 预训练模型进行英文情感分析

**核心概念**：

- Pipeline API：Hugging Face 提供的高层封装，一行代码实现推理
- 预训练模型：使用已经在大规模数据上训练好的模型
- 零配置：自动下载模型和分词器

**示例代码**：

```python
from transformers import pipeline

clf = pipeline("sentiment-analysis")
result = clf("this movie is very boring!")
print(result)  # [{'label': 'NEGATIVE', 'score': 0.9998}]
```

**运行方式**：

```bash
cd pipeline
python sentiment_analysis_demo.py
```

---

### 2. 中文情感分析 (`sentiment_pipeline_chinese.py`)

**功能**：使用中文预训练模型进行情感分析

**核心概念**：

- 指定模型：使用针对中文训练的模型
- RoBERTa 模型：基于 BERT 改进的预训练模型
- 在京东评论数据上微调：适用于中文电商评论场景

**示例代码**：

```python
from transformers import pipeline

clf = pipeline(
    "sentiment-analysis",
    model="uer/roberta-base-finetuned-jd-binary-chinese"
)

texts = [
    "这个电影真的太好看了，我特别喜欢！",
    "这个手机太卡了，我非常失望。"
]

for t in texts:
    result = clf(t)
    print(t, "=>", result)
```

**运行方式**：

```bash
cd pipeline
python sentiment_pipeline_chinese.py
```

---

## 🎯 sentiment_classifier

使用 BERT 预训练模型进行微调 (Fine-tuning)，实现更高精度的情感分类。

### BERT 情感分类微调 (`bert_finetune_sentiment.py`)

**功能**：在 ChnSentiCorp 中文情感数据集上微调 BERT 模型

**核心概念**：

- **Fine-tuning**：在预训练模型基础上，针对特定任务继续训练
- **BERT**：Bidirectional Encoder Representations from Transformers
- **数据集**：ChnSentiCorp（中文情感分析数据集，包含酒店、书籍等评论）

**技术细节**：

| 配置项     | 值                  |
| ---------- | ------------------- |
| 模型       | bert-base-chinese   |
| 参数量     | 110M                |
| 层数       | 12                  |
| 隐藏维度   | 768                 |
| 最大长度   | 128                 |
| Batch Size | 8                   |
| Epochs     | 1                   |
| 任务类型   | 二分类（正面/负面） |

**模型缓存**：

- 训练完成后自动保存到 `./my_bert_model`
- 再次运行时自动加载，无需重新训练

**运行方式**：

```bash
cd sentiment_classifier
python bert_finetune_sentiment.py
```

**预期输出**：

```
测试文本：'这个电影非常好看，我很喜欢！'
预测：正面 😊
```

---

## 📰 news_classifier

使用 BERT 进行多分类任务，实现中文新闻分类。

### BERT 新闻多分类 (`bert_news_multiclass.py`)

**功能**：在 CLUE tnews 数据集上微调 BERT，实现 15 类新闻分类

**核心概念**：

- **多分类任务**：输出 15 个类别的概率分布
- **CLUE Benchmark**：中文语言理解评测基准
- **tnews 数据集**：今日头条中文新闻短文本分类

**数据集信息**：

| 属性       | 值              |
| ---------- | --------------- |
| 数据集     | CLUE tnews      |
| 训练集大小 | 53,360 条       |
| 验证集大小 | 10,000 条       |
| 类别数量   | 15 类           |
| 文本类型   | 新闻标题/短文本 |

**15 个新闻类别**：

```
0: 故事    3: 体育    6: 社会    9: 军事    12: 股票
1: 文化    4: 财经    7: 教育   10: 旅游    13: 农业
2: 娱乐    5: 房产    8: 科技   11: 国际    14: 电竞
```

**技术细节**：

| 配置项        | 值                      |
| ------------- | ----------------------- |
| 模型          | bert-base-chinese       |
| 参数量        | 110M                    |
| Batch Size    | 16                      |
| Learning Rate | 2e-5                    |
| Epochs        | 1                       |
| 评估指标      | Accuracy + F1 Macro     |
| 动态 Padding  | DataCollatorWithPadding |

**模型缓存**：

- 训练完成后自动保存到 `./tnews_bert_model`
- 再次运行时自动加载，无需重新训练

**运行方式**：

```bash
cd news_classifier
python bert_news_multiclass.py
```

**预期输出**：

```
测试文本: '中国男篮在亚洲杯比赛中大胜日本队'
预测类别: 体育
```

---

## 💡 学习要点

### 词向量 (Word Embedding)

- 将词转换为稠密向量表示
- 捕获词的语义信息
- 相似的词在向量空间中距离更近

### 文本分类基础

- 句子表示：通过平均词向量得到
- 简单但有效的 baseline 方法
- 理解 NLP 任务的基本流程

### RNN 序列建模

- 捕获词序信息，理解上下文
- 序列填充 (Padding) 处理变长输入
- 使用隐藏状态作为句子表示
- 相比平均池化，能更好地理解语序

### Transformer 架构

- Self-Attention 机制捕获全局依赖关系
- 并行计算，训练速度快于 RNN
- 位置编码提供序列顺序信息
- 残差连接和层归一化稳定训练
- 注意力权重可视化，模型可解释性强

### 预训练模型应用

- Hugging Face 生态：丰富的预训练模型库
- Pipeline API：简化模型调用流程
- 迁移学习：利用大模型的知识
- 快速原型开发：无需从零训练
- 多语言支持：轻松处理中文等非英语任务

### BERT 微调 (Fine-tuning)

- 在预训练模型基础上针对特定任务继续训练
- 二分类：情感分析（正面/负面）
- 多分类：新闻分类（15 个类别）
- 使用 Hugging Face Trainer 简化训练流程
- 模型缓存机制：避免重复训练

## 🛠️ 依赖环境

- Python 3.x
- PyTorch
- transformers (Hugging Face)
- datasets (Hugging Face)
- evaluate
- accelerate

**安装命令**：

```bash
pip install torch transformers datasets evaluate accelerate
```

## 📍 学习路径

### 阶段一：基础概念（simple_demo）

1. **词向量训练** → 理解如何将词转换为向量
2. **简单情感分类** → 理解如何使用词向量进行文本分类
3. **RNN 文本分类** → 理解如何用循环神经网络处理序列数据
4. **Transformer 情感分类（手写）** → 理解 Self-Attention 和 Transformer 架构原理
5. **Transformer 情感分类（PyTorch）** → 学习使用 PyTorch 内置模块

### 阶段二：快速应用（pipeline）

6. **Hugging Face Pipeline** → 学习使用预训练模型进行推理
7. **中文 NLP 应用** → 掌握中文模型的使用

### 阶段三：模型微调（sentiment_classifier & news_classifier）

8. **BERT 二分类微调** → 学习在情感分类任务上微调 BERT
9. **BERT 多分类微调** → 学习在新闻分类任务上微调 BERT

## 📚 后续计划

**已完成**：

- [X] 词向量训练
- [X] 简单情感分类
- [X] RNN 文本分类
- [X] Transformer 模型（手写 & PyTorch 模块）
- [X] Hugging Face Pipeline 应用
- [X] 中文 NLP 任务
- [X] BERT 情感分类微调（二分类）
- [X] BERT 新闻分类微调（多分类 15 类）

**进行中 / 计划中**：

- [ ] LSTM / GRU 序列标注
- [ ] 注意力机制深入可视化
- [ ] 命名实体识别 (NER)
- [ ] 文本生成任务
- [ ] 大语言模型 (LLM) 应用

---

**欢迎 Star ⭐ 和 Fork 🍴**
