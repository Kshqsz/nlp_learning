
# NLP Learning

这是一个用于学习自然语言处理 (NLP) 基础概念和实现的仓库，包含从零开始的简单 demo 代码。

## 📁 项目结构

```

nlp_learning/

├── README.md

└── simple_demo/

    ├── minimal_word_embedding_train.py         # 词向量训练

    ├── simple_sentiment_classifier.py          # 情感分类器

    ├── rnn_text_classifier.py                  # RNN 文本分类

    └── transformer_sentiment_classification.py # Transformer 情感分类

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

### 4. Transformer 情感分类器 (`transformer_sentiment_classification.py`)

**功能**：使用 Transformer 编码器进行文本情感分类

**核心概念**：

- Self-Attention：自注意力机制，捕获词之间的全局关系
- Position Encoding：位置编码，为模型提供序列位置信息
- Residual Connection：残差连接，帮助深层网络训练
- Layer Normalization：层归一化，稳定训练过程

**模型结构**：

```

MiniTransformerEncoder:

  - Embedding 层 + 位置编码 (vocab_size → 8)

  - Self-Attention 层 (Q, K, V 变换)

  - LayerNorm + 残差连接

  - FeedForward 层 (8 → 16 → 8)

  - LayerNorm + 残差连接

  - 分类层 (8 → 2)

  - 输出：正面 / 负面

```

**关键技术**：

- **Self-Attention**：$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- **位置编码**：使用可学习的位置参数，而非固定的 sin/cos 编码
- **残差连接**：$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$
- **CLS Token**：使用第一个位置的输出作为句子表示

**演示效果**：

- 训练 50 轮，快速收敛
- 可视化注意力权重矩阵，理解模型关注的词
- 对测试句子进行情感预测

**运行方式**：

```bash

cd simple_demo

python transformer_sentiment_classification.py

```

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

## 🛠️ 依赖环境

- Python 3.x
- PyTorch
- torch

## 📝 学习路径

1. **词向量训练** → 理解如何将词转换为向量

2. **简单情感分类** → 理解如何使用词向量进行文本分类

3. **RNN 文本分类** → 理解如何用循环神经网络处理序列数据

4. **Transformer 情感分类** → 理解 Self-Attention 和 Transformer 架构

## 📚 后续计划

- [x] RNN 文本分类
- [x] Transformer 模型
- [ ] LSTM 序列标注
- [ ] 注意力机制可视化
- [ ] 预训练模型微调 (BERT)

---

**欢迎 Star ⭐ 和 Fork 🍴**
