
# NLP Learning

这是一个用于学习自然语言处理 (NLP) 基础概念和实现的仓库，包含从零开始的简单 demo 代码。

## 📁 项目结构

```

nlp_learning/

├── README.md

└── simple_demo/

    ├── minimal_word_embedding_train.py    # 词向量训练

    └── simple_sentiment_classifier.py     # 情感分类器

```

## 🚀 simple_demo

包含两个最基础的 NLP 概念演示，使用 PyTorch 从零实现。

### 1. 极简词向量训练 (`minimal_word_embedding_train.py`)

**功能**：从零开始训练词向量 (Word Embedding)

**核心概念**：

- 使用简单的上下文预测任务：根据当前词预测下一个词
- 通过 `nn.Embedding` 层学习词的向量表示
- 词向量维度：8 维

**训练数据**：

```python

sentences = [

"i love cats", 

"i love dogs", 

"you love animals",

"dogs are cute", 

"cats are cute"

]

```

**模型结构**：

- Embedding 层：将词 ID 映射到 8 维向量
- Linear 层：将词向量映射回词汇表大小，用于预测下一个词

**演示效果**：

- 训练 1000 轮后，计算 "cats" 和 "dogs" 的余弦相似度
- 验证语义相近的词在向量空间中距离更近

**运行方式**：

```bash

cdsimple_demo

pythonminimal_word_embedding_train.py

```

---

### 2. 简单情感分类器 (`simple_sentiment_classifier.py`)

**功能**：实现基于词向量平均的文本情感分类

**核心概念**：

- Mean Pooling：对句子中所有词向量求平均，得到句子表示
- 二分类任务：正面情感 vs 负面情感

**训练数据**：

```python

# 正面样本

"i love this movie"

"this is amazing"

"i really like this film"

"the story is great"


# 负面样本

"i hate this movie"

"this is bad"

"i really dislike this film"

"the story is terrible"

```

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

**运行方式**：

```bash

cdsimple_demo

pythonsimple_sentiment_classifier.py

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

## 🛠️ 依赖环境

- Python 3.x
- PyTorch
- torch

## 📝 学习路径

1.**词向量训练** → 理解如何将词转换为向量

2.**情感分类** → 理解如何使用词向量进行文本分类

## 📚 后续计划

- [ ] RNN 文本分类
- [ ] LSTM 序列标注
- [ ] Attention 机制
- [ ] Transformer 模型

---

**欢迎 Star ⭐ 和 Fork 🍴**
