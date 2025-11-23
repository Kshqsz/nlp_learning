# nlp_learning

这是一个用于学习 NLP (自然语言处理) 基础概念和代码实现的仓库。

## 📂 simple_demo

包含一些最基础的 NLP 概念演示代码，使用 PyTorch 实现。

### 1. 极简词向量训练 (`minimal_word_embedding_train.py`)

-**功能**: 演示如何从零开始训练一个简单的词向量 (Word Embedding)。

-**原理**: 使用一个简单的预测任务（根据当前词预测下一个词），通过 `nn.Embedding` 层学习词的向量表示。

-**展示**: 训练结束后，计算并展示了 "cats" 和 "dogs" 之间的余弦相似度，验证语义相近的词在向量空间中距离更近。

### 2. 简单情感分类器 (`simple_sentiment_classifier.py`)

-**功能**: 实现一个基于词向量平均 (Mean Pooling) 的简单文本情感分类模型。

-**数据**: 包含少量的正面和负面句子样本。

-**模型结构**: Embedding 层 -> 求平均 -> 全连接层 -> ReLU -> 输出层。

-**展示**: 训练模型并对测试句子 "i love this" 进行情感预测（正面/负面）。
