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
├── gpt/                                        # GPT 模型学习
│   ├── gpt2_generate.py                        # GPT-2 文本生成（Hugging Face）
│   └── mini_gpt.py                             # 从零实现 Mini GPT
├── pipeline/                                   # Hugging Face Pipeline 快速应用
│   ├── sentiment_analysis_demo.py              # 英文情感分析
│   └── sentiment_pipeline_chinese.py           # 中文情感分析
├── sentiment_classifier/                       # BERT 二分类微调
│   └── bert_finetune_sentiment.py              # 中文情感分类（ChnSentiCorp）
├── news_classifier/                            # BERT 多分类微调
│   └── bert_news_multiclass.py                 # 中文新闻分类（CLUE tnews 15分类）
└── qwen_chinese_finetune/                      # Qwen 中文大模型完整训练流程
    ├── pretrain_chinese-nvidia.py              # 中文继续预训练（NVIDIA GPU 版）
    ├── pretrain_chinese-mac.py                 # 中文继续预训练（Apple M4 版）
    ├── sft_chinese_qwen.py                     # 指令微调 (SFT)
    ├── dpo_chinese_qwen.py                     # 偏好对齐 (DPO)
    ├── reward_model.py                         # 奖励模型训练 (RLHF Step 1)
    ├── ppo_alignment.py                        # PPO 对齐训练 (RLHF Step 2)
    ├── compare_alignment.py                    # SFT vs DPO vs PPO 对比
    ├── test_gen.py                             # SFT 模型测试
    ├── test_sft.py                             # SFT 模型测试
    ├── test_dpo.py                             # DPO 模型测试
    └── test_ppo.py                             # PPO 模型测试
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

## 🤖 gpt

GPT (Generative Pre-trained Transformer) 模型学习，包含从零实现和使用预训练模型。

### 1. GPT-2 文本生成 (`gpt2_generate.py`)

**功能**：使用 Hugging Face 预训练的 GPT-2 模型进行文本生成

**核心概念**：

- **自回归生成**：根据前文逐词预测下一个词
- **采样策略**：temperature、top_p 控制生成多样性
- **因果语言模型**：只能看到左侧上下文

**示例代码**：

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

inputs = tokenizer("once upon a time", return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

**生成参数说明**：

| 参数 | 作用 |
|------|------|
| `max_new_tokens` | 最多生成的新 token 数量 |
| `temperature` | 控制随机性（越低越确定，越高越随机） |
| `top_p` | 核采样，只从累积概率达到该值的词中选择 |
| `do_sample` | 启用随机采样（否则用贪婪解码） |

**运行方式**：

```bash
cd gpt
python gpt2_generate.py
```

---

### 2. 从零实现 Mini GPT (`mini_gpt.py`)

**功能**：使用 PyTorch 从零实现一个简化版 GPT 模型

**核心概念**：

- **Decoder-only 架构**：GPT 只使用 Transformer 的解码器部分
- **Causal Mask**：因果掩码，确保每个位置只能看到之前的 token
- **位置编码**：可学习的位置嵌入
- **语言模型头**：将隐藏状态映射回词汇表进行预测

**模型结构**：

```
MiniGPT:
  - Token Embedding (vocab_size → embed_dim)
  - Position Embedding (max_len → embed_dim)
  - TransformerDecoderLayer × num_layers
    - Self-Attention (with causal mask)
    - FeedForward (embed_dim → 4×embed_dim → embed_dim)
  - LM Head (embed_dim → vocab_size)
```

**技术细节**：

| 配置项 | 值 |
|--------|-----|
| 嵌入维度 | 64 |
| 注意力头数 | 2 |
| 解码器层数 | 2 |
| 训练轮数 | 100 |

**Causal Mask 示意**：

```
位置:    0   1   2   3
     0 [ 1   0   0   0 ]  ← 位置 0 只能看自己
     1 [ 1   1   0   0 ]  ← 位置 1 能看 0, 1
     2 [ 1   1   1   0 ]  ← 位置 2 能看 0, 1, 2
     3 [ 1   1   1   1 ]  ← 位置 3 能看所有
```

**运行方式**：

```bash
cd gpt
python mini_gpt.py
```

**预期输出**：

```
Epoch 0, Loss: 1.8135
Epoch 20, Loss: 0.2747
...
生成结果: i love cats
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

## 🦙 qwen_chinese_finetune

完整的 Qwen 中文大模型训练流程：继续预训练 → 指令微调 (SFT) → 偏好对齐 (DPO) → 推理测试。

```
训练流程: Qwen-0.5B → 继续预训练 → SFT 微调 → DPO 对齐 → 部署推理
                ↓            ↓           ↓           ↓
            领域适配      遵循指令    人类偏好对齐   交互测试
```

### 1. 中文继续预训练

提供两个版本，适配不同硬件：

#### NVIDIA GPU 版 (`pretrain_chinese-nvidia.py`)

**功能**：在中文维基百科数据上对 Qwen 模型进行继续预训练（针对 NVIDIA GPU 优化）

**技术细节**：

| 配置项 | 值 |
|--------|-----|
| 基座模型 | Qwen/Qwen1.5-0.5B |
| 参数量 | 500M |
| 数据集 | pleisto/wikipedia-cn-20230720-filtered |
| 训练样本 | 3000 条（演示用）|
| 最大长度 | 512 tokens |
| Batch Size | 8 |
| 混合精度 | FP16 |
| 数据加载 | 4 workers 多进程 |

**运行方式**：

```bash
cd qwen_chinese_finetune
python pretrain_chinese-nvidia.py
```

#### Apple M4 版 (`pretrain_chinese-mac.py`)

**功能**：适配 Apple Silicon（M4/M3/M2/M1）的预训练版本

**技术细节**：

| 配置项 | NVIDIA 版 | Apple M4 版 |
|--------|-----------|-------------|
| 混合精度 | FP16 | Float32 |
| Batch Size | 8 | 2 |
| 梯度累积 | 1 | 4 |
| 数据加载 | 4 workers | 0 (单进程) |
| 设备 | CUDA | MPS |

**运行方式**：

```bash
cd qwen_chinese_finetune
python pretrain_chinese-mac.py
```

---

### 2. 指令微调 SFT (`sft_chinese_qwen.py`)

**功能**：在 Firefly 中文指令数据集上对预训练模型进行监督微调

**核心概念**：

- **SFT (Supervised Fine-Tuning)**：监督微调，让模型学会遵循指令
- **对话模板**：使用 Qwen 的 ChatML 格式 (`<|im_start|>user/assistant`)
- **Labels 掩码**：只对 assistant 回复部分计算 loss

**技术细节**：

| 配置项 | 值 |
|--------|-----|
| 基座模型 | 预训练后的 Qwen |
| 数据集 | YeungNLP/firefly-train-1.1M |
| 训练样本 | 10,000 条（过滤后）|
| 最大长度 | 512 tokens |
| Batch Size | 4 |
| 梯度累积 | 2 (有效 batch = 8) |
| Epochs | 2 |
| Learning Rate | 2e-5 |

**对话格式**：

```
<|im_start|>user
请介绍一下人工智能<|im_end|>
<|im_start|>assistant
人工智能是...<|im_end|>
```

**运行方式**：

```bash
cd qwen_chinese_finetune
python sft_chinese_qwen.py
```

---

### 3. 偏好对齐 DPO (`dpo_chinese_qwen.py`)

**功能**：使用 DPO (Direct Preference Optimization) 算法进行人类偏好对齐

**核心概念**：

- **DPO**：直接偏好优化，无需训练奖励模型的 RLHF 替代方案
- **偏好数据**：包含 chosen（好回答）和 rejected（差回答）的配对数据
- **参考模型**：使用冻结的 SFT 模型作为参考，防止模型偏离太远

**技术细节**：

| 配置项 | 值 |
|--------|-----|
| 基座模型 | SFT 微调后的 Qwen |
| 数据集 | shibing624/DPO-En-Zh-20k-Preference (中文子集) |
| 训练样本 | 5,000 条真实人类偏好数据 |
| Beta | 0.1（控制偏离参考模型的程度）|
| Batch Size | 2 |
| 梯度累积 | 4 (有效 batch = 8) |
| Learning Rate | 5e-6（比 SFT 更小）|

**DPO 原理**：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

其中 $y_w$ 是 chosen 回答，$y_l$ 是 rejected 回答。

**偏好数据格式**：

```json
{
  "prompt": "请解释什么是机器学习",
  "chosen": "机器学习是人工智能的一个分支，通过数据和算法让计算机...",
  "rejected": "机器学习就是让机器学习。"
}
```

**运行方式**：

```bash
cd qwen_chinese_finetune
python dpo_chinese_qwen.py
```

---

### 4. SFT 模型测试 (`test_gen.py`)

**功能**：测试 SFT 微调后的模型推理效果

**运行方式**：

```bash
cd qwen_chinese_finetune
python test_gen.py
```

---

### 5. DPO 模型测试 (`test_dpo.py`)

**功能**：测试 DPO 对齐后的模型效果，支持 SFT vs DPO 对比

**三种测试模式**：

| 模式 | 命令 | 说明 |
|------|------|------|
| 对比模式 | `python test_dpo.py --mode compare` | 对比 SFT vs DPO 模型输出 |
| DPO模式 | `python test_dpo.py --mode dpo` | 只测试 DPO 模型 |
| 交互模式 | `python test_dpo.py --mode interactive` | 交互式聊天测试 |

**运行方式**：

```bash
cd qwen_chinese_finetune

# 对比 SFT 和 DPO 模型（默认）
python test_dpo.py

# 交互式测试
python test_dpo.py --mode interactive
```

---

### 6. 交互式聊天 (`chat.py`)

**功能**：多轮对话交互测试，支持上下文记忆

**特性**：

- 支持多轮对话（最多 5 轮历史）
- 支持切换不同模型（预训练/SFT/DPO）
- 内置命令：`clear`（清空历史）、`history`（查看历史）、`quit`（退出）

**运行方式**：

```bash
cd qwen_chinese_finetune
python chat.py
```

---

### 7. RLHF 奖励模型 (`reward_model.py`)

**功能**：训练奖励模型（Reward Model），学习人类偏好打分

**核心概念**：

- **奖励模型**：给模型回答打分，分数越高表示人类越喜欢
- **Pairwise Ranking Loss**：让 chosen 的分数高于 rejected
- **模型架构**：SFT 模型 + 线性奖励头

**技术细节**：

| 配置项 | 值 |
|--------|-----|
| 基座模型 | SFT 模型 |
| 输出 | 标量分数 |
| 损失函数 | -log(sigmoid(r_chosen - r_rejected)) |
| Learning Rate | 1e-5 |

**运行方式**：

```bash
cd qwen_chinese_finetune
python reward_model.py
```

---

### 8. PPO 对齐训练 (`ppo_alignment.py`)

**功能**：使用 PPO 算法进行强化学习对齐（完整 RLHF 流程）

**核心概念**：

- **PPO**：Proximal Policy Optimization，策略梯度强化学习算法
- **Actor-Critic**：策略网络生成回答，价值网络估计状态价值
- **KL 惩罚**：防止模型偏离参考模型太远

**RLHF vs DPO 对比**：

```
RLHF: 偏好数据 → Reward Model → PPO → 对齐模型（两阶段）
DPO:  偏好数据 → 直接优化 → 对齐模型（一阶段）
```

**技术细节**：

| 配置项 | 值 |
|--------|-----|
| 基座模型 | SFT 模型 |
| 奖励来源 | 训练好的 Reward Model |
| PPO Epochs | 4 |
| KL Penalty | 0.1 |
| Learning Rate | 1e-5 |

**运行方式**：

```bash
cd qwen_chinese_finetune
# 先训练奖励模型
python reward_model.py
# 再运行 PPO
python ppo_alignment.py
```

---

### 9. 对齐方法对比 (`compare_alignment.py`)

**功能**：对比 SFT、DPO、PPO 三种方法的效果

**对比维度**：
- 回答质量（主观评估）
- 回答长度分布
- 词汇多样性
- 奖励模型分数

**运行方式**：

```bash
cd qwen_chinese_finetune

# 对比所有模型
python compare_alignment.py --mode all

# 只对比 DPO 和 PPO
python compare_alignment.py --mode dpo-ppo
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

### GPT 生成模型

- **Decoder-only 架构**：只使用 Transformer 解码器
- **Causal Mask**：因果掩码，确保只能看到左侧上下文
- **自回归生成**：逐词预测，每次生成一个 token
- **采样策略**：temperature 控制随机性，top_p 核采样控制多样性
- **语言模型目标**：最大化 $P(x_t | x_1, x_2, ..., x_{t-1})$

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

### 继续预训练 (Continued Pretraining)

- 在已有模型基础上，使用领域语料继续训练
- 因果语言模型 (Causal LM)：预测下一个 token
- 适用场景：领域适配、知识注入
- 混合精度训练：fp16 加速训练、节省显存
- 多进程数据加载：充分利用 CPU 资源

### 指令微调 (Instruction Tuning / SFT)

- 监督微调：让模型学会遵循用户指令
- 对话模板：ChatML 格式 (`<|im_start|>user/assistant`)
- Labels 掩码：只对回复部分计算 loss
- 数据集：Firefly 中文指令数据集

### 偏好对齐 (DPO)

- **DPO vs RLHF**：DPO 直接从偏好数据学习，无需训练奖励模型
- **偏好数据**：真实人类标注的 chosen/rejected 回答对
- **参考模型**：使用 SFT 模型作为参考，控制模型变化幅度
- **Beta 参数**：控制偏离参考模型的程度（通常 0.1~0.5）
- **学习率**：比 SFT 更小（5e-6 vs 2e-5）

### RLHF / PPO 强化学习对齐

- **完整 RLHF 流程**：SFT → Reward Model → PPO → 对齐模型
- **奖励模型**：学习人类偏好，给回答打分（Pairwise Ranking Loss）
- **PPO 算法**：Proximal Policy Optimization，策略梯度强化学习
- **Actor-Critic**：策略网络（生成）+ 价值网络（估值）
- **KL 惩罚**：防止模型偏离参考模型太远，保持稳定性
- **对比 DPO**：PPO 更复杂但可在线学习，DPO 更简单稳定

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

### 阶段二：GPT 生成模型（gpt）

6. **GPT-2 文本生成** → 学习使用预训练 GPT-2 进行文本续写
7. **从零实现 Mini GPT** → 理解 GPT Decoder-only 架构和 Causal Mask

### 阶段三：快速应用（pipeline）

8. **Hugging Face Pipeline** → 学习使用预训练模型进行推理
9. **中文 NLP 应用** → 掌握中文模型的使用

### 阶段四：模型微调（sentiment_classifier & news_classifier）

10. **BERT 二分类微调** → 学习在情感分类任务上微调 BERT
11. **BERT 多分类微调** → 学习在新闻分类任务上微调 BERT

### 阶段五：LLM 训练全流程（qwen_chinese_finetune）

12. **中文继续预训练** → 学习如何对 LLM 进行领域适配
13. **指令微调 (SFT)** → 学习如何让模型遵循指令对话
14. **偏好对齐 (DPO)** → 学习如何使用人类偏好数据对齐模型
15. **模型对比测试** → 学习如何评估 SFT vs DPO 的效果差异
16. **交互式部署** → 学习如何构建多轮对话系统

## 📚 后续计划

**已完成**：

- [X] 词向量训练
- [X] 简单情感分类
- [X] RNN 文本分类
- [X] Transformer 模型（手写 & PyTorch 模块）
- [X] GPT-2 文本生成
- [X] 从零实现 Mini GPT（Decoder-only + Causal Mask）
- [X] Hugging Face Pipeline 应用
- [X] 中文 NLP 任务
- [X] BERT 情感分类微调（二分类）
- [X] BERT 新闻分类微调（多分类 15 类）
- [X] LLM 继续预训练（Qwen 0.5B + 中文维基）
- [X] 指令微调 SFT（Firefly 中文指令数据集）
- [X] 多平台支持（NVIDIA GPU / Apple M4）
- [X] DPO 偏好对齐（真实人类偏好数据集）
- [X] 模型对比测试（SFT vs DPO）
- [X] 交互式多轮对话
- [X] RLHF 奖励模型训练（Reward Model）
- [X] PPO 强化学习对齐（完整 RLHF 流程）
- [X] SFT vs DPO vs PPO 对比实验

**进行中 / 计划中**：

- [ ] LSTM / GRU 序列标注
- [ ] 注意力机制深入可视化
- [ ] 命名实体识别 (NER)
- [ ] 文本生成任务
- [ ] LoRA / QLoRA 高效微调
- [ ] RAG 检索增强生成
- [ ] Agent 工具调用

---

**欢迎 Star ⭐ 和 Fork 🍴**
