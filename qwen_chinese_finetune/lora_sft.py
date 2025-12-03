# lora_sft.py
"""
LoRA (Low-Rank Adaptation) 高效微调

LoRA 是什么：
- 冻结预训练模型权重，只训练低秩分解矩阵
- 大幅减少可训练参数量（通常 < 1%）
- 显存占用更小，训练速度更快
- 效果接近全量微调

LoRA 原理：
原始权重: W (d × d)
LoRA:     W' = W + ΔW = W + BA
          其中 B (d × r), A (r × d), r << d (r 通常取 8, 16, 32)

参数量对比（以 d=1024, r=8 为例）：
- 全量微调: 1024 × 1024 = 1,048,576 参数
- LoRA:     1024 × 8 + 8 × 1024 = 16,384 参数 (节省 98.4%)

本脚本使用 peft 库实现 LoRA 微调
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
import numpy as np

# ===== 配置 =====
# 基座模型（可以用 SFT 模型或原始预训练模型）
BASE_MODEL_PATH = "./qwen_sft"  # 使用 SFT 模型作为基座
# BASE_MODEL_PATH = "Qwen/Qwen1.5-0.5B"  # 或直接使用原始模型

OUTPUT_DIR = "./qwen_lora_sft"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # 有效 batch = 16
LEARNING_RATE = 2e-4              # LoRA 可以用更大的学习率
NUM_EPOCHS = 2
NUM_SAMPLES = 5000                # 训练样本数

# LoRA 配置
LORA_R = 8                        # LoRA 秩（rank）
LORA_ALPHA = 16                   # LoRA 缩放系数
LORA_DROPOUT = 0.05               # LoRA dropout
TARGET_MODULES = [                # 要应用 LoRA 的模块
    "q_proj",                     # Query 投影
    "k_proj",                     # Key 投影
    "v_proj",                     # Value 投影
    "o_proj",                     # Output 投影
    "gate_proj",                  # FFN gate
    "up_proj",                    # FFN up
    "down_proj",                  # FFN down
]


# ===== 1. 加载模型和 Tokenizer =====
print("=" * 60)
print("🚀 LoRA 高效微调")
print("=" * 60)

print("\n📦 加载基座模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 打印原始模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"基座模型参数量: {total_params / 1e6:.2f}M")


# ===== 2. 配置 LoRA =====
print("\n🔧 配置 LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,     # 因果语言模型任务
    r=LORA_R,                          # LoRA 秩
    lora_alpha=LORA_ALPHA,             # 缩放系数
    lora_dropout=LORA_DROPOUT,         # Dropout
    target_modules=TARGET_MODULES,     # 目标模块
    bias="none",                       # 不训练 bias
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 打印 LoRA 参数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"LoRA 可训练参数: {trainable_params / 1e6:.4f}M")
print(f"总参数量: {all_params / 1e6:.2f}M")
print(f"可训练比例: {100 * trainable_params / all_params:.2f}%")

# 打印 LoRA 配置
model.print_trainable_parameters()


# ===== 3. 加载数据集 =====
print("\n📊 加载训练数据...")

raw_dataset = load_dataset(
    "YeungNLP/firefly-train-1.1M",
    split=f"train[:{NUM_SAMPLES}]"
)


def preprocess_function(examples):
    """预处理数据：构建对话格式"""
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for kind, inp, target in zip(examples["kind"], examples["input"], examples["target"]):
        # 跳过过长的样本
        if len(inp) > 300 or len(target) > 300:
            continue
        
        # 构建对话格式
        prompt = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"{prompt}{target}<|im_end|>"
        
        # Tokenize
        tokenized = tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # 计算 prompt 长度，用于 labels 掩码
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        
        # Labels: prompt 部分为 -100（不计算 loss）
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # 确保长度一致
        if len(labels) < len(input_ids):
            labels = labels + [-100] * (len(input_ids) - len(labels))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }


# 处理数据
dataset = raw_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_dataset.column_names,
    desc="Processing data"
)

# 过滤空样本
dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)

print(f"✅ 处理后样本数: {len(dataset)}")


# ===== 4. 训练配置 =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    gradient_checkpointing=True,  # 节省显存
    optim="adamw_torch",
)


# ===== 5. 训练 =====
print("\n" + "=" * 60)
print("🏋️ 开始 LoRA 训练")
print("=" * 60)
print(f"   - 训练样本: {len(dataset)}")
print(f"   - Batch Size: {BATCH_SIZE}")
print(f"   - 梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
print(f"   - 有效 Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   - Learning Rate: {LEARNING_RATE}")
print(f"   - LoRA Rank: {LORA_R}")
print(f"   - LoRA Alpha: {LORA_ALPHA}")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()


# ===== 6. 保存 LoRA 权重 =====
print(f"\n💾 保存 LoRA 权重到 {OUTPUT_DIR}...")

# 只保存 LoRA 权重（很小，通常几十 MB）
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ LoRA 训练完成！")
print(f"\n📁 LoRA 权重已保存到: {OUTPUT_DIR}")
print("   注意：这只是 LoRA 增量权重，需要配合基座模型使用")
print("   如需合并权重，请运行 merge_lora.py")
