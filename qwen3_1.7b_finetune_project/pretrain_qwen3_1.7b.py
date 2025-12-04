# pretrain_qwen3_1.7b.py
"""
Qwen3-1.7B 继续预训练（DeepSpeed ZeRO-2 优化）

硬件要求：NVIDIA 4090D (24GB) - 显存充裕
显存优化：
  - DeepSpeed ZeRO-2: 分片优化器状态和梯度
  - gradient_checkpointing: 用计算换显存
  - bf16: 半精度训练

运行方式（二选一）：
  python pretrain_qwen3_1.7b.py
  或
  deepspeed --num_gpus=1 pretrain_qwen3_1.7b.py

继续预训练 vs 从零预训练：
  - 从零预训练：随机初始化，需要数万亿 token
  - 继续预训练：利用已有知识，数百万 token 即可增强特定领域
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
    DataCollatorForLanguageModeling,
)

# ===== 配置 =====
MODEL_NAME = "/public/huggingface-models/Qwen/Qwen3-1.7B"  # Qwen3 1.7B 模型
OUTPUT_DIR = "./qwen3_1.7b_pretrain"
MAX_LENGTH = 512          # 1.7B 可以用更长序列
BATCH_SIZE = 1            # 1.7B 可以用更大 batch
GRADIENT_ACCUMULATION_STEPS = 8  # 有效 batch = 2
LEARNING_RATE = 1e-5      # 继续预训练用较小学习率
NUM_EPOCHS = 1
NUM_SAMPLES = 1000       # 训练样本数（可根据需要调整）
SAVE_STEPS = 500
LOGGING_STEPS = 50

# ===== DeepSpeed 配置 =====
DEEPSPEED_CONFIG = {
    "zero_optimization": {
        "stage": 2,  # ZeRO-2：分片优化器状态和梯度
        "offload_optimizer": {
            "device": "cpu",  # 必须开启，否则 OOM
            "pin_memory": True,
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 5e7,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e7,
        "overlap_comm": True,
        "contiguous_gradients": True,
    },
    "bf16": {
        "enabled": True
    },
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}

# ===== 1. 加载模型和 Tokenizer =====
print("=" * 60)
print("🚀 Qwen3-1.7B 继续预训练 (DeepSpeed ZeRO-2)")
print("=" * 60)
print(f"模型: {MODEL_NAME}")
print(f"序列长度: {MAX_LENGTH}")
print(f"Batch Size: {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

print("\n📦 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 注意：使用 DeepSpeed 时不要用 device_map="auto"，DeepSpeed 会自己管理
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype="bfloat16",  # 新版 transformers 使用 dtype
    low_cpu_mem_usage=True,  # 降低 CPU 内存占用
    # device_map="auto",  # DeepSpeed 不需要这个
)

# 启用梯度检查点（用计算换显存，必须开启）
model.gradient_checkpointing_enable()

# 打印模型信息
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ 模型加载完成")
print(f"   参数量: {total_params / 1e9:.2f}B")


# ===== 2. 加载数据集 =====
print("\n📊 加载训练数据...")

# 从本地 JSON 文件加载中文维基百科数据
DATA_PATH = "./wikipedia-cn-20230720-filtered.json"  # 本地 JSON 文件路径

raw_dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

# 如果数据量大于 NUM_SAMPLES，只取前 NUM_SAMPLES 条
if len(raw_dataset) > NUM_SAMPLES:
    raw_dataset = raw_dataset.select(range(NUM_SAMPLES))

text_column = "completion"  # JSON 中的文本字段名，根据实际情况修改

print(f"✅ 加载 {len(raw_dataset)} 条数据")


# ===== 3. 数据预处理 =====
def tokenize_function(examples):
    """
    预训练数据处理：
    - 纯文本，无对话格式
    - 直接预测下一个 token
    """
    # 使用 text_column 指定的字段
    texts = examples[text_column]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_special_tokens_mask=True,
    )
    
    return tokenized


print("\n🔄 处理数据...")
tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_dataset.column_names,
    desc="Tokenizing",
    num_proc=4,  # 多进程加速
)

# 过滤太短的样本
tokenized_dataset = tokenized_dataset.filter(
    lambda x: len(x["input_ids"]) >= 64,
    desc="Filtering short samples"
)

print(f"✅ 处理完成，有效样本: {len(tokenized_dataset)}")


# ===== 4. 数据整理器 =====
# 使用 MLM=False 的 DataCollator，即标准的 CLM（Causal LM）训练
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 因果语言模型，不是 BERT 的 MLM
)


# ===== 5. 训练配置 (DeepSpeed) =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    
    # 训练参数
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    
    # 学习率
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    
    # 显存优化
    bf16=True,
    gradient_checkpointing=True,
    
    # DeepSpeed 配置
    deepspeed=DEEPSPEED_CONFIG,
    
    # 日志和保存
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    
    # 其他
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=True,
)


# ===== 6. 开始训练 =====
print("\n" + "=" * 60)
print("🏋️ 开始继续预训练 (DeepSpeed ZeRO-2)")
print("=" * 60)
print(f"   训练样本: {len(tokenized_dataset)}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
print(f"   有效 Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   学习率: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   DeepSpeed: ZeRO-2 + CPU Offload")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 训练
trainer.train()


# ===== 7. 保存模型 =====
print(f"\n💾 保存模型到 {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 显存统计
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n📊 显存峰值: {peak_memory:.2f} GB")

print("\n✅ 继续预训练完成！")
print(f"📁 模型已保存到: {OUTPUT_DIR}")


# ===== 8. 简单测试 =====
print("\n" + "=" * 60)
print("🧪 测试生成效果")
print("=" * 60)

# 重新加载模型测试
del model
torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    trust_remote_code=True,
    dtype="bfloat16",
    device_map="auto"
)

test_prompts = [
    "人工智能的发展",
    "中国的首都北京",
    "机器学习是一种",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n输入: {prompt}")
    print(f"生成: {generated}")
