# pretrain_qwen3b_deepspeed.py
"""
Qwen2.5-3B 继续预训练 - DeepSpeed ZeRO 优化版

如果普通版本 OOM，使用这个脚本 + DeepSpeed ZeRO-2 可以进一步降低显存

运行方式：
    accelerate launch --config_file ds_config.yaml pretrain_qwen3b_deepspeed.py
    
或者：
    deepspeed --num_gpus=1 pretrain_qwen3b_deepspeed.py
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
MODEL_NAME = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = "./qwen3b_pretrain_ds"
MAX_LENGTH = 512
BATCH_SIZE = 2            # DeepSpeed 可以稍大一点
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
NUM_SAMPLES = 50000

# ===== 加载模型 =====
print("=" * 60)
print("🚀 Qwen2.5-3B 继续预训练 (DeepSpeed)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 注意：使用 DeepSpeed 时不要用 device_map="auto"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    # device_map="auto",  # DeepSpeed 会自己处理
)

model.gradient_checkpointing_enable()

total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params / 1e9:.2f}B")


# ===== 加载数据 =====
print("\n📊 加载数据...")

try:
    raw_dataset = load_dataset(
        "pleisto/wikipedia-cn-20230720-filtered",
        split=f"train[:{NUM_SAMPLES}]"
    )
    text_column = "completion"
except:
    raw_dataset = load_dataset(
        "YeungNLP/firefly-train-1.1M",
        split=f"train[:{NUM_SAMPLES}]"
    )
    text_column = None


def tokenize_function(examples):
    if text_column and text_column in examples:
        texts = examples[text_column]
    else:
        texts = [f"{inp}\n{tgt}" for inp, tgt in zip(examples["input"], examples["target"])]
    
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_dataset.column_names,
    num_proc=4,
)

tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) >= 64)
print(f"✅ 有效样本: {len(tokenized_dataset)}")


# ===== 训练配置 (DeepSpeed) =====
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
    gradient_checkpointing=True,
    
    # DeepSpeed 配置
    deepspeed={
        "zero_optimization": {
            "stage": 2,  # ZeRO-2：分片优化器状态和梯度
            "offload_optimizer": {
                "device": "cpu",  # 优化器状态卸载到 CPU
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
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
    },
    
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)


# ===== 训练 =====
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("\n🏋️ 开始训练...")
trainer.train()

print(f"\n💾 保存模型到 {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ 完成！")
