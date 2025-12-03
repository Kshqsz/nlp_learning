# qlora_sft.py
"""
QLoRA (Quantized LoRA) 量化 + LoRA 微调

QLoRA 是什么：
- 4-bit 量化 + LoRA 的组合
- 进一步降低显存占用（可在消费级显卡训练大模型）
- 用 NF4 量化格式存储基座模型
- 只训练 LoRA 参数（反向传播时用 bf16）

显存对比（以 Qwen-7B 为例）：
- Full Fine-tuning:  ~60GB
- LoRA (fp16):       ~15GB  
- QLoRA (4-bit):     ~6GB   ← 可在 4090 上训练 7B 模型！

QLoRA 关键技术：
1. NF4 量化：专为正态分布权重设计的 4-bit 量化
2. 双重量化：量化量化常数，进一步压缩
3. 分页优化器：处理显存峰值

本脚本演示 QLoRA 的使用方法
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
    BitsAndBytesConfig,  # 量化配置
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,  # QLoRA 必需
)

# ===== 配置 =====
# 对于小模型（0.5B）使用量化意义不大，这里主要是演示
# 实际应用中，QLoRA 更适合 7B+ 的大模型
BASE_MODEL_PATH = "Qwen/Qwen1.5-0.5B"
OUTPUT_DIR = "./qwen_qlora_sft"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
NUM_SAMPLES = 5000

# LoRA 配置
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ===== 1. 配置 4-bit 量化 =====
print("=" * 60)
print("🚀 QLoRA (4-bit 量化 + LoRA) 微调")
print("=" * 60)

# BitsAndBytes 4-bit 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # 使用 4-bit 量化
    bnb_4bit_quant_type="nf4",          # NF4 量化类型（推荐）
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用 bf16
    bnb_4bit_use_double_quant=True,     # 双重量化（进一步压缩）
)

print("\n📦 加载 4-bit 量化模型...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    quantization_config=quantization_config,  # 应用 4-bit 量化
    device_map="auto",
)

# 打印量化后显存占用
print(f"模型加载完成")
if torch.cuda.is_available():
    print(f"GPU 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


# ===== 2. 准备模型用于 k-bit 训练 =====
print("\n🔧 准备 QLoRA 训练...")

# 关键步骤：为 k-bit 训练准备模型
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True  # 使用梯度检查点节省显存
)


# ===== 3. 配置 LoRA =====
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
)

model = get_peft_model(model, lora_config)

# 打印参数信息
print("\n📊 参数统计:")
model.print_trainable_parameters()


# ===== 4. 加载数据 =====
print("\n📊 加载训练数据...")

raw_dataset = load_dataset(
    "YeungNLP/firefly-train-1.1M",
    split=f"train[:{NUM_SAMPLES}]"
)


def preprocess_function(examples):
    """预处理数据"""
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for inp, target in zip(examples["input"], examples["target"]):
        if len(inp) > 300 or len(target) > 300:
            continue
        
        prompt = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"{prompt}{target}<|im_end|>"
        
        tokenized = tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
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


dataset = raw_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_dataset.column_names,
    desc="Processing data"
)

dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
print(f"✅ 处理后样本数: {len(dataset)}")


# ===== 5. 训练配置 =====
# QLoRA 特定优化
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
    gradient_checkpointing=True,
    # QLoRA 特定配置
    optim="paged_adamw_8bit",  # 分页 8-bit AdamW 优化器
    max_grad_norm=0.3,        # 梯度裁剪
)


# ===== 6. 训练 =====
print("\n" + "=" * 60)
print("🏋️ 开始 QLoRA 训练")
print("=" * 60)
print(f"   - 量化: 4-bit NF4")
print(f"   - 训练样本: {len(dataset)}")
print(f"   - Batch Size: {BATCH_SIZE}")
print(f"   - 梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
print(f"   - 有效 Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   - Learning Rate: {LEARNING_RATE}")
print(f"   - LoRA Rank: {LORA_R}")

if torch.cuda.is_available():
    print(f"   - 训练前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

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


# ===== 7. 保存 =====
print(f"\n💾 保存 QLoRA 权重到 {OUTPUT_DIR}...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ QLoRA 训练完成！")
print(f"\n📁 权重已保存到: {OUTPUT_DIR}")

# 显存占用总结
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n📊 显存峰值: {peak_memory:.2f} GB")

print("""
💡 QLoRA vs LoRA 对比：
   
   | 方法    | 基座存储 | 显存占用 | 适用场景 |
   |---------|----------|----------|----------|
   | LoRA    | FP16/BF16| 中等     | 中等显存 |
   | QLoRA   | 4-bit    | 较小     | 有限显存 |
   
   对于 0.5B 小模型，QLoRA 优势不明显
   对于 7B+ 大模型，QLoRA 可节省 60%+ 显存！
""")
