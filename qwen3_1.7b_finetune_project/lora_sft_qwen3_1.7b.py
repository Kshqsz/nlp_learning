# lora_sft_qwen3_1.7b.py
"""
Qwen3-1.7B LoRA SFT（监督微调）

在继续预训练的模型基础上，使用 LoRA 进行监督微调
让模型学会遵循指令、进行对话

LoRA 优势：
  - 只训练 ~1% 的参数
  - 显存占用小，不需要 CPU Offload
  - 训练速度快（~2-3s/step）
  - 效果接近全量微调

运行方式：
  python lora_sft_qwen3_1.7b.py
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
)

# ===== 配置 =====
# 使用继续预训练后的模型作为基座
BASE_MODEL_PATH = "./qwen3_1.7b_pretrain"  # 预训练后的模型
# 如果预训练还没完成，可以先用原始模型
# BASE_MODEL_PATH = "/public/huggingface-models/Qwen/Qwen3-1.7B"

OUTPUT_DIR = "./qwen3_1.7b_lora_sft"
MAX_LENGTH = 512
BATCH_SIZE = 4              # LoRA 可以用更大的 batch
GRADIENT_ACCUMULATION_STEPS = 4  # 有效 batch = 16
LEARNING_RATE = 2e-4        # LoRA 通常用较大学习率
NUM_EPOCHS = 2
NUM_SAMPLES = 10000         # SFT 数据量

# LoRA 配置
LORA_R = 16                 # LoRA 秩
LORA_ALPHA = 32             # 缩放系数
LORA_DROPOUT = 0.05
TARGET_MODULES = [          # Qwen3 的目标模块
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ===== 1. 加载模型和 Tokenizer =====
print("=" * 60)
print("🚀 Qwen3-1.7B LoRA SFT 微调")
print("=" * 60)
print(f"基座模型: {BASE_MODEL_PATH}")
print(f"LoRA Rank: {LORA_R}, Alpha: {LORA_ALPHA}")

print("\n📦 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # LoRA 可以用 device_map
)

# 打印原始模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"基座模型参数量: {total_params / 1e9:.2f}B")


# ===== 2. 配置 LoRA =====
print("\n🔧 配置 LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
)

# 应用 LoRA
model = get_peft_model(model, lora_config)

# 启用 gradient checkpointing（必须在应用 LoRA 之后）
model.enable_input_require_grads()  # 关键：让输入需要梯度
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# 打印 LoRA 信息
model.print_trainable_parameters()


# ===== 3. 加载 SFT 数据集 =====
print("\n📊 加载 SFT 数据...")

# 从本地 JSONL 文件加载数据
DATA_PATH = "./firefly-train-1.1M.jsonl"

raw_dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

# 如果数据量大于 NUM_SAMPLES，只取前 NUM_SAMPLES 条
if len(raw_dataset) > NUM_SAMPLES:
    raw_dataset = raw_dataset.select(range(NUM_SAMPLES))

print(f"✅ 加载 {len(raw_dataset)} 条数据")


# ===== 4. 数据预处理 =====
def preprocess_function(examples):
    """
    SFT 数据处理：构建对话格式
    只对 assistant 的回复计算 loss
    """
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for kind, inp, target in zip(examples["kind"], examples["input"], examples["target"]):
        # 跳过过长的样本
        if len(inp) > 400 or len(target) > 400:
            continue
        
        # 构建 Qwen 对话格式
        # Qwen3 使用 ChatML 格式
        prompt = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"{prompt}{target}<|im_end|>"
        
        # Tokenize 完整文本
        tokenized = tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # 计算 prompt 长度，用于构建 labels
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        
        # Labels: prompt 部分设为 -100（不计算 loss）
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


print("\n🔄 处理数据...")
dataset = raw_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_dataset.column_names,
    desc="Processing SFT data",
    num_proc=4,
)

# 过滤空样本
dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
print(f"✅ 处理完成，有效样本: {len(dataset)}")


# ===== 5. 训练配置 =====
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
    gradient_checkpointing=False,  # 已在模型上手动启用
    optim="adamw_torch",
    
    # 日志和保存
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    
    # 其他
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=True,
)


# ===== 6. 开始训练 =====
print("\n" + "=" * 60)
print("🏋️ 开始 LoRA SFT 训练")
print("=" * 60)
print(f"   训练样本: {len(dataset)}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
print(f"   有效 Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   LoRA Rank: {LORA_R}")
print(f"   Epochs: {NUM_EPOCHS}")

if torch.cuda.is_available():
    print(f"   训练前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

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

# 训练
trainer.train()


# ===== 7. 保存 LoRA 权重 =====
print(f"\n💾 保存 LoRA 权重到 {OUTPUT_DIR}...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n📊 显存峰值: {peak_memory:.2f} GB")

print("\n✅ LoRA SFT 训练完成！")
print(f"📁 LoRA 权重已保存到: {OUTPUT_DIR}")


# ===== 8. 测试对话效果 =====
print("\n" + "=" * 60)
print("🧪 测试对话效果")
print("=" * 60)

# 测试问题
test_questions = [
    "请介绍一下人工智能的发展历史",
    "如何学习编程？",
    "写一首关于春天的诗",
    "Python 和 Java 有什么区别？",
]

model.eval()
for question in test_questions:
    # 构建对话格式
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 回复
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    print(f"\n【问题】{question}")
    print(f"【回答】{response[:300]}...")  # 截断显示


print("\n" + "=" * 60)
print("💡 后续步骤")
print("=" * 60)
print("""
1. 如需合并 LoRA 权重到基座模型，运行:
   from peft import PeftModel
   merged = model.merge_and_unload()
   merged.save_pretrained("./qwen3_1.7b_sft_merged")

2. 如需进一步进行 DPO/RLHF 对齐，可基于此 LoRA 模型继续训练
""")
