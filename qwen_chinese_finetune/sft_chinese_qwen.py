# sft_chinese_qwen.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    set_seed
)
import torch

# ===== 配置 =====
MODEL_PATH = "./qwen_pretrained"     # ← 修正：指向预训练模型的正确路径
OUTPUT_DIR = "./qwen_sft"
MAX_LENGTH = 512
NUM_TRAIN_EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5

set_seed(42)

# ===== 1. 加载 tokenizer 和模型 =====
print(f"Loading tokenizer and model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# ===== 2. 加载 Firefly 中文指令数据集 =====
print("Loading Firefly dataset (Chinese instruction tuning data)...")
raw_dataset = load_dataset("json", data_files="./firefly-train-1.1M.jsonl", split="train[:10000]")

# 不再过滤，直接使用全部样本，防止数据为空
dataset = raw_dataset
print(f"Loaded {len(dataset)} samples from local file.")

# ===== 3. 应用 Qwen 对话模板并 tokenize =====
def format_and_tokenize(examples):
    # 适配本地数据集字段名：input -> instruction, target -> output
    instructions = examples["input"]
    outputs = examples["target"]
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for inst, out in zip(instructions, outputs):
        # 构建 Qwen 对话格式
        prompt = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n"
        full_text = prompt + out + "<|im_end|>"
        
        # Tokenize 完整文本
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # 创建 labels：只对 assistant 的回复计算 loss
        prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokenized["input_ids"])
        
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[:len(input_ids)]
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    format_and_tokenize,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

# ===== 4. 数据整理器 =====
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

# ===== 5. 训练配置 =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    bf16=True,  # 使用 bf16 代替 fp16，更稳定
    report_to="none",
    dataloader_num_workers=4,
    optim="adamw_torch",
    gradient_accumulation_steps=2,
    remove_unused_columns=False,
)

# ===== 6. 创建 Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ===== 7. 开始训练 =====
print("🚀 Starting SFT training...")
trainer.train()

# ===== 8. 保存模型 =====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ SFT 模型已保存到 {OUTPUT_DIR}")