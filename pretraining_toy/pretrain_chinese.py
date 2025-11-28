# pretrain_chinese.py
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

# ===== 1. 配置（中文 + CUDA 优化）=====
MODEL_NAME = "Qwen/Qwen1.5-0.5B"
DATASET_NAME = "pleisto/wikipedia-cn-20230720-filtered"
OUTPUT_DIR = "./qwen_pretrained"
MAX_LENGTH = 512
NUM_TRAIN_EPOCHS = 1

# ===== 2. 加载 tokenizer 和 model =====
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# ===== 3. 加载并预处理数据集 =====
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train[:3000]")

def tokenize_function(examples):
    return tokenizer(
        examples["completion"], 
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
        return_overflowing_tokens=True
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

# ===== 4. 数据整理器 =====
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ===== 5. 训练配置（NVIDIA 4090D 优化）=====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=8,          # ← 关键：4090D 显存大，可设 8~16
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,                              # ← NVIDIA 推荐用 fp16（更快、更省内存）
    # bf16=True 也可，但 fp16 在 consumer GPU 上更成熟
    report_to="none",
    dataloader_num_workers=4,               # ← 利用多进程加速数据加载
    optim="adamw_torch",
    gradient_accumulation_steps=1,          # 如果 batch 太大可调高，这里不需要
)

# ===== 6. 创建 Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,  # 新版本推荐用 processing_class
)

# ===== 7. 开始训练 =====
print("🚀 Starting Chinese continued pretraining on NVIDIA 4090D...")
trainer.train()

# ===== 8. 保存模型 =====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 中文预训练模型已保存到 {OUTPUT_DIR}")