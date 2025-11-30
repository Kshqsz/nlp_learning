from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
import torch

# ===== 1. 配置（中文 + Apple M4 MPS 优化）=====
MODEL_NAME = "Qwen/Qwen1.5-0.5B"
DATASET_NAME = "pleisto/wikipedia-cn-20230720-filtered"
OUTPUT_DIR = "./qwen_pretrained"
MAX_LENGTH = 512
NUM_TRAIN_EPOCHS = 1

# 检测设备
if torch.backends.mps.is_available():
    print("🍎 使用 Apple M4 MPS 加速")
else:
    print("⚠️ MPS 不可用，使用 CPU")

# ===== 2. 加载 tokenizer 和 model =====
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    torch_dtype=torch.float32,  # MPS 对 float32 支持更好
)

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

# ===== 5. 训练配置（Apple M4 MPS 优化）=====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=2,          # ← M4 统一内存，batch 设小一点
    save_steps=500,
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=False,                             # ← MPS 不支持 fp16 训练
    bf16=False,                             # ← MPS 不支持 bf16 训练
    report_to="none",
    dataloader_num_workers=0,               # ← macOS 多进程有问题，必须设为 0
    optim="adamw_torch",
    gradient_accumulation_steps=4,          # ← 累积梯度，等效 batch_size = 2*4 = 8
    use_mps_device=True,                    # ← 启用 MPS
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
print("🚀 Starting Chinese continued pretraining on Apple M4...")
trainer.train()

# ===== 8. 保存模型 =====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 中文预训练模型已保存到 {OUTPUT_DIR}")