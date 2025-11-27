from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
import torch

MODEL_NAME = "Qwen/Qwen1.5-0.5B"
DATASET_NAME = "pleisto/wikipedia-cn-20230720-filtered"
OUTPUT_DIR = "./qwen_pretrained"
MAX_LENGTH = 512
NUM_TRAIN_EPOCHS = 1

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=1,
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16 = True,
    fp16 = False,
    report_to = "none",
    dataloader_num_workers = 0,
    optim = "adamw_torch",
)
trainer = Trainer(
    model=model,
    
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("🚀 Starting Chinese continued pretraining...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 中文预训练模型已保存到 {OUTPUT_DIR}")