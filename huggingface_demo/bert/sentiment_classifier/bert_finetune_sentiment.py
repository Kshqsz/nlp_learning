from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import os
import torch

# 模型保存路径
MODEL_PATH = "./my_bert_model"

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 检查是否有已训练的模型
if os.path.exists(MODEL_PATH):
    print("✅ 发现已训练的模型，直接加载...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
else:
    print("⏳ 未发现已训练的模型，开始训练...")
    
    dataset = load_dataset("lansinuote/ChnSentiCorp")

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation = True,
            padding = "max_length",
            max_length = 128
        )

    tokenized_dataset = dataset.map(tokenize, batched = True)

    # 加载预训练的bert
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels = 2
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(p):
        logits = p.predictions
        preds = np.argmax(logits, axis = 1)
        return accuracy.compute(predictions = preds, references = p.label_ids)

    training_args = TrainingArguments(
        output_dir = "./results",
        eval_strategy = "epoch",
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        num_train_epochs = 1,
        logging_steps = 20
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset = tokenized_dataset["test"],
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )

    trainer.train()
    
    # 保存训练好的模型
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"✅ 模型已保存到 {MODEL_PATH}")

# 测试预测
model.eval()  # 切换到评估模式
model.to("cpu")  # 移到CPU上进行推理

text = "这个电影非常好看，我很喜欢！"                       
inputs = tokenizer(text, return_tensors="pt", truncation = True, padding = True)
logits = model(**inputs).logits
pred = logits.argmax(dim=1).item()
print(f"测试文本：'{text}'")
print("预测：", "正面 😊" if pred == 1 else "负面 😞")
