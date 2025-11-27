from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
import os
import torch

# ===== 1. 设置参数 =====
MODEL_NAME = "bert-base-chinese"
NUM_LABELS = 15  # tnews 有15个类别
MODEL_SAVE_PATH = "./tnews_bert_model"

# tnews 标签映射 (CLUE 版本标签已经是 0-14)
LABEL_NAMES = ["故事", "文化", "娱乐", "体育", "财经", "房产", "社会", "教育", "科技", "军事", "旅游", "国际", "股票", "农业", "电竞"]
id2label = {i: name for i, name in enumerate(LABEL_NAMES)}
label2id = {name: i for i, name in enumerate(LABEL_NAMES)}

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ===== 检查是否有已训练的模型 =====
if os.path.exists(MODEL_SAVE_PATH):
    print("✅ 发现已训练的模型，直接加载...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
else:
    print("⏳ 未发现已训练的模型，开始训练...")
    
    # ===== 2. 加载数据集 =====
    dataset = load_dataset("clue", "tnews")
    # 注意: CLUE 版本的 tnews 标签已经是 0-14，无需重映射

    # ===== 3. 加载 model =====
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id
    )

    # ===== 4. 分词函数 =====
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding=False,
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # ===== 5. 数据整理器（动态 padding）=====
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ===== 6. 评估指标（准确率 + F1）=====
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy.compute(predictions=predictions, references=labels)
        f1_macro = f1.compute(predictions=predictions, references=labels, average="macro")
        return {
            "accuracy": acc["accuracy"],
            "f1_macro": f1_macro["f1"]
        }

    # ===== 7. 训练配置 =====
    training_args = TrainingArguments(
        output_dir="./tnews_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )

    # ===== 8. 创建 Trainer =====
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ===== 9. 开始训练 =====
    trainer.train()

    # ===== 10. 保存模型 =====
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"✅ 模型已保存到 {MODEL_SAVE_PATH}")

# ===== 11. 测试推理 =====
model.eval()
model.to("cpu")

text = "中国男篮在亚洲杯比赛中大胜日本队"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
pred_id = outputs.logits.argmax().item()
print(f"测试文本: '{text}'")
print(f"预测类别: {id2label[pred_id]}")