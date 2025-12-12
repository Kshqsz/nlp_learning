# train_reward_model.py
"""
Qwen3-1.7B 奖励模型训练

训练一个奖励模型来评估模型输出的质量
奖励模型用于后续的 PPO 强化学习训练

运行方式：
  python train_reward_model.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import json

# ===== 配置 =====
# 基座模型路径
BASE_MODEL_PATH = "/root/data/hsk-models/qwen3_1.7b_lora_sft"  # SFT 后的模型
ORIGINAL_MODEL_PATH = "/public/huggingface-models/Qwen/Qwen3-1.7B"

# 输出路径
OUTPUT_DIR = "/root/data/hsk-models/qwen3_1.7b_reward_model"

# 超参数
MAX_LENGTH = 512
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4  # 有效 batch = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
NUM_SAMPLES = 10000  # 奖励模型数据量

# LoRA 配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_reward_dataset():
    """加载或创建奖励模型训练数据集"""
    print("\n📊 加载奖励模型训练数据...")
    
    # 尝试加载本地数据
    local_reward_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dpo_zh.jsonl")
    
    if os.path.exists(local_reward_path):
        print(f"   使用本地 SFT 数据: {local_reward_path}")
        dataset = load_dataset("json", data_files=local_reward_path, split="train")
        
        # 将 SFT 数据转换为奖励模型格式
        # 在这个简化版本中，我们假设高质量的 SFT 数据对应奖励 1，
        # 需要另外准备反例数据对应奖励 0
        
        if len(dataset) > NUM_SAMPLES:
            dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
        
        print(f"   ✅ 加载 {len(dataset)} 条数据")
        return dataset
    
    # 如果没有本地数据，创建演示数据
    print("   ⚠️ 未找到本地数据，创建演示数据...")
    
    # 演示数据格式：good response 和 bad response 对
    demo_data = []
    demo_examples = [
        {
            "question": "如何学习编程？",
            "good_response": "学习编程需要：1) 学习基础语法和概念 2) 通过项目实践 3) 阅读优质代码 4) 持续刷题训练。建议从 Python 或 JavaScript 开始。",
            "bad_response": "编程很难。",
        },
        {
            "question": "Python 的优势是什么？",
            "good_response": "Python 具有以下优势：1) 语法简洁易学 2) 库生态丰富 3) 应用广泛（Web、数据科学、AI 等）4) 社区活跃 5) 跨平台兼容。",
            "bad_response": "Python 不错。",
        },
        {
            "question": "怎样保持身体健康？",
            "good_response": "保持健康需要：1) 规律运动（每周 3-5 次）2) 均衡饮食 3) 充足睡眠（7-9 小时）4) 压力管理 5) 定期体检。",
            "bad_response": "多运动。",
        },
    ]
    
    # 扩展演示数据到所需数量
    for _ in range(NUM_SAMPLES // len(demo_examples)):
        demo_data.extend(demo_examples)
    
    # 创建数据集
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "question": [d["question"] for d in demo_data],
        "good_response": [d["good_response"] for d in demo_data],
        "bad_response": [d["bad_response"] for d in demo_data],
    })
    
    print(f"   ✅ 创建演示数据集：{len(dataset)} 条")
    return dataset


def preprocess_reward_data(examples, tokenizer):
    """
    将数据转换为奖励模型格式
    
    奖励模型的输入格式：
    - 对于 good response：[prompt, good_response] → 标签 1
    - 对于 bad response：[prompt, bad_response] → 标签 0
    """
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    # 处理不同的数据格式
    if "input" in examples and "target" in examples:
        # SFT 数据格式：{input, target}
        # 所有 SFT 数据假设都是高质量的，标签为 1
        for inp, target in zip(examples["input"], examples["target"]):
            if len(inp) > 400 or len(target) > 400:
                continue
            
            # 构建完整文本
            prompt = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n"
            response = f"{target}<|im_end|>"
            full_text = prompt + response
            
            # Tokenize
            tokenized = tokenizer(
                full_text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )
            
            input_ids_list.append(tokenized["input_ids"])
            labels_list.append(1)  # 高质量回复，标签为 1
            attention_mask_list.append(tokenized["attention_mask"])
    
    elif "question" in examples and ("response_chosen" in examples or "good_response" in examples):
        # 偏好对数据格式：{question, good_response, bad_response} 或 {question, response_chosen, response_rejected}
        
        # 统一列名
        if "response_chosen" in examples:
            good_responses = examples["response_chosen"]
            bad_responses = examples["response_rejected"]
        else:
            good_responses = examples["good_response"]
            bad_responses = examples["bad_response"]

        for question, good_resp, bad_resp in zip(
            examples["question"],
            good_responses,
            bad_responses
        ):
            if len(question) > 400 or len(good_resp) > 400 or len(bad_resp) > 400:
                continue
            
            # 处理 good response (标签 1)
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            good_full = prompt + f"{good_resp}<|im_end|>"
            
            tokenized_good = tokenizer(
                good_full,
                max_length=MAX_LENGTH,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )
            
            input_ids_list.append(tokenized_good["input_ids"])
            labels_list.append(1)
            attention_mask_list.append(tokenized_good["attention_mask"])
            
            # 处理 bad response (标签 0)
            bad_full = prompt + f"{bad_resp}<|im_end|>"
            
            tokenized_bad = tokenizer(
                bad_full,
                max_length=MAX_LENGTH,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )
            
            input_ids_list.append(tokenized_bad["input_ids"])
            labels_list.append(0)
            attention_mask_list.append(tokenized_bad["attention_mask"])
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def create_reward_model(model_path, tokenizer):
    """创建奖励模型"""
    print("📦 创建奖励模型...")
    
    # 从 causal LM 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 添加 value head（用于评分）
    # 简单方法：使用最后一个 token 的 hidden state 来预测分数
    hidden_size = model.config.hidden_size
    
    # 创建一个简单的线性层作为 reward head
    class RewardHead(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = torch.nn.Linear(hidden_size, 1)
        
        def forward(self, hidden_states):
            # 使用最后一个 token 的 hidden state
            last_hidden_state = hidden_states[:, -1, :]
            return self.linear(last_hidden_state)
    
    # 添加 reward head
    model.reward_head = RewardHead(hidden_size).to(model.device)
    
    return model


def main():
    print("=" * 60)
    print("🎯 Qwen3-1.7B 奖励模型训练")
    print("=" * 60)
    
    # 加载 tokenizer
    print("\n📦 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据
    dataset = load_reward_dataset()
    
    # 预处理数据
    print("\n🔄 处理数据...")
    processed_dataset = dataset.map(
        lambda x: preprocess_reward_data(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing reward data",
        num_proc=4,
    )
    
    # 过滤空样本
    processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"✅ 有效样本: {len(processed_dataset)}")
    
    # 加载模型
    print("\n📦 加载基座模型...")
    if os.path.exists(BASE_MODEL_PATH):
        print(f"   使用 SFT 模型: {BASE_MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print(f"   使用原始模型: {ORIGINAL_MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            ORIGINAL_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    
    # 配置 LoRA
    print("\n🔧 配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 添加 reward head
    hidden_size = model.config.hidden_size
    
    class RewardHead(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = torch.nn.Linear(hidden_size, 1)
        
        def forward(self, hidden_states):
            last_hidden_state = hidden_states[:, -1, :]
            return self.linear(last_hidden_state)
    
    model.reward_head = RewardHead(hidden_size).to(model.device)
    model.enable_input_require_grads()
    
    # 自定义 Data Collator
    class RewardDataCollator:
        def __init__(self, tokenizer, max_length=512):
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __call__(self, features):
            # 获取最大长度
            max_len = max(len(f["input_ids"]) for f in features)
            max_len = min(max_len, self.max_length)
            
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            
            for feature in features:
                input_ids = feature["input_ids"][:max_len]
                attention_mask = feature["attention_mask"][:max_len]
                
                # Padding
                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(float(feature["labels"]))
            
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.float),
            }
    
    # 自定义 Trainer（支持 reward head）
    class RewardTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # 前向传播
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            
            hidden_states = outputs.hidden_states[-1]  # 最后一层的 hidden states
            rewards = model.reward_head(hidden_states).squeeze(-1)
            
            # 计算二分类损失
            labels = inputs["labels"]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(rewards, labels)
            
            if return_outputs:
                return loss, (rewards, labels)
            return loss
    
    # 训练配置
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
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        
        # 显存优化
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        
        # 日志和保存
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        
        # 其他
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=42,
    )
    
    # 开始训练
    print("\n" + "=" * 60)
    print("🏋️ 开始奖励模型训练")
    print("=" * 60)
    print(f"   训练样本: {len(processed_dataset)}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   有效 Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    
    if torch.cuda.is_available():
        print(f"   训练前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    data_collator = RewardDataCollator(tokenizer)
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # 保存
    print(f"\n💾 保存奖励模型到 {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 单独保存 reward head 权重，便于在 PPO 阶段加载
    try:
        torch.save(model.reward_head.state_dict(), os.path.join(OUTPUT_DIR, "reward_head.pt"))
    except Exception as e:
        print(f"   ⚠️ 保存 reward_head.pt 失败: {e}")
    
    # 保存配置
    config = {
        "model_type": "reward_model",
        "base_model": BASE_MODEL_PATH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
    }
    with open(os.path.join(OUTPUT_DIR, "reward_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n📊 显存峰值: {peak_memory:.2f} GB")
    
    print("\n✅ 奖励模型训练完成！")
    print(f"📁 模型已保存到: {OUTPUT_DIR}")
    
    print("\n" + "=" * 60)
    print("💡 后续步骤")
    print("=" * 60)
    print("""
1. 使用奖励模型进行 PPO 训练:
   python lora_ppo_qwen3_1.7b.py

2. 评估模型性能:
   python eval_benchmarks.py --model_path /root/data/hsk-models/qwen3_1.7b_lora_ppo
""")


if __name__ == "__main__":
    main()
