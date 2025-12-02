# reward_model.py
"""
RLHF 第一步：训练奖励模型 (Reward Model)

奖励模型的作用：
- 学习人类偏好，给模型回答打分
- 输入：prompt + response
- 输出：一个标量分数（越高表示人类越喜欢）

训练方式：
- 使用偏好数据（chosen vs rejected）
- 目标：让 chosen 的分数高于 rejected
- 损失函数：-log(sigmoid(r_chosen - r_rejected))

数据集：shibing624/DPO-En-Zh-20k-Preference（与 DPO 使用相同数据集）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    AutoModel,
    AutoConfig
)
from typing import Optional, Dict, List
import numpy as np

# ===== 配置 =====
SFT_MODEL_PATH = "./qwen_sft"           # 从 SFT 模型初始化 RM
OUTPUT_DIR = "./qwen_reward_model"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4         # 有效 batch = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
NUM_SAMPLES = 5000                      # 使用多少样本训练


# ===== 1. 自定义奖励模型 =====
class QwenRewardModel(PreTrainedModel):
    """
    奖励模型架构：
    
    Qwen Base Model (冻结或微调)
           ↓
    取最后一个 token 的隐藏状态
           ↓
    Linear 层 (hidden_size → 1)
           ↓
    输出：标量奖励分数
    """
    
    def __init__(self, config, base_model=None):
        super().__init__(config)
        
        if base_model is not None:
            self.model = base_model
        else:
            self.model = AutoModel.from_pretrained(
                SFT_MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        
        # 奖励头：将隐藏状态映射到标量分数
        self.reward_head = nn.Linear(config.hidden_size, 1, bias=False)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # 获取最后一层隐藏状态
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # 取最后一个 token 的隐藏状态（类似 [CLS] 的作用）
        # 对于每个样本，找到最后一个非 padding token
        if attention_mask is not None:
            # 找到每个序列最后一个 1 的位置
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            
            # 获取最后一个 token 的隐藏状态
            last_hidden_state = outputs.hidden_states[-1]
            pooled_output = last_hidden_state[
                torch.arange(batch_size, device=input_ids.device),
                sequence_lengths
            ]
        else:
            # 没有 attention_mask，直接取最后一个
            pooled_output = outputs.hidden_states[-1][:, -1, :]
        
        # 通过奖励头得到分数
        rewards = self.reward_head(pooled_output).squeeze(-1)
        
        return rewards


# ===== 2. 简化版：使用 AutoModelForSequenceClassification =====
# 如果上面的自定义模型有问题，可以用这个简化版
def load_reward_model_simple():
    """使用 HuggingFace 的序列分类模型作为奖励模型"""
    from transformers import AutoModelForSequenceClassification
    
    model = AutoModelForSequenceClassification.from_pretrained(
        SFT_MODEL_PATH,
        num_labels=1,  # 输出一个分数
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model


# ===== 3. 加载数据集 =====
print("Loading preference dataset...")
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载与 DPO 相同的数据集
raw_dataset = load_dataset(
    "shibing624/DPO-En-Zh-20k-Preference",
    name="zh",
    split=f"train[:{NUM_SAMPLES}]"
)


def preprocess_reward_data(examples):
    """
    将偏好数据处理成奖励模型训练格式
    
    对于每条数据，生成两个样本：
    - chosen 样本（标签为 1）
    - rejected 样本（标签为 0）
    
    但实际上我们使用 pairwise loss，所以不需要显式标签
    """
    chosen_input_ids = []
    chosen_attention_mask = []
    rejected_input_ids = []
    rejected_attention_mask = []
    
    for system, history, question, chosen, rejected in zip(
        examples["system"],
        examples["history"],
        examples["question"],
        examples["response_chosen"],
        examples["response_rejected"]
    ):
        # 构建 prompt（与 DPO 相同的格式）
        prompt_parts = []
        if system and system.strip():
            prompt_parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        
        if history:
            for turn in history:
                if len(turn) >= 2:
                    prompt_parts.append(f"<|im_start|>user\n{turn[0]}<|im_end|>")
                    prompt_parts.append(f"<|im_start|>assistant\n{turn[1]}<|im_end|>")
        
        prompt_parts.append(f"<|im_start|>user\n{question}<|im_end|>")
        prompt = "\n".join(prompt_parts)
        
        # chosen 完整序列
        chosen_text = f"{prompt}\n<|im_start|>assistant\n{chosen}<|im_end|>"
        # rejected 完整序列
        rejected_text = f"{prompt}\n<|im_start|>assistant\n{rejected}<|im_end|>"
        
        # Tokenize
        chosen_tokens = tokenizer(
            chosen_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )
        rejected_tokens = tokenizer(
            rejected_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )
        
        chosen_input_ids.append(chosen_tokens["input_ids"])
        chosen_attention_mask.append(chosen_tokens["attention_mask"])
        rejected_input_ids.append(rejected_tokens["input_ids"])
        rejected_attention_mask.append(rejected_tokens["attention_mask"])
    
    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
    }


print("Preprocessing data...")
dataset = raw_dataset.map(
    preprocess_reward_data,
    batched=True,
    remove_columns=raw_dataset.column_names,
    desc="Processing preference pairs"
)

print(f"✅ Processed {len(dataset)} preference pairs")


# ===== 4. 自定义 Trainer（Pairwise Ranking Loss）=====
class RewardModelTrainer(Trainer):
    """
    奖励模型训练器
    
    使用 Pairwise Ranking Loss:
    loss = -log(sigmoid(r_chosen - r_rejected))
    
    目标：让 chosen 的分数高于 rejected
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 获取 chosen 和 rejected 的分数
        chosen_rewards = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"]
        )
        rejected_rewards = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"]
        )
        
        # Pairwise Ranking Loss
        # 我们希望 chosen_rewards > rejected_rewards
        # loss = -log(sigmoid(chosen - rejected))
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        if return_outputs:
            return loss, {
                "chosen_rewards": chosen_rewards,
                "rejected_rewards": rejected_rewards
            }
        return loss


# ===== 5. 数据整理器 =====
class RewardDataCollator:
    """将 batch 数据整理成模型需要的格式"""
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {
            "chosen_input_ids": torch.tensor([f["chosen_input_ids"] for f in features]),
            "chosen_attention_mask": torch.tensor([f["chosen_attention_mask"] for f in features]),
            "rejected_input_ids": torch.tensor([f["rejected_input_ids"] for f in features]),
            "rejected_attention_mask": torch.tensor([f["rejected_attention_mask"] for f in features]),
        }
        return batch


# ===== 6. 训练 =====
def train_reward_model():
    print("=" * 60)
    print("🏆 开始训练奖励模型 (Reward Model)")
    print("=" * 60)
    
    # 加载配置和基础模型
    config = AutoConfig.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        SFT_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 创建奖励模型
    reward_model = QwenRewardModel(config, base_model)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    
    # 创建 Trainer
    trainer = RewardModelTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=RewardDataCollator(),
    )
    
    # 开始训练
    print(f"📊 训练配置:")
    print(f"   - 样本数: {len(dataset)}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - 梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   - 有效 Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    
    trainer.train()
    
    # 保存模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 奖励模型已保存到 {OUTPUT_DIR}")
    
    return reward_model


# ===== 7. 测试奖励模型 =====
def test_reward_model():
    """测试奖励模型的打分效果"""
    print("\n" + "=" * 60)
    print("🧪 测试奖励模型")
    print("=" * 60)
    
    # 加载模型
    config = AutoConfig.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        OUTPUT_DIR,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    reward_model = QwenRewardModel(config, base_model)
    reward_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    
    # 测试用例
    test_cases = [
        {
            "prompt": "请介绍一下人工智能",
            "good_response": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、问题解决、感知和语言理解等能力。",
            "bad_response": "不知道。"
        },
        {
            "prompt": "如何学习编程？",
            "good_response": "学习编程可以从以下几个步骤开始：1）选择一门入门语言如Python；2）学习基础语法和概念；3）多做练习项目；4）阅读优秀代码；5）参与开源社区。",
            "bad_response": "随便学学就行了。"
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- 测试 {i+1} ---")
        print(f"问题: {case['prompt']}")
        
        # 构建完整输入
        good_text = f"<|im_start|>user\n{case['prompt']}<|im_end|>\n<|im_start|>assistant\n{case['good_response']}<|im_end|>"
        bad_text = f"<|im_start|>user\n{case['prompt']}<|im_end|>\n<|im_start|>assistant\n{case['bad_response']}<|im_end|>"
        
        good_tokens = tokenizer(good_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        bad_tokens = tokenizer(bad_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
        
        with torch.no_grad():
            good_tokens = {k: v.to(reward_model.device) for k, v in good_tokens.items()}
            bad_tokens = {k: v.to(reward_model.device) for k, v in bad_tokens.items()}
            
            good_score = reward_model(**good_tokens).item()
            bad_score = reward_model(**bad_tokens).item()
        
        print(f"好回答分数: {good_score:.4f}")
        print(f"差回答分数: {bad_score:.4f}")
        print(f"差值: {good_score - bad_score:.4f} {'✅' if good_score > bad_score else '❌'}")


if __name__ == "__main__":
    # 训练奖励模型
    train_reward_model()
    
    # 测试奖励模型
    test_reward_model()
