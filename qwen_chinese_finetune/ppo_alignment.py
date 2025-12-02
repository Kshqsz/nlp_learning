# ppo_alignment.py
"""
RLHF 第二步：PPO (Proximal Policy Optimization) 对齐训练

PPO 是什么：
- 一种策略梯度强化学习算法
- 在 RLHF 中，用奖励模型的分数作为反馈信号
- 优化策略（LLM）生成高奖励的回答

RLHF 完整流程：
1. SFT 模型（已完成）→ 会遵循指令
2. Reward Model（上一步）→ 学会打分
3. PPO 训练（本脚本）→ 优化生成高分回答

PPO 的关键组件：
- Actor（策略模型）：生成回答，就是要训练的 LLM
- Critic（价值模型）：预测状态价值，帮助计算优势函数
- Reward Model：给生成的回答打分
- Reference Model：KL 约束，防止模型跑偏

使用 trl 库简化实现（与 DPO 使用同一个库）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import numpy as np
from typing import List
import torch.nn as nn

# ===== 配置 =====
SFT_MODEL_PATH = "./qwen_sft"              # SFT 模型（作为初始策略）
REWARD_MODEL_PATH = "./qwen_reward_model"  # 奖励模型
OUTPUT_DIR = "./qwen_ppo"
MAX_LENGTH = 512
MAX_NEW_TOKENS = 128                        # 生成的最大新 token 数
BATCH_SIZE = 4                              # PPO mini-batch size
MINI_BATCH_SIZE = 2                         # PPO 内部 mini-batch
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
PPO_EPOCHS = 4                              # 每个 batch 的 PPO 更新轮数
NUM_TRAIN_STEPS = 500                       # 总训练步数
KL_PENALTY = 0.1                            # KL 散度惩罚系数
GAMMA = 1.0                                 # 折扣因子
LAM = 0.95                                  # GAE lambda
NUM_SAMPLES = 2000                          # 使用多少 prompt 训练


# ===== 1. 加载模型 =====
print("=" * 60)
print("🚀 PPO 对齐训练")
print("=" * 60)

print("\n📦 加载模型...")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载策略模型（带 Value Head）
# trl 提供的 AutoModelForCausalLMWithValueHead 会自动添加价值头
print("加载策略模型 (Actor + Critic)...")
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    SFT_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载参考模型（冻结，用于 KL 约束）
print("加载参考模型 (Reference)...")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    SFT_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载奖励模型
print("加载奖励模型 (Reward Model)...")

# 自定义奖励模型类（与 reward_model.py 相同）
class QwenRewardModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.model = base_model
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.shape[0]
            last_hidden_state = outputs.hidden_states[-1]
            pooled_output = last_hidden_state[
                torch.arange(batch_size, device=input_ids.device),
                sequence_lengths
            ]
        else:
            pooled_output = outputs.hidden_states[-1][:, -1, :]
        
        rewards = self.reward_head(pooled_output).squeeze(-1)
        return rewards


# 检查奖励模型是否存在
if os.path.exists(REWARD_MODEL_PATH):
    config = AutoConfig.from_pretrained(REWARD_MODEL_PATH, trust_remote_code=True)
    base_rm = AutoModel.from_pretrained(
        REWARD_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    reward_model = QwenRewardModel(base_rm, config.hidden_size)
    print(f"✅ 从 {REWARD_MODEL_PATH} 加载奖励模型")
else:
    print(f"⚠️ 未找到奖励模型 {REWARD_MODEL_PATH}")
    print("   将使用简化的奖励函数（基于回答长度和关键词）")
    reward_model = None


# ===== 2. 加载数据集（只需要 prompt）=====
print("\n📊 加载训练数据...")
raw_dataset = load_dataset(
    "shibing624/DPO-En-Zh-20k-Preference",
    name="zh",
    split=f"train[:{NUM_SAMPLES}]"
)


def extract_prompts(examples):
    """提取 prompt 用于 PPO 训练"""
    prompts = []
    
    for system, history, question in zip(
        examples["system"],
        examples["history"],
        examples["question"]
    ):
        prompt_parts = []
        if system and system.strip():
            prompt_parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        
        if history:
            for turn in history:
                if len(turn) >= 2:
                    prompt_parts.append(f"<|im_start|>user\n{turn[0]}<|im_end|>")
                    prompt_parts.append(f"<|im_start|>assistant\n{turn[1]}<|im_end|>")
        
        prompt_parts.append(f"<|im_start|>user\n{question}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        
        prompts.append("\n".join(prompt_parts))
    
    return {"prompt": prompts}


dataset = raw_dataset.map(
    extract_prompts,
    batched=True,
    remove_columns=raw_dataset.column_names
)

print(f"✅ 加载 {len(dataset)} 个训练 prompt")


# ===== 3. PPO 配置 =====
ppo_config = PPOConfig(
    model_name=SFT_MODEL_PATH,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    mini_batch_size=MINI_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    ppo_epochs=PPO_EPOCHS,
    gamma=GAMMA,
    lam=LAM,
    cliprange=0.2,                    # PPO clip 范围
    cliprange_value=0.2,              # Value 函数 clip 范围
    vf_coef=0.1,                      # Value loss 系数
    kl_penalty="kl",                  # KL 惩罚类型
    init_kl_coef=KL_PENALTY,          # 初始 KL 系数
    target_kl=0.1,                    # 目标 KL（自适应调整）
    log_with=None,                    # 不使用 wandb
    seed=42,
)


# ===== 4. 奖励函数 =====
def compute_rewards(
    prompts: List[str],
    responses: List[str],
    reward_model=None
) -> List[float]:
    """
    计算生成回答的奖励
    
    如果有奖励模型，使用 RM 打分
    否则使用简化的规则（仅用于演示）
    """
    rewards = []
    
    if reward_model is not None:
        # 使用训练好的奖励模型
        reward_model.eval()
        
        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt}{response}<|im_end|>"
            
            tokens = tokenizer(
                full_text,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                truncation=True
            )
            tokens = {k: v.to(model.pretrained_model.device) for k, v in tokens.items()}
            
            with torch.no_grad():
                reward = reward_model(**tokens).item()
            
            rewards.append(reward)
    else:
        # 简化的奖励函数（仅用于演示，实际应使用 RM）
        for response in responses:
            reward = 0.0
            
            # 奖励适中长度的回答（不太短也不太长）
            length = len(response)
            if 50 <= length <= 300:
                reward += 1.0
            elif length < 20:
                reward -= 1.0
            
            # 惩罚重复
            if len(set(response)) / max(len(response), 1) < 0.3:
                reward -= 0.5
            
            # 奖励包含一些正面词汇
            positive_words = ["谢谢", "帮助", "了解", "学习", "方法", "步骤"]
            for word in positive_words:
                if word in response:
                    reward += 0.2
            
            rewards.append(reward)
    
    return rewards


# ===== 5. PPO 训练循环 =====
def train_ppo():
    """PPO 主训练循环"""
    print("\n" + "=" * 60)
    print("🏋️ 开始 PPO 训练")
    print("=" * 60)
    print(f"   - 训练步数: {NUM_TRAIN_STEPS}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - PPO Epochs: {PPO_EPOCHS}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - KL Penalty: {KL_PENALTY}")
    
    # 创建 PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    
    # 生成配置
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # 训练统计
    all_rewards = []
    all_kls = []
    
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= NUM_TRAIN_STEPS:
            break
        
        # 获取 prompt
        prompt_tensors = batch["input_ids"]
        
        # 生成回答
        response_tensors = ppo_trainer.generate(
            prompt_tensors,
            **generation_kwargs
        )
        
        # 解码
        prompts = [tokenizer.decode(p, skip_special_tokens=False) for p in prompt_tensors]
        responses = [tokenizer.decode(r[len(p):], skip_special_tokens=True) 
                    for p, r in zip(prompt_tensors, response_tensors)]
        
        # 计算奖励
        rewards = compute_rewards(prompts, responses, reward_model)
        reward_tensors = [torch.tensor(r) for r in rewards]
        
        # PPO 更新
        stats = ppo_trainer.step(prompt_tensors, response_tensors, reward_tensors)
        
        # 记录统计
        all_rewards.extend(rewards)
        
        # 打印进度
        if step % 10 == 0:
            mean_reward = np.mean(rewards)
            print(f"Step {step}/{NUM_TRAIN_STEPS} | "
                  f"Mean Reward: {mean_reward:.4f} | "
                  f"KL: {stats.get('objective/kl', 0):.4f}")
    
    # 保存模型
    print(f"\n💾 保存模型到 {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ PPO 训练完成！")
    print(f"   平均奖励: {np.mean(all_rewards):.4f}")
    
    return model


# ===== 6. 测试 PPO 模型 =====
def test_ppo_model():
    """测试 PPO 训练后的模型"""
    print("\n" + "=" * 60)
    print("🧪 测试 PPO 模型")
    print("=" * 60)
    
    # 加载 PPO 模型
    if os.path.exists(OUTPUT_DIR):
        test_model = AutoModelForCausalLM.from_pretrained(
            OUTPUT_DIR,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        test_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    else:
        print(f"⚠️ PPO 模型不存在，使用当前训练的模型")
        test_model = model.pretrained_model
        test_tokenizer = tokenizer
    
    test_model.eval()
    
    test_prompts = [
        "请介绍一下人工智能",
        "如何学习编程？",
        "写一首关于春天的诗",
    ]
    
    for prompt in test_prompts:
        print(f"\n问题: {prompt}")
        print("-" * 40)
        
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = test_tokenizer(full_prompt, return_tensors="pt").to(test_model.device)
        
        with torch.no_grad():
            outputs = test_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=test_tokenizer.pad_token_id,
            )
        
        response = test_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        print(f"回答: {response}")


if __name__ == "__main__":
    # PPO 训练
    train_ppo()
    
    # 测试模型
    test_ppo_model()
