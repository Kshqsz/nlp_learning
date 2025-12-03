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

本实现使用手动 PPO 训练循环，不依赖 trl 的 PPOTrainer
这样可以更清晰地理解 PPO 的工作原理，同时避免版本兼容问题
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple

# ===== 配置 =====
SFT_MODEL_PATH = "./qwen_sft"              # SFT 模型（作为初始策略）
REWARD_MODEL_PATH = "./qwen_reward_model"  # 奖励模型
OUTPUT_DIR = "./qwen_ppo"
MAX_LENGTH = 512
MAX_NEW_TOKENS = 64                         # 生成的最大新 token 数（减少以节省显存）
BATCH_SIZE = 2                              # 每批处理的 prompt 数量（减少以节省显存）
GRADIENT_ACCUMULATION_STEPS = 2             # 梯度累积步数（有效 batch = 4）
LEARNING_RATE = 1e-5
NUM_TRAIN_STEPS = 200                       # 总训练步数
KL_COEF = 0.1                               # KL 散度惩罚系数
CLIP_RANGE = 0.2                            # PPO clip 范围
VALUE_CLIP_RANGE = 0.2                      # Value 函数 clip 范围
VALUE_COEF = 0.5                            # Value loss 系数
ENTROPY_COEF = 0.01                         # 熵奖励系数
GAE_LAMBDA = 0.95                           # GAE lambda
GAMMA = 1.0                                 # 折扣因子
NUM_PPO_EPOCHS = 2                          # 每个 batch 的 PPO 更新轮数（减少以节省显存）
NUM_SAMPLES = 1000                          # 使用多少 prompt 训练


# ===== 1. Value Head 模块 =====
class ValueHead(nn.Module):
    """
    价值头模块
    
    将 LLM 的隐藏状态映射到标量价值
    用于 Actor-Critic 中的 Critic 部分
    """
    def __init__(self, hidden_size: int, dtype=torch.bfloat16):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, 1, dtype=dtype)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        # 取最后一个 token 的隐藏状态
        x = hidden_states[:, -1, :]  # (batch, hidden_size)
        x = torch.tanh(self.dense(x))
        value = self.out_proj(x).squeeze(-1)  # (batch,)
        return value


# ===== 2. 策略模型（带 Value Head）=====
class PolicyModelWithValueHead(nn.Module):
    """
    策略模型 + 价值头
    
    - policy_model: 原始 LLM，负责生成 token
    - value_head: 预测状态价值
    """
    def __init__(self, model_path: str, dtype=torch.bfloat16):
        super().__init__()
        
        # 加载预训练模型
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto"
        )
        
        # 获取隐藏维度
        hidden_size = self.policy_model.config.hidden_size
        
        # 添加价值头
        self.value_head = ValueHead(hidden_size, dtype=dtype)
        # 将 value_head 移动到与 policy_model 相同的设备
        self.value_head.to(self.policy_model.device)
    
    @property
    def device(self):
        return self.policy_model.device
    
    def parameters(self):
        """返回所有需要训练的参数"""
        # 合并 policy_model 和 value_head 的参数
        for param in self.policy_model.parameters():
            yield param
        for param in self.value_head.parameters():
            yield param
    
    def train(self, mode=True):
        """设置训练模式"""
        self.policy_model.train(mode)
        self.value_head.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        self.policy_model.eval()
        self.value_head.eval()
        return self
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor = None,
        return_value: bool = False
    ):
        """
        前向传播
        
        返回：
        - logits: token 预测 logits
        - value: 状态价值（如果 return_value=True）
        """
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_value
        )
        
        logits = outputs.logits
        
        if return_value:
            hidden_states = outputs.hidden_states[-1]
            value = self.value_head(hidden_states)
            return logits, value
        
        return logits
    
    def generate(self, **kwargs):
        """生成文本"""
        return self.policy_model.generate(**kwargs)
    
    def save_pretrained(self, save_directory: str):
        """保存模型"""
        os.makedirs(save_directory, exist_ok=True)
        # 保存 policy model
        self.policy_model.save_pretrained(save_directory)
        # 保存 value head
        torch.save(
            self.value_head.state_dict(),
            os.path.join(save_directory, "value_head.pt")
        )


# ===== 3. 奖励模型（与 reward_model.py 相同）=====
class QwenRewardModel(nn.Module):
    """奖励模型：给回答打分"""
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.model = base_model
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        # 确保 reward_head 与 base_model 在同一设备和 dtype
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        self.reward_head.to(device=device, dtype=dtype)
        
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


# ===== 4. PPO 核心函数 =====
def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    计算 token 级别的 log 概率
    
    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len)
    mask: (batch, seq_len) - 只计算生成部分的 log prob
    
    返回: (batch,) - 每个样本的平均 log prob
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # 取出 labels 对应位置的 log prob
    # labels[:, 1:] 因为要预测下一个 token
    selected_log_probs = torch.gather(
        log_probs[:, :-1, :], 
        dim=-1, 
        index=labels[:, 1:].unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len-1)
    
    if mask is not None:
        # mask 也要对齐
        mask = mask[:, 1:]
        selected_log_probs = selected_log_probs * mask
        # 返回平均 log prob
        return selected_log_probs.sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    
    return selected_log_probs.mean(dim=-1)


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 GAE (Generalized Advantage Estimation) 计算优势函数
    
    在我们的场景中，每个 episode 只有一步（生成一个完整回答）
    所以简化为：advantage = reward - value
    
    返回：advantages, returns
    """
    # 简化版：单步 episode
    # advantage = reward - value (相当于 TD error)
    # return = reward (因为没有后续状态)
    advantages = rewards - values.detach()
    returns = rewards
    
    return advantages, returns


def ppo_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2
) -> torch.Tensor:
    """
    PPO 的 Clipped Surrogate Loss
    
    L = min(r * A, clip(r, 1-ε, 1+ε) * A)
    
    其中 r = exp(new_log_prob - old_log_prob) 是概率比
    """
    # 概率比
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Clipped 版本
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    
    # 取最小值（悲观估计）
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    
    # PPO loss 是负的（因为要最大化期望奖励）
    loss = -torch.min(surrogate1, surrogate2).mean()
    
    return loss


def value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float = 0.2
) -> torch.Tensor:
    """
    Value 函数 loss（带 clipping）
    
    类似于 PPO 的策略 loss，对 value 的更新也做 clip
    """
    # Clipped value
    clipped_values = old_values + torch.clamp(
        values - old_values, 
        -clip_range, 
        clip_range
    )
    
    # 两种 loss
    loss1 = (values - returns) ** 2
    loss2 = (clipped_values - returns) ** 2
    
    # 取最大值（悲观估计）
    loss = 0.5 * torch.max(loss1, loss2).mean()
    
    return loss


# ===== 5. 数据加载 =====
def load_prompts(tokenizer, num_samples: int) -> List[Dict]:
    """加载训练 prompt"""
    print("📊 加载训练数据...")
    
    raw_dataset = load_dataset(
        "shibing624/DPO-En-Zh-20k-Preference",
        name="zh",
        split=f"train[:{num_samples}]"
    )
    
    prompts = []
    for item in raw_dataset:
        prompt_parts = []
        
        if item["system"] and item["system"].strip():
            prompt_parts.append(f"<|im_start|>system\n{item['system']}<|im_end|>")
        
        if item["history"]:
            for turn in item["history"]:
                if len(turn) >= 2:
                    prompt_parts.append(f"<|im_start|>user\n{turn[0]}<|im_end|>")
                    prompt_parts.append(f"<|im_start|>assistant\n{turn[1]}<|im_end|>")
        
        prompt_parts.append(f"<|im_start|>user\n{item['question']}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        
        prompt_text = "\n".join(prompt_parts)
        prompts.append({"text": prompt_text})
    
    print(f"✅ 加载 {len(prompts)} 个训练 prompt")
    return prompts


# ===== 6. PPO 训练循环 =====
def train_ppo():
    """PPO 主训练循环"""
    print("=" * 60)
    print("🚀 PPO 对齐训练")
    print("=" * 60)
    
    # ===== 加载 tokenizer =====
    print("\n📦 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ===== 加载策略模型（Actor + Critic）=====
    print("加载策略模型 (Policy + Value Head)...")
    policy_model = PolicyModelWithValueHead(SFT_MODEL_PATH)
    
    # ===== 加载参考模型（冻结）=====
    print("加载参考模型 (Reference)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # ===== 加载奖励模型 =====
    print("加载奖励模型 (Reward Model)...")
    if os.path.exists(REWARD_MODEL_PATH):
        config = AutoConfig.from_pretrained(REWARD_MODEL_PATH, trust_remote_code=True)
        base_rm = AutoModel.from_pretrained(
            REWARD_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        reward_model = QwenRewardModel(base_rm, config.hidden_size)
        reward_model.eval()
        for param in reward_model.parameters():
            param.requires_grad = False
        print(f"✅ 从 {REWARD_MODEL_PATH} 加载奖励模型")
        use_reward_model = True
    else:
        print(f"⚠️ 未找到奖励模型 {REWARD_MODEL_PATH}")
        print("   将使用简化的奖励函数（基于回答长度）")
        reward_model = None
        use_reward_model = False
    
    # ===== 加载数据 =====
    prompts = load_prompts(tokenizer, NUM_SAMPLES)
    
    # ===== 设置优化器 =====
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95)
    )
    
    # 学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=NUM_TRAIN_STEPS
    )
    
    # ===== 训练统计 =====
    all_rewards = []
    all_kls = []
    
    print("\n" + "=" * 60)
    print("🏋️ 开始 PPO 训练")
    print("=" * 60)
    print(f"   - 训练步数: {NUM_TRAIN_STEPS}")
    print(f"   - Batch Size: {BATCH_SIZE}")
    print(f"   - 梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   - 有效 Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   - PPO Epochs: {NUM_PPO_EPOCHS}")
    print(f"   - Learning Rate: {LEARNING_RATE}")
    print(f"   - KL Coefficient: {KL_COEF}")
    
    # ===== 主训练循环 =====
    prompt_idx = 0
    optimizer.zero_grad()  # 初始化梯度
    
    for step in tqdm(range(NUM_TRAIN_STEPS), desc="PPO Training"):
        # 采样一个 batch 的 prompt
        batch_prompts = []
        for _ in range(BATCH_SIZE):
            batch_prompts.append(prompts[prompt_idx % len(prompts)])
            prompt_idx += 1
        
        # ===== 生成回答 =====
        policy_model.eval()
        
        generated_texts = []
        full_sequences = []
        prompt_lengths = []
        
        for prompt_data in batch_prompts:
            prompt_text = prompt_data["text"]
            inputs = tokenizer(prompt_text, return_tensors="pt").to(policy_model.device)
            prompt_len = inputs["input_ids"].shape[1]
            prompt_lengths.append(prompt_len)
            
            with torch.no_grad():
                outputs = policy_model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            full_seq = outputs[0]
            response = tokenizer.decode(full_seq[prompt_len:], skip_special_tokens=True)
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
            
            generated_texts.append(response)
            full_sequences.append(full_seq)
        
        # ===== 计算奖励 =====
        rewards = []
        
        for prompt_data, response in zip(batch_prompts, generated_texts):
            if use_reward_model and reward_model is not None:
                # 使用奖励模型打分
                full_text = f"{prompt_data['text']}{response}<|im_end|>"
                tokens = tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=MAX_LENGTH,
                    truncation=True
                ).to(next(reward_model.parameters()).device)
                
                with torch.no_grad():
                    reward = reward_model(**tokens).item()
            else:
                # 简化的奖励函数
                reward = 0.0
                length = len(response)
                if 50 <= length <= 300:
                    reward += 1.0
                elif length < 20:
                    reward -= 1.0
                # 惩罚重复
                if len(response) > 0 and len(set(response)) / len(response) < 0.3:
                    reward -= 0.5
            
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=policy_model.device)
        all_rewards.append(rewards.mean().item())
        
        # 清理生成阶段的显存
        torch.cuda.empty_cache()
        
        # ===== 计算 old log probs 和 values =====
        policy_model.eval()
        
        # Pad sequences to same length
        max_len = max(seq.shape[0] for seq in full_sequences)
        padded_input_ids = []
        attention_masks = []
        response_masks = []
        
        for seq, prompt_len in zip(full_sequences, prompt_lengths):
            seq_len = seq.shape[0]
            padding_len = max_len - seq_len
            
            if padding_len > 0:
                padded_seq = F.pad(seq, (0, padding_len), value=tokenizer.pad_token_id)
                attn_mask = torch.cat([
                    torch.ones(seq_len, device=seq.device),
                    torch.zeros(padding_len, device=seq.device)
                ])
            else:
                padded_seq = seq
                attn_mask = torch.ones(seq_len, device=seq.device)
            
            # Response mask: 只对生成的 token 计算 loss
            resp_mask = torch.zeros(max_len, device=seq.device)
            resp_mask[prompt_len:seq_len] = 1.0
            
            padded_input_ids.append(padded_seq)
            attention_masks.append(attn_mask)
            response_masks.append(resp_mask)
        
        input_ids = torch.stack(padded_input_ids)  # (batch, max_len)
        attention_mask = torch.stack(attention_masks)  # (batch, max_len)
        response_mask = torch.stack(response_masks)  # (batch, max_len)
        
        with torch.no_grad():
            # Policy model log probs and values
            logits, values = policy_model(
                input_ids, attention_mask, return_value=True
            )
            old_log_probs = compute_log_probs(logits, input_ids, response_mask)
            old_values = values
            
            # Reference model log probs (for KL penalty)
            ref_logits = ref_model(input_ids, attention_mask=attention_mask).logits
            ref_log_probs = compute_log_probs(ref_logits, input_ids, response_mask)
        
        # KL divergence
        kl = (old_log_probs - ref_log_probs).mean()
        all_kls.append(kl.item())
        
        # 计算 advantages 和 returns
        # 奖励需要减去 KL 惩罚
        adjusted_rewards = rewards - KL_COEF * kl
        advantages, returns = compute_advantages(
            adjusted_rewards, old_values, GAMMA, GAE_LAMBDA
        )
        
        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # ===== PPO 更新 =====
        policy_model.train()
        
        for ppo_epoch in range(NUM_PPO_EPOCHS):
            # Forward pass
            new_logits, new_values = policy_model(
                input_ids, attention_mask, return_value=True
            )
            new_log_probs = compute_log_probs(new_logits, input_ids, response_mask)
            
            # Policy loss (PPO clipped)
            policy_loss = ppo_loss(
                old_log_probs.detach(),
                new_log_probs,
                advantages,
                CLIP_RANGE
            )
            
            # Value loss
            v_loss = value_loss(
                new_values,
                old_values.detach(),
                returns,
                VALUE_CLIP_RANGE
            )
            
            # Entropy bonus (鼓励探索) - 简化计算以节省显存
            # 只在 response 部分计算熵
            response_logits = new_logits[:, :-1, :] * response_mask[:, 1:].unsqueeze(-1)
            log_probs_all = F.log_softmax(response_logits, dim=-1)
            probs_all = F.softmax(response_logits, dim=-1)
            entropy = -(probs_all * log_probs_all).sum(dim=-1)
            entropy = (entropy * response_mask[:, 1:]).sum() / (response_mask[:, 1:].sum() + 1e-8)
            
            # Total loss (除以梯度累积步数)
            total_loss = (policy_loss + VALUE_COEF * v_loss - ENTROPY_COEF * entropy) / GRADIENT_ACCUMULATION_STEPS
            
            # Backward
            total_loss.backward()
            
            # 梯度累积：每 GRADIENT_ACCUMULATION_STEPS 步更新一次
            if (ppo_epoch + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or ppo_epoch == NUM_PPO_EPOCHS - 1:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 清理中间变量
            del new_logits, new_values, response_logits, log_probs_all, probs_all
        
        # 清理显存
        torch.cuda.empty_cache()
        
        scheduler.step()
        
        # ===== 日志 =====
        if (step + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            avg_kl = np.mean(all_kls[-10:])
            print(f"\n[Step {step+1}/{NUM_TRAIN_STEPS}] "
                  f"Avg Reward: {avg_reward:.4f}, "
                  f"Avg KL: {avg_kl:.4f}, "
                  f"Policy Loss: {policy_loss.item():.4f}")
    
    # ===== 保存模型 =====
    print(f"\n💾 保存模型到 {OUTPUT_DIR}...")
    policy_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ PPO 训练完成！")
    
    # 打印训练统计
    print(f"\n📈 训练统计:")
    print(f"   平均奖励: {np.mean(all_rewards):.4f}")
    print(f"   最终奖励: {np.mean(all_rewards[-10:]):.4f}")
    print(f"   平均 KL: {np.mean(all_kls):.4f}")
    
    return policy_model


# ===== 7. 测试 PPO 模型 =====
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
        print(f"✅ 从 {OUTPUT_DIR} 加载 PPO 模型")
    else:
        print(f"⚠️ PPO 模型不存在，请先运行训练")
        return
    
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
        
        response = test_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        print(f"回答: {response}")


if __name__ == "__main__":
    # PPO 训练
    train_ppo()
    
    # 测试模型
    test_ppo_model()
