# compare_alignment.py
"""
对比实验：SFT vs DPO vs PPO

本脚本对比三种对齐方法的效果：
1. SFT（监督微调）：只学习如何回答，没有偏好对齐
2. DPO（直接偏好优化）：直接从偏好数据学习
3. PPO（强化学习）：通过奖励模型反馈学习

对比维度：
- 回答质量（主观评估）
- 回答长度分布
- 词汇多样性
- 奖励模型分数
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
import numpy as np
from collections import Counter
import torch.nn as nn

# ===== 配置 =====
SFT_MODEL_PATH = "./qwen_sft"
DPO_MODEL_PATH = "./qwen_dpo"
PPO_MODEL_PATH = "./qwen_ppo"
REWARD_MODEL_PATH = "./qwen_reward_model"

# 测试问题
TEST_PROMPTS = [
    "请介绍一下人工智能",
    "如何学习编程？",
    "什么是深度学习？",
    "推荐一些学习Python的方法",
    "解释一下机器学习和深度学习的区别",
    "如何保持健康的生活方式？",
    "写一首关于春天的诗",
    "如何提高英语口语？",
]

# 生成参数
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}


# ===== 1. 加载模型 =====
def load_model(model_path: str, model_name: str):
    """加载模型"""
    if not os.path.exists(model_path):
        print(f"⚠️ {model_name} 不存在: {model_path}")
        return None, None
    
    print(f"加载 {model_name} from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# ===== 2. 加载奖励模型 =====
class QwenRewardModel(nn.Module):
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


def load_reward_model():
    """加载奖励模型"""
    if not os.path.exists(REWARD_MODEL_PATH):
        print("⚠️ 奖励模型不存在，将跳过奖励分数计算")
        return None, None
    
    print(f"加载奖励模型 from {REWARD_MODEL_PATH}...")
    config = AutoConfig.from_pretrained(REWARD_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        REWARD_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    reward_model = QwenRewardModel(base_model, config.hidden_size)
    reward_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH, trust_remote_code=True)
    
    return reward_model, tokenizer


# ===== 3. 生成回答 =====
def generate_response(model, tokenizer, prompt: str) -> str:
    """生成回答"""
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **GENERATION_CONFIG
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    return response.strip()


# ===== 4. 计算奖励分数 =====
def compute_reward(reward_model, tokenizer, prompt: str, response: str) -> float:
    """使用奖励模型计算分数"""
    if reward_model is None:
        return 0.0
    
    full_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    tokens = tokenizer(full_text, return_tensors="pt", max_length=512, truncation=True)
    tokens = {k: v.to(next(reward_model.parameters()).device) for k, v in tokens.items()}
    
    with torch.no_grad():
        reward = reward_model(**tokens).item()
    
    return reward


# ===== 5. 计算文本指标 =====
def compute_text_metrics(text: str) -> dict:
    """计算文本质量指标"""
    # 长度
    length = len(text)
    
    # 词汇多样性（unique chars / total chars）
    if length > 0:
        diversity = len(set(text)) / length
    else:
        diversity = 0
    
    # 重复率（检测重复的 n-gram）
    words = list(text)
    if len(words) >= 3:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        unique_trigrams = len(set(trigrams))
        total_trigrams = len(trigrams)
        repetition = 1 - (unique_trigrams / total_trigrams) if total_trigrams > 0 else 0
    else:
        repetition = 0
    
    return {
        "length": length,
        "diversity": diversity,
        "repetition": repetition
    }


# ===== 6. 对比实验 =====
def run_comparison():
    """运行对比实验"""
    print("=" * 80)
    print("🔬 SFT vs DPO vs PPO 对比实验")
    print("=" * 80)
    
    # 加载所有模型
    models = {}
    
    sft_model, sft_tokenizer = load_model(SFT_MODEL_PATH, "SFT")
    if sft_model:
        models["SFT"] = (sft_model, sft_tokenizer)
    
    dpo_model, dpo_tokenizer = load_model(DPO_MODEL_PATH, "DPO")
    if dpo_model:
        models["DPO"] = (dpo_model, dpo_tokenizer)
    
    ppo_model, ppo_tokenizer = load_model(PPO_MODEL_PATH, "PPO")
    if ppo_model:
        models["PPO"] = (ppo_model, ppo_tokenizer)
    
    if not models:
        print("❌ 没有找到任何模型，请先训练模型")
        return
    
    # 加载奖励模型
    reward_model, rm_tokenizer = load_reward_model()
    
    # 存储结果
    results = {name: {"responses": [], "rewards": [], "metrics": []} for name in models}
    
    # 对每个问题生成回答
    print("\n" + "=" * 80)
    print("📝 生成回答并评估")
    print("=" * 80)
    
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'─' * 80}")
        print(f"问题 {i+1}: {prompt}")
        print("─" * 80)
        
        for name, (model, tokenizer) in models.items():
            # 生成回答
            response = generate_response(model, tokenizer, prompt)
            results[name]["responses"].append(response)
            
            # 计算奖励
            if reward_model:
                reward = compute_reward(reward_model, rm_tokenizer, prompt, response)
            else:
                reward = 0.0
            results[name]["rewards"].append(reward)
            
            # 计算文本指标
            metrics = compute_text_metrics(response)
            results[name]["metrics"].append(metrics)
            
            # 打印回答
            print(f"\n🔵 {name} 模型:")
            print(f"   回答: {response[:200]}{'...' if len(response) > 200 else ''}")
            print(f"   长度: {metrics['length']} | 多样性: {metrics['diversity']:.3f} | 奖励: {reward:.4f}")
    
    # 汇总统计
    print("\n" + "=" * 80)
    print("📊 汇总统计")
    print("=" * 80)
    
    print("\n### 各模型平均指标")
    print(f"{'模型':<10} {'平均长度':<12} {'平均多样性':<12} {'平均重复率':<12} {'平均奖励':<12}")
    print("-" * 60)
    
    for name in models:
        avg_length = np.mean([m["length"] for m in results[name]["metrics"]])
        avg_diversity = np.mean([m["diversity"] for m in results[name]["metrics"]])
        avg_repetition = np.mean([m["repetition"] for m in results[name]["metrics"]])
        avg_reward = np.mean(results[name]["rewards"])
        
        print(f"{name:<10} {avg_length:<12.1f} {avg_diversity:<12.3f} {avg_repetition:<12.3f} {avg_reward:<12.4f}")
    
    # 打印结论
    print("\n" + "=" * 80)
    print("💡 结论分析")
    print("=" * 80)
    
    print("""
    SFT vs DPO vs PPO 对比：
    
    1. SFT (监督微调):
       - 只学习「如何回答」，没有学习「什么是好回答」
       - 可能生成流畅但不够有帮助的回答
       - 是 DPO/PPO 的基础
    
    2. DPO (直接偏好优化):
       - 直接从偏好数据学习
       - 无需训练奖励模型，更简单稳定
       - 可能过度优化某些表面特征
    
    3. PPO (强化学习):
       - 通过奖励模型获得反馈
       - 可以在线探索和改进
       - 训练更复杂，需要调参
       - 可能出现奖励 hacking
    
    理论上：PPO > DPO > SFT（在人类偏好上）
    实际上：DPO 通常与 PPO 效果相当，但更简单稳定
    """)


# ===== 7. 单独对比两个模型 =====
def compare_two_models(model1_path: str, model2_path: str, name1: str, name2: str):
    """详细对比两个模型"""
    print(f"\n{'=' * 80}")
    print(f"🔬 {name1} vs {name2} 详细对比")
    print("=" * 80)
    
    model1, tokenizer1 = load_model(model1_path, name1)
    model2, tokenizer2 = load_model(model2_path, name2)
    
    if not model1 or not model2:
        print("❌ 模型加载失败")
        return
    
    reward_model, rm_tokenizer = load_reward_model()
    
    for prompt in TEST_PROMPTS[:3]:  # 只测试前3个
        print(f"\n{'─' * 80}")
        print(f"问题: {prompt}")
        
        # Model 1
        response1 = generate_response(model1, tokenizer1, prompt)
        reward1 = compute_reward(reward_model, rm_tokenizer, prompt, response1) if reward_model else 0
        
        # Model 2
        response2 = generate_response(model2, tokenizer2, prompt)
        reward2 = compute_reward(reward_model, rm_tokenizer, prompt, response2) if reward_model else 0
        
        print(f"\n🔵 {name1}: (奖励: {reward1:.4f})")
        print(f"   {response1[:300]}")
        
        print(f"\n🟢 {name2}: (奖励: {reward2:.4f})")
        print(f"   {response2[:300]}")
        
        # 判断哪个更好
        if reward_model:
            winner = name1 if reward1 > reward2 else name2
            print(f"\n   👑 奖励模型判断: {winner} 更好 (差值: {abs(reward1-reward2):.4f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="对比 SFT/DPO/PPO 模型")
    parser.add_argument("--mode", choices=["all", "sft-dpo", "sft-ppo", "dpo-ppo"], 
                        default="all", help="对比模式")
    args = parser.parse_args()
    
    if args.mode == "all":
        run_comparison()
    elif args.mode == "sft-dpo":
        compare_two_models(SFT_MODEL_PATH, DPO_MODEL_PATH, "SFT", "DPO")
    elif args.mode == "sft-ppo":
        compare_two_models(SFT_MODEL_PATH, PPO_MODEL_PATH, "SFT", "PPO")
    elif args.mode == "dpo-ppo":
        compare_two_models(DPO_MODEL_PATH, PPO_MODEL_PATH, "DPO", "PPO")
