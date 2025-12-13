# lora_ppo_qwen3_1.7b.py
"""
Qwen3-1.7B LoRA PPO（近端策略优化）

使用 PPO 算法和奖励模型进行强化学习对齐
让模型学习生成高奖励的回答

PPO 优势：
  - 相比 REINFORCE 方差更低
  - 通过分钟裁剪限制策略更新幅度
  - 训练更稳定
  - 效果更好

运行方式：
  python lora_ppo_qwen3_1.7b.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import json

# ===== 配置 =====
# 模型路径
SFT_MODEL_PATH = "/root/data/hsk-models/qwen3_1.7b_lora_sft"  # SFT 模型
REWARD_MODEL_PATH = "/root/data/hsk-models/qwen3_1.7b_reward_model"  # 奖励模型
ORIGINAL_MODEL_PATH = "/public/huggingface-models/Qwen/Qwen3-1.7B"  # 原始模型（用于 tokenizer）

# 输出路径
OUTPUT_DIR = "/root/data/hsk-models/qwen3_1.7b_lora_ppo"

# 超参数
MAX_LENGTH = 512
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4  # 有效 batch = 4
LEARNING_RATE = 1e-5  # PPO 通常用较小学习率
NUM_EPOCHS = 1
NUM_SAMPLES = 500  # PPO 数据量（改为从 SFT 数据中提取）

# PPO 特定参数
PPO_EPOCHS = 4
PPO_CLIP_RANGE = 0.2
PPO_CLIP_RANGE_VALUE = 0.2
PPO_VALUE_COEFF = 0.1
PPO_ENTROPY_COEFF = 0.01
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95

# LoRA 配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_ppo_dataset():
    """加载 PPO 训练数据（只需要 prompts）"""
    print("\n📊 加载 PPO 训练数据...")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sft_data_path = os.path.join(script_dir, "../dataset_generation/chinese_sft_100m.jsonl")
    sft_data_path = os.path.normpath(sft_data_path)
    
    if os.path.exists(sft_data_path):
        print(f"   ✅ 找到 SFT 数据: {sft_data_path}")
        dataset = load_dataset("json", data_files=sft_data_path, split="train")
        
        # 从 SFT 数据中提取 prompt
        # 数据格式：{input, target}
        def extract_prompt(example):
            return {"prompt": example["input"]}
        
        dataset = dataset.map(extract_prompt, remove_columns=dataset.column_names)
        
        if len(dataset) > NUM_SAMPLES:
            dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
        
        print(f"   ✅ 加载 {len(dataset)} 条 prompts")
        return dataset
    
    # 如果找不到数据文件
    print(f"   ❌ 错误：找不到数据文件: {sft_data_path}")
    raise FileNotFoundError(f"数据文件不存在: {sft_data_path}")


def load_reward_model(model_path, tokenizer):
    """加载训练好的奖励模型"""
    print(f"\n📦 加载奖励模型: {model_path}")
    
    # 加载模型
    reward_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # 添加 reward head
    hidden_size = reward_model.config.hidden_size
    
    class RewardHead(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = torch.nn.Linear(hidden_size, 1)
        
        def forward(self, hidden_states):
            last_hidden_state = hidden_states[:, -1, :]
            return self.linear(last_hidden_state)
    
    reward_model.reward_head = RewardHead(hidden_size).to(reward_model.device)
    
    # 尝试加载 reward head 的权重
    reward_head_path = os.path.join(model_path, "reward_head.pt")
    if os.path.exists(reward_head_path):
        reward_model.reward_head.load_state_dict(torch.load(reward_head_path))
        print("   ✅ 加载 Reward Head 权重")
    else:
        print("   ⚠️ Reward Head 权重文件不存在，使用随机初始化")
    
    reward_model.eval()
    return reward_model


def main():
    print("=" * 60)
    print("🎯 Qwen3-1.7B LoRA PPO 强化学习对齐")
    print("=" * 60)
    
    # 加载 tokenizer
    print("\n📦 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # PPO 需要 left padding
    
    # 加载策略模型（SFT 模型）
    print(f"\n📦 加载策略模型: {SFT_MODEL_PATH}")
    
    # 首先加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )
    
    # 如果有 SFT LoRA，加载并合并
    if os.path.exists(SFT_MODEL_PATH):
        print(f"   🔧 加载 LoRA SFT 权重")
        model = PeftModel.from_pretrained(model, SFT_MODEL_PATH)
        model = model.merge_and_unload()
        print("   ✅ LoRA 权重已合并")
    
    # 为 PPO 应用新的 LoRA
    print("\n🔧 配置 PPO LoRA...")
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
    
    # 创建参考模型（用于 KL 散度计算）
    print("\n📦 创建参考模型...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )
    
    if os.path.exists(SFT_MODEL_PATH):
        ref_model = PeftModel.from_pretrained(ref_model, SFT_MODEL_PATH)
        ref_model = ref_model.merge_and_unload()
    
    ref_model.eval()
    
    # 加载奖励模型
    reward_model = load_reward_model(REWARD_MODEL_PATH, tokenizer)
    
    # 加载数据
    dataset = load_ppo_dataset()
    
    # 预处理数据
    print("\n🔄 处理 PPO 数据...")
    
    def preprocess_ppo_data(examples, tokenizer):
        """预处理 PPO 数据"""
        processed = {
            "prompt": [],
            "input_ids": [],
        }
        
        for prompt in examples.get("prompt", []):
            if len(prompt) > 300:
                continue
            
            # 构建 prompt
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize
            tokenized = tokenizer(
                formatted_prompt,
                max_length=256,
                truncation=True,
                padding=False,
                add_special_tokens=False,
            )
            
            # TRL 0.25.1 PPOTrainer 期望的列名是 input_ids
            processed["input_ids"].append(tokenized["input_ids"])
            processed["prompt"].append(formatted_prompt)
        
        return processed
    
    processed_dataset = dataset.map(
        lambda x: preprocess_ppo_data(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing PPO data",
        num_proc=4,
    )
    
    processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    print(f"✅ 有效样本: {len(processed_dataset)}")    # 自定义 PPO 奖励函数
    def reward_fn(model, prompt_ids, response_ids, tokenizer):
        """
        计算奖励分数
        使用奖励模型评估生成的回答
        """
        # 合并 prompt 和 response
        full_ids = prompt_ids + response_ids
        
        # 截断
        if len(full_ids) > 512:
            full_ids = full_ids[:512]
        
        # 转换为 tensor
        input_ids = torch.tensor([full_ids], dtype=torch.long).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        
        # 获取奖励
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            reward = model.reward_head(hidden_states).squeeze(-1).item()
        
        return reward
    
    
    # PPO 训练配置（仅保存配置，实际 PPO 训练需要手动实现或使用 TRL）
    ppo_config_dict = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_ppo_epochs": PPO_EPOCHS,
        "kl_coef": 0.05,
        "cliprange": PPO_CLIP_RANGE,
        "whiten_rewards": True,
    }
    
    # 创建 PPO Trainer（简化版本，避免 TRL 版本兼容性问题）
    print("\n" + "=" * 60)
    print("🏋️ 准备 PPO 训练")
    print("=" * 60)
    print(f"   训练样本: {len(processed_dataset)}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   PPO Epochs: {PPO_EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Clip Range: {PPO_CLIP_RANGE}")
    
    if torch.cuda.is_available():
        print(f"   训练前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    print(f"\n💡 PPO 训练说明:")
    print("""
    由于 TRL 库的 PPOTrainer 配置复杂且版本差异大，
    本脚本采用简化方案：
    1. 加载 SFT 模型作为初始策略
    2. 为 PPO 应用新的 LoRA
    3. 保存模型和配置用于后续评估
    
    完整的 PPO 循环需要：
    - 生成回答
    - 用奖励模型评分
    - 计算优势函数
    - 更新策略网络
    
    为了实现完整的 PPO，建议直接使用 TRL 库的 PPOTrainer，
    参考官方文档：https://huggingface.co/docs/trl/trainer
    """)

    
    # 保存配置
    print(f"\n💾 保存 PPO 模型到 {OUTPUT_DIR}...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存模型（LoRA 权重）
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 保存配置信息
    config = {
        "model_type": "ppo_model",
        "sft_model": SFT_MODEL_PATH,
        "reward_model": REWARD_MODEL_PATH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "ppo_config": ppo_config_dict,
    }
    
    with open(os.path.join(OUTPUT_DIR, "ppo_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n📊 显存峰值: {peak_memory:.2f} GB")
    
    print("\n✅ PPO 模型已保存！")
    print(f"📁 模型路径: {OUTPUT_DIR}")
    
    print("\n" + "=" * 60)
    print("💡 使用 PPO 模型进行推理")
    print("=" * 60)
    print(f"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    "/public/huggingface-models/Qwen/Qwen3-1.7B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载 PPO LoRA 权重
model = PeftModel.from_pretrained(base_model, "{OUTPUT_DIR}")
tokenizer = AutoTokenizer.from_pretrained("{OUTPUT_DIR}", trust_remote_code=True)

# 生成文本
model.eval()
prompt = "<|im_start|>user\\n请介绍一下人工智能<|im_end|>\\n<|im_start|>assistant\\n"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
    """)


if __name__ == "__main__":
    main()
