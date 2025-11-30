# dpo_chinese_qwen.py
"""
DPO (Direct Preference Optimization) 训练脚本

使用真正的人类偏好数据集：shibing624/DPO-En-Zh-20k-Preference
该数据集包含：
- 10k 中文偏好对（来自 wenbopan/Chinese-dpo-pairs）
- 10k 英文偏好对（来自 argilla 高质量数据）

每条数据格式：
- system: 系统提示
- history: 多轮对话历史 [[user1, assistant1], [user2, assistant2], ...]
- question: 当前问题
- response_chosen: 被人类选中的好回答
- response_rejected: 被人类拒绝的差回答

这是真正的偏好数据，不是伪造的！
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import DPOTrainer, DPOConfig
import torch

# ===== 配置 =====
SFT_MODEL_PATH = "./qwen_sft"          # ← 你的 SFT 模型路径
OUTPUT_DIR = "./qwen_dpo"
MAX_LENGTH = 512
MAX_PROMPT_LENGTH = 256                # prompt 不能太长，留空间给回答
BATCH_SIZE = 2                         # DPO 显存占用高，建议 1~2
GRADIENT_ACCUMULATION_STEPS = 4        # 模拟更大 batch
LEARNING_RATE = 5e-6                   # DPO 学习率通常比 SFT 小
BETA = 0.1                             # DPO 的 beta 参数，控制偏离参考模型的程度
NUM_SAMPLES = 5000                     # 使用多少样本训练（中文共10k）

# ===== 1. 加载 tokenizer 和 SFT 模型 =====
print(f"Loading SFT model from: {SFT_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载参考模型（DPO 需要一个冻结的参考模型）
print("Loading reference model...")
ref_model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 修复 pad token（必须！）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# ===== 2. 加载真正的中文偏好数据集 =====
print("Loading Chinese preference dataset: shibing624/DPO-En-Zh-20k-Preference")

# 加载中文子集（zh），共10k样本
raw_dataset = load_dataset(
    "shibing624/DPO-En-Zh-20k-Preference",
    name="zh",                          # 中文子集
    split=f"train[:{NUM_SAMPLES}]"      # 取前N条
)

def format_preference_data(examples):
    """
    将原始偏好数据转换为 DPO 训练格式
    
    原始格式：
    - system: 系统提示（可能为空）
    - history: 多轮对话历史 [[user, assistant], ...]
    - question: 当前问题
    - response_chosen: 人类选择的好回答
    - response_rejected: 人类拒绝的差回答
    
    目标格式（DPO Trainer 需要）：
    - prompt: 完整的用户输入（包含历史对话）
    - chosen: 好回答
    - rejected: 差回答
    """
    prompts = []
    chosens = []
    rejecteds = []
    
    for system, history, question, chosen, rejected in zip(
        examples["system"],
        examples["history"],
        examples["question"],
        examples["response_chosen"],
        examples["response_rejected"]
    ):
        # 构建 prompt（使用 Qwen ChatML 格式）
        prompt_parts = []
        
        # 添加系统提示（如果有）
        if system and system.strip():
            prompt_parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        
        # 添加历史对话
        if history:
            for turn in history:
                if len(turn) >= 2:
                    user_msg, assistant_msg = turn[0], turn[1]
                    prompt_parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>")
                    prompt_parts.append(f"<|im_start|>assistant\n{assistant_msg}<|im_end|>")
        
        # 添加当前问题
        prompt_parts.append(f"<|im_start|>user\n{question}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        
        prompt = "\n".join(prompt_parts)
        
        # 回答部分（加上结束标记）
        chosen_response = chosen + "<|im_end|>"
        rejected_response = rejected + "<|im_end|>"
        
        prompts.append(prompt)
        chosens.append(chosen_response)
        rejecteds.append(rejected_response)
    
    return {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds
    }

print("Formatting preference data...")
dataset = raw_dataset.map(
    format_preference_data,
    batched=True,
    remove_columns=raw_dataset.column_names,
    desc="Formatting preference pairs"
)

print(f"✅ Loaded {len(dataset)} real human preference pairs!")

# 展示一个样本
print("\n===== 样本展示 =====")
print(f"Prompt:\n{dataset[0]['prompt'][:300]}...")
print(f"\nChosen (好回答):\n{dataset[0]['chosen'][:200]}...")
print(f"\nRejected (差回答):\n{dataset[0]['rejected'][:200]}...")

# ===== 3. DPO 训练配置 =====
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    beta=BETA,                          # DPO 核心参数
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    report_to="none",
    optim="adamw_torch",
    remove_unused_columns=False,
    gradient_checkpointing=True,        # 节省显存
)

# ===== 4. 创建 DPO Trainer =====
print("Creating DPO Trainer...")
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,                # 参考模型（冻结）
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ===== 5. 开始 DPO 训练 =====
print("🚀 Starting DPO training...")
print(f"   - Beta: {BETA}")
print(f"   - Learning Rate: {LEARNING_RATE}")
print(f"   - Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
dpo_trainer.train()

# ===== 6. 保存模型 =====
dpo_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ DPO 模型已保存到 {OUTPUT_DIR}")

# ===== 7. 测试推理 =====
print("\n===== 测试 DPO 模型 =====")
model.eval()

test_prompt = "<|im_start|>user\n请介绍一下人工智能<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"生成结果:\n{response}")