# lora_dpo_qwen3_1.7b.py
"""
Qwen3-1.7B LoRA DPO（直接偏好优化）

在 LoRA SFT 模型基础上进行 DPO 偏好对齐
让模型学会生成人类更偏好的回答

DPO 优势：
  - 不需要训练奖励模型
  - 不需要 PPO 强化学习
  - 直接从偏好数据学习
  - 训练稳定，效果好

运行方式：
  python lora_dpo_qwen3_1.7b.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import DPOTrainer, DPOConfig

# ===== 配置 =====
# 原始模型路径（用于加载 tokenizer）
ORIGINAL_MODEL_PATH = "/public/huggingface-models/Qwen/Qwen3-1.7B"

# SFT 模型路径（DPO 的基座）
# 方式1：使用预训练 + LoRA SFT 的模型
BASE_MODEL_PATH = "/root/data/hsk-models/qwen3_1.7b_pretrain"  # 预训练后的模型
LORA_SFT_PATH = "/root/data/hsk-models/qwen3_1.7b_lora_sft"    # LoRA SFT 权重

# 如果没有预训练，可以直接用原始模型 + LoRA SFT
# BASE_MODEL_PATH = ORIGINAL_MODEL_PATH
# LORA_SFT_PATH = "/root/data/hsk-models/qwen3_1.7b_lora_sft"

OUTPUT_DIR = "/root/data/hsk-models/qwen3_1.7b_lora_dpo"
MAX_LENGTH = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4  # 有效 batch = 8
LEARNING_RATE = 5e-5             # DPO 通常用较小学习率
NUM_EPOCHS = 1
NUM_SAMPLES = 5000               # DPO 数据量

# LoRA 配置（与 SFT 保持一致）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# DPO 配置
DPO_BETA = 0.1  # KL 散度惩罚系数


def load_model_and_tokenizer():
    """加载 SFT 后的模型作为 DPO 的基座"""
    print("📦 加载模型...")
    
    # 从原始模型加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO 需要 left padding
    
    # 检查基座模型
    if os.path.exists(BASE_MODEL_PATH) and os.path.exists(os.path.join(BASE_MODEL_PATH, "config.json")):
        model_path = BASE_MODEL_PATH
        print(f"   ✅ 使用预训练模型: {model_path}")
    else:
        model_path = ORIGINAL_MODEL_PATH
        print(f"   ⚠️ 预训练模型不存在，使用原始模型: {model_path}")
    
    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.cuda()
    
    # 检查是否有 SFT LoRA 权重
    if os.path.exists(LORA_SFT_PATH):
        print(f"   🔧 加载 LoRA SFT 权重: {LORA_SFT_PATH}")
        # 加载 SFT LoRA 并合并到基座模型
        model = PeftModel.from_pretrained(model, LORA_SFT_PATH)
        model = model.merge_and_unload()  # 合并 LoRA 到基座
        print("   ✅ LoRA 权重已合并")
    else:
        print(f"   ⚠️ LoRA SFT 权重不存在: {LORA_SFT_PATH}")
        print("   将直接在基座模型上进行 DPO")
    
    return model, tokenizer


def load_dpo_dataset(tokenizer):
    """加载 DPO 偏好数据集"""
    print("\n📊 加载 DPO 数据...")
    
    # 尝试加载不同的偏好数据集
    dataset = None
    
    # 方式1：尝试加载本地数据
    local_dpo_path = "./dpo_zh.jsonl"
    if os.path.exists(local_dpo_path):
        print(f"   使用本地数据: {local_dpo_path}")
        dataset = load_dataset("json", data_files=local_dpo_path, split="train")
    
    # 方式2：尝试加载 HuggingFace 数据集
    if dataset is None:
        try:
            print("   尝试加载 HuggingFace 偏好数据集...")
            # 中文偏好数据集
            dataset = load_dataset(
                "beyond/rlhf-reward-single-round-trans_chinese",
                split="train",
                trust_remote_code=True
            )
            print("   ✅ 加载 rlhf-reward-single-round-trans_chinese")
        except Exception as e:
            print(f"   ⚠️ 加载失败: {e}")
    
    # 方式3：尝试另一个数据集
    if dataset is None:
        try:
            dataset = load_dataset(
                "Anthropic/hh-rlhf",
                split="train",
                trust_remote_code=True
            )
            print("   ✅ 加载 Anthropic/hh-rlhf")
        except Exception as e:
            print(f"   ⚠️ 加载失败: {e}")
    
    if dataset is None:
        raise ValueError("无法加载偏好数据集，请准备本地数据文件 dpo_data.jsonl")
    
    # 采样
    if len(dataset) > NUM_SAMPLES:
        dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
    
    print(f"   ✅ 加载 {len(dataset)} 条数据")
    return dataset


def preprocess_dpo_data(examples, tokenizer):
    """
    预处理 DPO 数据
    DPO 需要三元组：(prompt, chosen, rejected)
    """
    processed = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    
    # 根据数据集格式处理
    # 格式1: 已有 prompt, chosen, rejected 字段
    if "prompt" in examples and "chosen" in examples and "rejected" in examples:
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            if len(prompt) > 300 or len(chosen) > 400 or len(rejected) > 400:
                continue
            processed["prompt"].append(prompt)
            processed["chosen"].append(chosen)
            processed["rejected"].append(rejected)
    
    # 格式2: rlhf-reward 格式 (prompt, response, label)
    elif "prompt" in examples and "response" in examples:
        # 需要成对处理，这里简化处理
        prompts = examples.get("prompt", [])
        responses = examples.get("response", [])
        labels = examples.get("label", [])
        
        # 按 prompt 分组
        prompt_responses = {}
        for p, r, l in zip(prompts, responses, labels):
            if p not in prompt_responses:
                prompt_responses[p] = {"chosen": None, "rejected": None}
            if l == 1:
                prompt_responses[p]["chosen"] = r
            else:
                prompt_responses[p]["rejected"] = r
        
        for p, resp in prompt_responses.items():
            if resp["chosen"] and resp["rejected"]:
                if len(p) > 300:
                    continue
                processed["prompt"].append(p)
                processed["chosen"].append(resp["chosen"])
                processed["rejected"].append(resp["rejected"])
    
    # 格式3: hh-rlhf 格式
    elif "chosen" in examples and "rejected" in examples:
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            # 从 chosen/rejected 中提取 prompt
            # hh-rlhf 格式: "Human: xxx\n\nAssistant: xxx"
            if "Human:" in chosen and "Assistant:" in chosen:
                parts = chosen.split("Assistant:")
                if len(parts) >= 2:
                    prompt = parts[0].replace("Human:", "").strip()
                    chosen_resp = parts[-1].strip()
                    
                    rej_parts = rejected.split("Assistant:")
                    rejected_resp = rej_parts[-1].strip() if len(rej_parts) >= 2 else rejected
                    
                    if len(prompt) > 300:
                        continue
                    
                    processed["prompt"].append(prompt)
                    processed["chosen"].append(chosen_resp)
                    processed["rejected"].append(rejected_resp)
    
    return processed


def format_for_dpo(example):
    """
    将数据格式化为 DPO Trainer 需要的格式
    使用 ChatML 对话格式
    """
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    
    # 构建 ChatML 格式
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    formatted_chosen = f"{chosen}<|im_end|>"
    formatted_rejected = f"{rejected}<|im_end|>"
    
    return {
        "prompt": formatted_prompt,
        "chosen": formatted_chosen,
        "rejected": formatted_rejected,
    }


def main():
    print("=" * 60)
    print("🚀 Qwen3-1.7B LoRA DPO 偏好对齐")
    print("=" * 60)
    print(f"基座模型: {BASE_MODEL_PATH}")
    print(f"SFT LoRA: {LORA_SFT_PATH}")
    print(f"DPO Beta: {DPO_BETA}")
    print(f"LoRA Rank: {LORA_R}, Alpha: {LORA_ALPHA}")
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer()
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n基座模型参数量: {total_params / 1e9:.2f}B")
    
    # 配置新的 LoRA（用于 DPO）
    print("\n🔧 配置 DPO LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据
    raw_dataset = load_dpo_dataset(tokenizer)
    
    # 预处理数据
    print("\n🔄 处理 DPO 数据...")
    dataset = raw_dataset.map(
        lambda x: preprocess_dpo_data(x, tokenizer),
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Preprocessing",
        num_proc=4,
    )
    
    # 过滤空样本
    dataset = dataset.filter(lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0)
    print(f"✅ 有效样本: {len(dataset)}")
    
    # 格式化为 ChatML
    dataset = dataset.map(format_for_dpo, desc="Formatting")
    
    # 打印样本示例
    if len(dataset) > 0:
        print("\n📝 数据样例:")
        print(f"   Prompt: {dataset[0]['prompt'][:100]}...")
        print(f"   Chosen: {dataset[0]['chosen'][:100]}...")
        print(f"   Rejected: {dataset[0]['rejected'][:100]}...")
    
    # DPO 训练配置
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        
        # 训练参数
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        
        # 学习率
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # DPO 参数
        beta=DPO_BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=256,
        
        # 显存优化
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # 日志和保存
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        
        # 其他
        report_to="none",
        remove_unused_columns=False,
    )
    
    # 创建 DPO Trainer
    print("\n" + "=" * 60)
    print("🏋️ 开始 DPO 训练")
    print("=" * 60)
    print(f"   训练样本: {len(dataset)}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   有效 Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   DPO Beta: {DPO_BETA}")
    print(f"   Epochs: {NUM_EPOCHS}")
    
    if torch.cuda.is_available():
        print(f"   训练前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # 创建参考模型（用于计算 KL 散度）
    # DPOTrainer 会自动处理参考模型
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 使用 model 的副本作为参考模型
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # 训练
    trainer.train()
    
    # 保存
    print(f"\n💾 保存 LoRA 权重到 {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n📊 显存峰值: {peak_memory:.2f} GB")
    
    print("\n✅ LoRA DPO 训练完成！")
    print(f"📁 权重已保存到: {OUTPUT_DIR}")
    
    # 测试
    print("\n" + "=" * 60)
    print("🧪 测试 DPO 模型")
    print("=" * 60)
    
    test_questions = [
        "如何保持健康的生活方式？",
        "请介绍一下人工智能的应用",
        "学习编程有什么好的建议？",
    ]
    
    model.eval()
    for question in test_questions:
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(f"\n【问题】{question}")
        print(f"【回答】{response[:300]}...")
    
    print("\n" + "=" * 60)
    print("💡 后续步骤")
    print("=" * 60)
    print("""
1. 测试 DPO 模型效果:
   python test_dpo.py

2. 对比 SFT 和 DPO 模型:
   python eval_benchmarks.py --compare

3. 如需合并所有 LoRA 权重:
   # SFT LoRA + DPO LoRA 可以通过多次 merge 合并
""")


if __name__ == "__main__":
    main()
