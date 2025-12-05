# lora_sft_qwen3_1.7b.py
"""
Qwen3-1.7B LoRA SFT（监督微调）

在继续预训练的模型基础上，使用 LoRA 进行监督微调
让模型学会遵循指令、进行对话

LoRA 优势：
  - 只训练 ~1% 的参数
  - 显存占用小，不需要 CPU Offload
  - 训练速度快（~2-3s/step）
  - 效果接近全量微

运行方式：
  python lora_sft_qwen3_1.7b.py
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# ===== 配置 =====
# 使用继续预训练后的模型作为基座
BASE_MODEL_PATH = "/root/data/hsk-models/qwen3_1.7b_pretrain"  # 预训练后的模型
# 如果预训练还没完成，可以先用原始模型
# BASE_MODEL_PATH = "/public/huggingface-models/Qwen/Qwen3-1.7B"
OUTPUT_DIR = "/root/data/hsk-models/qwen3_1.7b_lora_sft"
MAX_LENGTH = 512

# ===== 超参数配置（已优化）=====
BATCH_SIZE = 4               # 减小 batch size，增加更新次数
GRADIENT_ACCUMULATION_STEPS = 8  # 有效 batch = 32
LEARNING_RATE = 1e-4         # LoRA 微调推荐更低学习率
NUM_EPOCHS = 3               # 增加训练轮次
NUM_SAMPLES = 10000          # SFT 数据量
WARMUP_STEPS = 100           # 固定 warmup 步数

# LoRA 配置（增强表达能力）
LORA_R = 64                  # 增大 LoRA 秩，提升表达能力
LORA_ALPHA = 128             # alpha 通常设为 2 * r
LORA_DROPOUT = 0.1           # 增加 dropout 防过拟合
TARGET_MODULES = [          # Qwen3 的目标模块
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ===== 1. 加载模型和 Tokenizer =====
print("=" * 60)
print("🚀 Qwen3-1.7B LoRA SFT 微调")
print("=" * 60)
print(f"基座模型: {BASE_MODEL_PATH}")
print(f"LoRA Rank: {LORA_R}, Alpha: {LORA_ALPHA}")

print("\n📦 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False,  # 训练时禁用 KV cache
)

# 打印原始模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"基座模型参数量: {total_params / 1e9:.2f}B")


# ===== 2. 配置 LoRA =====
print("\n🔧 配置 LoRA...")

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

# 启用 gradient checkpointing（必须在应用 LoRA 之后）
model.enable_input_require_grads()  # 关键：让输入需要梯度

# 打印 LoRA 信息
model.print_trainable_parameters()


# ===== 3. 加载 SFT 数据集 =====
print("\n📊 加载 SFT 数据...")

# 从本地 JSONL 文件加载数据
# 数据格式: {"input": "用户输入", "target": "模型回复"}
DATA_PATH = "./chinese_sft_100m.jsonl"

raw_dataset = load_dataset(
    "json",
    data_files=DATA_PATH,
    split="train"
)

# 如果数据量大于 NUM_SAMPLES，只取前 NUM_SAMPLES 条
if len(raw_dataset) > NUM_SAMPLES:
    raw_dataset = raw_dataset.select(range(NUM_SAMPLES))

print(f"✅ 加载 {len(raw_dataset)} 条数据")


# ===== 4. 数据预处理 =====
def preprocess_function(examples):
    """
    SFT 数据处理：构建对话格式
    只对 assistant 的回复计算 loss
    
    数据格式: {"input": "用户输入", "target": "模型回复"}
    关键修复：使用相同的 tokenize 参数确保长度一致
    """
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for inp, target in zip(examples["input"], examples["target"]):
        # 跳过过长的样本
        if len(inp) > 800 or len(target) > 800:
            continue
        
        # 构建 Qwen 对话格式 (ChatML)
        prompt = f"<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n"
        response = f"{target}<|im_end|>"
        full_text = prompt + response
        
        # 分别 tokenize prompt 和完整文本（使用相同的参数！）
        prompt_ids = tokenizer(
            prompt,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # 不添加特殊 token
        )["input_ids"]
        
        full_ids = tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
            add_special_tokens=False,  # 保持一致
        )["input_ids"]
        
        # 构建 labels：只对 response 部分计算 loss
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        
        # 确保长度完全一致
        assert len(labels) == len(full_ids), f"Length mismatch: labels={len(labels)}, input_ids={len(full_ids)}"
        
        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attention_mask_list.append([1] * len(full_ids))
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_mask_list,
    }


print("\n🔄 处理数据...")
dataset = raw_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_dataset.column_names,
    desc="Processing SFT data",
    num_proc=4,
)

# 过滤空样本
dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
print(f"✅ 处理完成，有效样本: {len(dataset)}")

# 打印几个样本进行验证
print("\n📋 数据样本验证:")
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    labels = sample["labels"]
    non_ignore = [l for l in labels if l != -100]
    print(f"  样本 {i}: input_ids 长度={len(sample['input_ids'])}, "
          f"有效 labels 数量={len(non_ignore)}, "
          f"比例={len(non_ignore)/len(sample['input_ids'])*100:.1f}%")


# ===== 5. 训练配置 =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    
    # 训练参数
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    
    # 学习率（优化后）
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,  # 使用固定步数而非比例
    weight_decay=0.01,  # 添加权重衰减
    max_grad_norm=1.0,  # 梯度裁剪
    
    # 显存优化
    bf16=True,
    gradient_checkpointing=True,  # 启用梯度检查点
    gradient_checkpointing_kwargs={"use_reentrant": False},  # 推荐设置
    optim="adamw_torch",
    
    # 日志和保存
    logging_steps=10,  # 更频繁记录
    save_steps=200,    # 更频繁保存
    save_total_limit=3,
    
    # 评估
    eval_strategy="no",  # 如有验证集可改为 "steps"
    
    # 其他
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,  # 保留所有列，避免数据问题
    dataloader_pin_memory=True,
    seed=42,  # 固定随机种子
)


# ===== 6. 自定义 Data Collator =====
class SFTDataCollator:
    """
    自定义 Data Collator，正确处理 labels 的 padding
    - input_ids 用 pad_token_id padding
    - labels 用 -100 padding（不计算 loss）
    """
    def __init__(self, tokenizer, padding_side="right"):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
    
    def __call__(self, features):
        # 获取最大长度
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            attention_mask = feature["attention_mask"]
            
            # 计算 padding 长度
            padding_length = max_length - len(input_ids)
            
            if self.padding_side == "right":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                labels = labels + [-100] * padding_length  # labels 用 -100 padding
                attention_mask = attention_mask + [0] * padding_length
            else:
                input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                labels = [-100] * padding_length + labels
                attention_mask = [0] * padding_length + attention_mask
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


# ===== 7. 开始训练 =====
print("\n" + "=" * 60)
print("🏋️ 开始 LoRA SFT 训练")
print("=" * 60)
print(f"   训练样本: {len(dataset)}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
print(f"   有效 Batch: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   LoRA Rank: {LORA_R}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   预计总步数: {len(dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS}")

if torch.cuda.is_available():
    print(f"   训练前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 使用自定义 Data Collator
data_collator = SFTDataCollator(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 训练
trainer.train()


# ===== 8. 保存 LoRA 权重 =====
print(f"\n💾 保存 LoRA 权重到 {OUTPUT_DIR}...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n📊 显存峰值: {peak_memory:.2f} GB")

print("\n✅ LoRA SFT 训练完成！")
print(f"📁 LoRA 权重已保存到: {OUTPUT_DIR}")

# ===== 9. 测试对话效果 =====
print("\n" + "=" * 60)
print("🧪 测试对话效果")
print("=" * 60)

# 测试问题
test_questions = [
    "请介绍一下人工智能的发展历史",
    "如何学习编程？",
    "写一首关于春天的诗",
    "Python 和 Java 有什么区别？",
]

# 测试时启用 cache
model.config.use_cache = True
model.eval()

for question in test_questions:
    # 构建对话格式
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
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
    
    # 提取 assistant 回复
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    print(f"\n【问题】{question}")
    print(f"【回答】{response[:300]}...")  # 截断显示


print("\n" + "=" * 60)
print("💡 后续步骤")
print("=" * 60)
print("""
1. 如需合并 LoRA 权重到基座模型，运行:
   from peft import PeftModel
   merged = model.merge_and_unload()
   merged.save_pretrained("./qwen3_1.7b_sft_merged")

2. 如需进一步进行 DPO/RLHF 对齐，可基于此 LoRA 模型继续训练
""")

