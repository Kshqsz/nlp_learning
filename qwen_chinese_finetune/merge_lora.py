# merge_lora.py
"""
合并 LoRA 权重到基座模型

为什么要合并：
1. 部署更简单：只需要一个模型文件
2. 推理更快：不需要额外的 LoRA 前向计算
3. 兼容性好：合并后就是普通的 Transformers 模型

合并公式：
W_merged = W_base + ΔW = W_base + (BA * scaling)
其中 scaling = lora_alpha / lora_r

本脚本将 LoRA 权重合并到基座模型，输出完整模型
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import shutil

# ===== 配置 =====
BASE_MODEL_PATH = "Qwen/Qwen1.5-0.5B"    # 基座模型
LORA_PATH = "./qwen_lora_sft"            # LoRA 权重
MERGED_OUTPUT_PATH = "./qwen_lora_merged"   # 合并后的输出路径


def merge_and_save():
    """合并 LoRA 权重并保存"""
    print("=" * 60)
    print("🔀 合并 LoRA 权重到基座模型")
    print("=" * 60)
    
    # 1. 加载基座模型
    print("\n📦 加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 2. 加载 LoRA 权重
    print("🔧 加载 LoRA 权重...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        torch_dtype=torch.bfloat16,
    )
    
    # 3. 合并权重
    print("🔀 合并权重...")
    # merge_and_unload() 会：
    # - 将 LoRA 权重合并到基座模型
    # - 移除 LoRA 层，恢复原始结构
    merged_model = model.merge_and_unload()
    
    # 4. 保存合并后的模型
    print(f"\n💾 保存合并后的模型到 {MERGED_OUTPUT_PATH}...")
    merged_model.save_pretrained(MERGED_OUTPUT_PATH)
    
    # 5. 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_OUTPUT_PATH)
    
    print("\n✅ 合并完成！")
    
    # 检查文件大小对比
    print("\n📊 文件大小对比:")
    
    # LoRA 权重大小
    lora_size = get_folder_size(LORA_PATH)
    print(f"   LoRA 权重: {lora_size:.2f} MB")
    
    # 合并后模型大小
    merged_size = get_folder_size(MERGED_OUTPUT_PATH)
    print(f"   合并后模型: {merged_size:.2f} MB")
    
    print(f"""
🎯 说明：
   - LoRA 权重很小（通常只有几十 MB）
   - 合并后模型大小 ≈ 基座模型大小
   
   使用场景：
   - 开发/实验阶段：使用 LoRA 权重（方便切换、节省存储）
   - 部署阶段：使用合并后模型（推理更快、更简单）
""")
    
    return merged_model, tokenizer


def get_folder_size(path):
    """获取文件夹大小（MB）"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)


def test_merged_model(model, tokenizer):
    """测试合并后的模型"""
    print("\n" + "=" * 60)
    print("🧪 测试合并后的模型")
    print("=" * 60)
    
    test_prompts = [
        "请介绍一下北京",
        "如何学习编程？",
    ]
    
    for prompt in test_prompts:
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        print(f"\n【问题】{prompt}")
        print(f"【回答】{response}")


def compare_lora_vs_merged():
    """
    对比 LoRA 推理和合并模型推理
    验证两者输出是否一致
    """
    print("\n" + "=" * 60)
    print("🔍 验证 LoRA 和合并模型输出一致性")
    print("=" * 60)
    
    # 加载合并后的模型
    print("\n加载合并后模型...")
    merged_model = AutoModelForCausalLM.from_pretrained(
        MERGED_OUTPUT_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载 LoRA 模型
    print("加载 LoRA 模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT_PATH, trust_remote_code=True)
    
    # 测试输入
    test_input = "你好"
    full_prompt = f"<|im_start|>user\n{test_input}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(merged_model.device) for k, v in inputs.items()}
    
    # 获取 logits（不使用采样，确保确定性）
    with torch.no_grad():
        merged_logits = merged_model(**inputs).logits
        lora_logits = lora_model(**inputs).logits
    
    # 计算差异
    diff = (merged_logits - lora_logits).abs().mean().item()
    print(f"\n平均 logits 差异: {diff:.6f}")
    
    if diff < 1e-4:
        print("✅ 验证通过：LoRA 和合并模型输出基本一致")
    else:
        print("⚠️ 存在差异，可能是数值精度问题")


if __name__ == "__main__":
    # 合并并保存
    merged_model, tokenizer = merge_and_save()
    
    # 测试合并后的模型
    test_merged_model(merged_model, tokenizer)
    
    # 可选：验证一致性
    # compare_lora_vs_merged()
