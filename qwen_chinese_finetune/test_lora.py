# test_lora.py
"""
测试 LoRA 微调后的模型

包含：
1. 加载 LoRA 权重（不合并）
2. 对比原始模型和 LoRA 模型的输出
3. 测试不同类型的问题
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== 配置 =====
BASE_MODEL_PATH = "Qwen/Qwen1.5-0.5B"    # 基座模型
LORA_PATH = "./qwen_lora_sft"            # LoRA 权重路径
# LORA_PATH = "./qwen_qlora_sft"         # 或 QLoRA 权重


def load_model_with_lora():
    """加载基座模型 + LoRA 权重"""
    print("📦 加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("🔧 加载 LoRA 权重...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_base_model():
    """只加载基座模型（对比用）"""
    print("📦 加载原始基座模型...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """生成回复"""
    # 构建对话格式
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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
    
    return response


def main():
    print("=" * 60)
    print("🧪 LoRA 模型测试")
    print("=" * 60)
    
    # 测试问题
    test_prompts = [
        "请用简短的话介绍一下人工智能",
        "写一首关于春天的诗",
        "Python 和 Java 有什么区别？",
        "如何保持健康的生活方式？",
        "帮我写一段故事开头，主角是一个机器人",
    ]
    
    # 加载 LoRA 模型
    print("\n" + "-" * 40)
    lora_model, lora_tokenizer = load_model_with_lora()
    print("✅ LoRA 模型加载完成")
    
    # 测试 LoRA 模型
    print("\n" + "=" * 60)
    print("📝 LoRA 模型回复")
    print("=" * 60)
    
    lora_responses = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n【问题 {i}】{prompt}")
        print("-" * 40)
        response = generate_response(lora_model, lora_tokenizer, prompt)
        print(f"【回答】{response}")
        lora_responses.append(response)
    
    # 释放显存
    del lora_model
    torch.cuda.empty_cache()
    
    # 加载原始模型对比
    print("\n" + "=" * 60)
    print("📝 原始模型回复（对比）")
    print("=" * 60)
    
    base_model, base_tokenizer = load_base_model()
    
    base_responses = []
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n【问题 {i}】{prompt}")
        print("-" * 40)
        response = generate_response(base_model, base_tokenizer, prompt)
        print(f"【回答】{response}")
        base_responses.append(response)
    
    # 对比总结
    print("\n" + "=" * 60)
    print("📊 对比总结")
    print("=" * 60)
    
    print("""
    | 模型      | 特点 |
    |-----------|------|
    | 原始模型  | 通用预训练，可能不擅长对话 |
    | LoRA 模型 | 经过指令微调，更擅长对话问答 |
    
    通过 LoRA，我们只训练了约 1% 的参数，
    但能让模型更好地遵循指令、产生有帮助的回复。
    """)


if __name__ == "__main__":
    main()
