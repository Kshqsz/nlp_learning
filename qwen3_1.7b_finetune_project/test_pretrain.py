# test_pretrain.py
"""
测试继续预训练后的模型

继续预训练的模型主要增强了：
- 领域知识（如中文维基百科知识）
- 语言建模能力

测试方式：给定开头，让模型续写
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 配置 =====
# 预训练后的模型路径
PRETRAIN_MODEL_PATH = "./qwen3_1.7b_pretrain"

# 原始模型路径（用于对比）
ORIGINAL_MODEL_PATH = "/public/huggingface-models/Qwen/Qwen3-1.7B"


def load_model(model_path):
    """加载模型"""
    print(f"📦 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=150):
    """
    文本续写（预训练模型的测试方式）
    直接给开头，让模型续写
    """
    inputs = tokenizer(prompt, return_tensors="pt")
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
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def test_model(model, tokenizer, model_name):
    """测试模型的续写能力"""
    print(f"\n{'='*60}")
    print(f"📝 {model_name} 续写测试")
    print("="*60)
    
    # 测试文本开头（适合预训练模型）
    test_prompts = [
        "人工智能是",
        "中国的首都北京是一座",
        "机器学习的主要方法包括",
        "自然语言处理技术可以用于",
        "深度学习在近年来取得了",
    ]
    
    for prompt in test_prompts:
        print(f"\n【开头】{prompt}")
        print("-" * 40)
        generated = generate_text(model, tokenizer, prompt)
        print(f"【续写】{generated}")


def compare_models():
    """对比原始模型和预训练后的模型"""
    print("=" * 60)
    print("🔬 预训练模型对比测试")
    print("=" * 60)
    
    # 测试预训练后的模型
    try:
        pretrain_model, pretrain_tokenizer = load_model(PRETRAIN_MODEL_PATH)
        test_model(pretrain_model, pretrain_tokenizer, "预训练后模型")
        
        # 释放显存
        del pretrain_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ 预训练模型加载失败: {e}")
        print("   请确认模型已训练完成并保存到对应路径")
    
    # 测试原始模型（对比）
    print("\n" + "=" * 60)
    print("📊 加载原始模型进行对比...")
    print("=" * 60)
    
    try:
        original_model, original_tokenizer = load_model(ORIGINAL_MODEL_PATH)
        test_model(original_model, original_tokenizer, "原始模型")
        
        del original_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ 原始模型加载失败: {e}")


def test_single_model():
    """只测试预训练后的模型"""
    print("=" * 60)
    print("🧪 预训练模型测试")
    print("=" * 60)
    
    model, tokenizer = load_model(PRETRAIN_MODEL_PATH)
    test_model(model, tokenizer, "预训练后模型")
    
    # 交互式测试
    print("\n" + "=" * 60)
    print("💬 交互式测试（输入 q 退出）")
    print("=" * 60)
    
    while True:
        prompt = input("\n请输入文本开头: ").strip()
        if prompt.lower() == 'q':
            break
        if not prompt:
            continue
        
        generated = generate_text(model, tokenizer, prompt)
        print(f"【续写】{generated}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # 对比模式
        compare_models()
    else:
        # 单独测试
        test_single_model()
