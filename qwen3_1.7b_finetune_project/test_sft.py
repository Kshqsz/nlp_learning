# test_sft.py
"""
测试 LoRA SFT 后的模型

SFT 后的模型主要增强了：
- 指令遵循能力
- 对话能力
- 回答问题的能力

测试方式：使用对话格式提问
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== 配置 =====
# 基座模型路径（预训练后的模型或原始模型）
BASE_MODEL_PATH = "./qwen3_1.7b_pretrain"
# BASE_MODEL_PATH = "/public/huggingface-models/Qwen/Qwen3-1.7B"

# LoRA SFT 权重路径
LORA_SFT_PATH = "./qwen3_1.7b_lora_sft"


def load_sft_model():
    """加载 LoRA SFT 模型"""
    print("📦 加载基座模型...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_SFT_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("🔧 加载 LoRA 权重...")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_SFT_PATH,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    
    return model, tokenizer


def load_base_model():
    """加载基座模型（用于对比）"""
    print("📦 加载基座模型（无 LoRA）...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    return model, tokenizer


def chat(model, tokenizer, question, max_new_tokens=256):
    """
    对话生成（SFT 模型的测试方式）
    使用 ChatML 格式
    """
    # 构建对话格式
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
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
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取 assistant 回复
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    # 清理结束标记
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()
    
    return response


def test_model(model, tokenizer, model_name):
    """测试模型的对话能力"""
    print(f"\n{'='*60}")
    print(f"💬 {model_name} 对话测试")
    print("="*60)
    
    # 测试问题（适合 SFT 模型）
    test_questions = [
        "请简单介绍一下人工智能",
        "如何学习编程？给我一些建议",
        "写一首关于春天的短诗",
        "Python 有哪些优点？",
        "请解释什么是机器学习",
        "帮我写一段自我介绍，我是一名大学生",
    ]
    
    for question in test_questions:
        print(f"\n【问题】{question}")
        print("-" * 40)
        response = chat(model, tokenizer, question)
        print(f"【回答】{response[:500]}")  # 截断显示


def compare_models():
    """对比基座模型和 SFT 模型"""
    print("=" * 60)
    print("🔬 SFT 模型对比测试")
    print("=" * 60)
    
    test_questions = [
        "请介绍一下北京",
        "如何保持健康？",
        "写一个简短的故事开头",
    ]
    
    # 测试 SFT 模型
    print("\n📦 加载 LoRA SFT 模型...")
    try:
        sft_model, sft_tokenizer = load_sft_model()
        
        print("\n" + "=" * 60)
        print("💬 LoRA SFT 模型回复")
        print("=" * 60)
        
        for question in test_questions:
            print(f"\n【问题】{question}")
            response = chat(sft_model, sft_tokenizer, question)
            print(f"【SFT 回答】{response[:300]}")
        
        del sft_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"⚠️ SFT 模型加载失败: {e}")
        print("   请确认 LoRA SFT 训练已完成")
        return
    
    # 测试基座模型（对比）
    print("\n" + "=" * 60)
    print("📊 基座模型回复（对比）")
    print("=" * 60)
    
    try:
        base_model, base_tokenizer = load_base_model()
        
        for question in test_questions:
            print(f"\n【问题】{question}")
            response = chat(base_model, base_tokenizer, question)
            print(f"【基座回答】{response[:300]}")
        
        del base_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"⚠️ 基座模型加载失败: {e}")
    
    # 对比总结
    print("\n" + "=" * 60)
    print("📊 对比总结")
    print("=" * 60)
    print("""
    | 模型      | 特点 |
    |-----------|------|
    | 基座模型  | 续写能力强，但可能不擅长对话 |
    | SFT 模型  | 更好地遵循指令，回答更有帮助 |
    
    通过 LoRA SFT，模型学会了：
    1. 理解用户的问题意图
    2. 按照指令格式回答
    3. 生成更有帮助、更相关的回复
    """)


def interactive_chat():
    """交互式对话"""
    print("=" * 60)
    print("💬 交互式对话测试")
    print("=" * 60)
    
    model, tokenizer = load_sft_model()
    
    print("\n✅ 模型加载完成，开始对话（输入 q 退出）\n")
    
    while True:
        question = input("你: ").strip()
        if question.lower() == 'q':
            print("再见！")
            break
        if not question:
            continue
        
        response = chat(model, tokenizer, question)
        print(f"AI: {response}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # 对比模式
            compare_models()
        elif sys.argv[1] == "--chat":
            # 交互对话
            interactive_chat()
    else:
        # 默认：测试 SFT 模型
        try:
            model, tokenizer = load_sft_model()
            test_model(model, tokenizer, "LoRA SFT 模型")
            
            # 进入交互模式
            print("\n" + "=" * 60)
            print("💬 进入交互模式（输入 q 退出）")
            print("=" * 60)
            
            while True:
                question = input("\n你: ").strip()
                if question.lower() == 'q':
                    break
                if not question:
                    continue
                response = chat(model, tokenizer, question)
                print(f"AI: {response}")
                
        except Exception as e:
            print(f"⚠️ 错误: {e}")
            print("请确认 LoRA SFT 训练已完成")
