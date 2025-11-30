# test_dpo.py
"""
DPO 模型测试脚本

功能：
1. 对比测试 SFT 模型 vs DPO 模型
2. 使用相同的 prompt 生成回答，观察 DPO 对齐效果
3. 支持自定义测试问题
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 配置 =====
SFT_MODEL_PATH = "./qwen_sft"    # SFT 模型路径
DPO_MODEL_PATH = "./qwen_dpo"    # DPO 模型路径

# 生成参数
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

# 测试问题列表
TEST_PROMPTS = [
    "请介绍一下人工智能",
    "如何学习编程？",
    "写一首关于春天的诗",
    "解释一下什么是机器学习",
    "如何保持健康的生活方式？",
    "请用简单的语言解释量子力学",
]


def build_prompt(user_input: str, system: str = None) -> str:
    """构建 Qwen ChatML 格式的 prompt"""
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def load_model(model_path: str):
    """加载模型和 tokenizer"""
    print(f"Loading model from: {model_path}")
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


def generate_response(model, tokenizer, prompt: str) -> str:
    """生成回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **GENERATION_CONFIG
        )
    
    # 只取新生成的部分
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 清理结束标记
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    return response.strip()


def compare_models():
    """对比 SFT 和 DPO 模型"""
    print("=" * 60)
    print("🔬 SFT vs DPO 模型对比测试")
    print("=" * 60)
    
    # 检查模型是否存在
    if not os.path.exists(SFT_MODEL_PATH):
        print(f"❌ SFT 模型不存在: {SFT_MODEL_PATH}")
        print("   请先运行 sft_chinese_qwen.py 训练 SFT 模型")
        return
    
    if not os.path.exists(DPO_MODEL_PATH):
        print(f"❌ DPO 模型不存在: {DPO_MODEL_PATH}")
        print("   请先运行 dpo_chinese_qwen.py 训练 DPO 模型")
        return
    
    # 加载两个模型
    print("\n📦 加载模型...")
    sft_model, sft_tokenizer = load_model(SFT_MODEL_PATH)
    dpo_model, dpo_tokenizer = load_model(DPO_MODEL_PATH)
    
    # 对比测试
    print("\n" + "=" * 60)
    print("📝 开始对比测试")
    print("=" * 60)
    
    for i, question in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 60}")
        print(f"问题 {i}: {question}")
        print("─" * 60)
        
        prompt = build_prompt(question)
        
        # SFT 模型回答
        sft_response = generate_response(sft_model, sft_tokenizer, prompt)
        print(f"\n🔵 SFT 模型回答:")
        print(f"   {sft_response[:500]}{'...' if len(sft_response) > 500 else ''}")
        
        # DPO 模型回答
        dpo_response = generate_response(dpo_model, dpo_tokenizer, prompt)
        print(f"\n🟢 DPO 模型回答:")
        print(f"   {dpo_response[:500]}{'...' if len(dpo_response) > 500 else ''}")
    
    print("\n" + "=" * 60)
    print("✅ 对比测试完成!")
    print("=" * 60)


def test_dpo_only():
    """只测试 DPO 模型"""
    print("=" * 60)
    print("🚀 DPO 模型测试")
    print("=" * 60)
    
    if not os.path.exists(DPO_MODEL_PATH):
        print(f"❌ DPO 模型不存在: {DPO_MODEL_PATH}")
        return
    
    model, tokenizer = load_model(DPO_MODEL_PATH)
    
    for i, question in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 60}")
        print(f"问题 {i}: {question}")
        print("─" * 60)
        
        prompt = build_prompt(question)
        response = generate_response(model, tokenizer, prompt)
        print(f"回答: {response}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")


def interactive_test():
    """交互式测试"""
    print("=" * 60)
    print("💬 DPO 模型交互测试")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'compare' 切换到对比模式")
    print("=" * 60)
    
    if not os.path.exists(DPO_MODEL_PATH):
        print(f"❌ DPO 模型不存在: {DPO_MODEL_PATH}")
        return
    
    model, tokenizer = load_model(DPO_MODEL_PATH)
    
    while True:
        try:
            user_input = input("\n🙋 你: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("👋 再见!")
                break
            if user_input.lower() == "compare":
                compare_models()
                continue
            
            prompt = build_prompt(user_input)
            response = generate_response(model, tokenizer, prompt)
            print(f"\n🤖 DPO模型: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DPO 模型测试脚本")
    parser.add_argument(
        "--mode", 
        choices=["compare", "dpo", "interactive"],
        default="compare",
        help="测试模式: compare(对比SFT和DPO), dpo(只测DPO), interactive(交互式)"
    )
    parser.add_argument(
        "--sft-path",
        default="./qwen_sft",
        help="SFT 模型路径"
    )
    parser.add_argument(
        "--dpo-path",
        default="./qwen_dpo",
        help="DPO 模型路径"
    )
    
    args = parser.parse_args()
    
    # 更新路径
    SFT_MODEL_PATH = args.sft_path
    DPO_MODEL_PATH = args.dpo_path
    
    if args.mode == "compare":
        compare_models()
    elif args.mode == "dpo":
        test_dpo_only()
    else:
        interactive_test()
