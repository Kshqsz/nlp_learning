# test_ppo.py
"""
测试 PPO 训练后的模型

功能：
1. 单独测试 PPO 模型
2. 交互式测试
3. 批量测试
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 配置 =====
PPO_MODEL_PATH = "./qwen_ppo"

GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

TEST_PROMPTS = [
    "请介绍一下人工智能",
    "如何学习编程？",
    "什么是机器学习？",
    "写一首关于秋天的诗",
    "如何保持健康的生活方式？",
]


def load_model(model_path: str):
    """加载模型"""
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return None, None
    
    print(f"加载模型 from {model_path}...")
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


def batch_test():
    """批量测试"""
    print("=" * 60)
    print("🧪 PPO 模型批量测试")
    print("=" * 60)
    
    model, tokenizer = load_model(PPO_MODEL_PATH)
    if not model:
        return
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 60}")
        print(f"问题 {i}: {prompt}")
        print("─" * 60)
        
        response = generate_response(model, tokenizer, prompt)
        print(f"回答: {response}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")


def interactive_test():
    """交互式测试"""
    print("=" * 60)
    print("💬 PPO 模型交互测试")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    model, tokenizer = load_model(PPO_MODEL_PATH)
    if not model:
        return
    
    while True:
        try:
            user_input = input("\n🙋 你: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print("👋 再见!")
                break
            
            response = generate_response(model, tokenizer, user_input)
            print(f"\n🤖 PPO模型: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 PPO 模型")
    parser.add_argument(
        "--mode",
        choices=["batch", "interactive"],
        default="batch",
        help="测试模式"
    )
    parser.add_argument(
        "--model-path",
        default="./qwen_ppo",
        help="PPO 模型路径"
    )
    
    args = parser.parse_args()
    PPO_MODEL_PATH = args.model_path
    
    if args.mode == "batch":
        batch_test()
    else:
        interactive_test()
