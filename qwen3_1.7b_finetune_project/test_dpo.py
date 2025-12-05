# test_dpo.py
"""
测试 LoRA DPO 后的模型

DPO 后的模型主要增强了：
- 回答质量（更符合人类偏好）
- 安全性和有益性
- 减少有害/不准确内容

测试方式：对比 SFT 和 DPO 模型的回答质量
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== 配置 =====
ORIGINAL_MODEL_PATH = "/public/huggingface-models/Qwen/Qwen3-1.7B"
BASE_MODEL_PATH = "./qwen3_1.7b_pretrain"
LORA_SFT_PATH = "./qwen3_1.7b_lora_sft"
LORA_DPO_PATH = "./qwen3_1.7b_lora_dpo"


def load_tokenizer():
    """加载 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model():
    """加载基座模型"""
    if os.path.exists(BASE_MODEL_PATH):
        model_path = BASE_MODEL_PATH
    else:
        model_path = ORIGINAL_MODEL_PATH
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model


def load_sft_model(tokenizer):
    """加载 SFT 模型"""
    print("📦 加载 SFT 模型...")
    base_model = load_base_model()
    
    if os.path.exists(LORA_SFT_PATH):
        model = PeftModel.from_pretrained(base_model, LORA_SFT_PATH)
        print(f"   ✅ 加载 LoRA SFT: {LORA_SFT_PATH}")
    else:
        model = base_model
        print("   ⚠️ SFT 权重不存在，使用基座模型")
    
    model.eval()
    return model


def load_dpo_model(tokenizer):
    """加载 DPO 模型"""
    print("📦 加载 DPO 模型...")
    
    # 先加载基座 + SFT
    base_model = load_base_model()
    
    if os.path.exists(LORA_SFT_PATH):
        model = PeftModel.from_pretrained(base_model, LORA_SFT_PATH)
        model = model.merge_and_unload()
        print(f"   ✅ 合并 SFT LoRA")
    else:
        model = base_model
    
    # 再加载 DPO LoRA
    if os.path.exists(LORA_DPO_PATH):
        model = PeftModel.from_pretrained(model, LORA_DPO_PATH)
        print(f"   ✅ 加载 DPO LoRA: {LORA_DPO_PATH}")
    else:
        raise FileNotFoundError(f"DPO 权重不存在: {LORA_DPO_PATH}")
    
    model.eval()
    return model


def chat(model, tokenizer, question, max_new_tokens=256):
    """对话生成"""
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
    
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0].strip()
    
    return response


def compare_sft_dpo():
    """对比 SFT 和 DPO 模型"""
    print("=" * 70)
    print("🔬 SFT vs DPO 模型对比测试")
    print("=" * 70)
    
    tokenizer = load_tokenizer()
    
    # 测试问题（包含一些可能有偏好差异的问题）
    test_questions = [
        # 一般知识问题
        "请简单介绍一下机器学习",
        "如何保持健康的生活习惯？",
        
        # 建议类问题
        "我想学习编程，应该从哪里开始？",
        "如何提高工作效率？",
        
        # 可能涉及偏好的问题
        "如何看待人工智能的发展？",
        "请给我一些时间管理的建议",
        
        # 创意类问题
        "写一个关于友谊的短句",
    ]
    
    # 加载 SFT 模型并测试
    print("\n" + "=" * 70)
    print("📝 SFT 模型回答")
    print("=" * 70)
    
    sft_responses = {}
    try:
        sft_model = load_sft_model(tokenizer)
        for q in test_questions:
            resp = chat(sft_model, tokenizer, q)
            sft_responses[q] = resp
            print(f"\n【问题】{q}")
            print(f"【SFT】{resp[:300]}")
        
        del sft_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ SFT 模型加载失败: {e}")
    
    # 加载 DPO 模型并测试
    print("\n" + "=" * 70)
    print("📝 DPO 模型回答")
    print("=" * 70)
    
    dpo_responses = {}
    try:
        dpo_model = load_dpo_model(tokenizer)
        for q in test_questions:
            resp = chat(dpo_model, tokenizer, q)
            dpo_responses[q] = resp
            print(f"\n【问题】{q}")
            print(f"【DPO】{resp[:300]}")
        
        del dpo_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ DPO 模型加载失败: {e}")
        return
    
    # 并排对比
    print("\n" + "=" * 70)
    print("📊 并排对比")
    print("=" * 70)
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"【问题】{q}")
        print("-" * 60)
        if q in sft_responses:
            print(f"【SFT】{sft_responses[q][:200]}...")
        print("-" * 60)
        if q in dpo_responses:
            print(f"【DPO】{dpo_responses[q][:200]}...")
    
    # 总结
    print("\n" + "=" * 70)
    print("📈 对比分析")
    print("=" * 70)
    print("""
    DPO 训练后，模型应该表现出：
    
    1. ✅ 更有帮助的回答（直接回答问题）
    2. ✅ 更安全的内容（避免有害建议）
    3. ✅ 更好的格式（清晰、有条理）
    4. ✅ 更符合人类偏好的语气
    
    如果效果不明显，可以尝试：
    - 增加 DPO 训练数据量
    - 调整 beta 参数（增大会更保守）
    - 增加训练轮数
    """)


def interactive_test():
    """交互式测试 DPO 模型"""
    print("=" * 60)
    print("💬 DPO 模型交互测试")
    print("=" * 60)
    
    tokenizer = load_tokenizer()
    model = load_dpo_model(tokenizer)
    
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


def test_dpo_only():
    """只测试 DPO 模型"""
    print("=" * 60)
    print("🧪 DPO 模型测试")
    print("=" * 60)
    
    tokenizer = load_tokenizer()
    model = load_dpo_model(tokenizer)
    
    test_questions = [
        "请介绍一下人工智能",
        "如何学习编程？",
        "写一首关于春天的短诗",
        "Python 有哪些优点？",
        "如何保持健康？",
    ]
    
    for q in test_questions:
        print(f"\n【问题】{q}")
        print("-" * 40)
        resp = chat(model, tokenizer, q)
        print(f"【回答】{resp}")
    
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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            compare_sft_dpo()
        elif sys.argv[1] == "--chat":
            interactive_test()
    else:
        # 默认：测试 + 对比
        try:
            test_dpo_only()
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
            print("请先运行 DPO 训练: python lora_dpo_qwen3_1.7b.py")
