# test_sft.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 配置 =====
MODEL_PATH = "./qwen_sft"  # SFT 微调后的模型路径
# MODEL_PATH = "./qwen_pretrained"  # 或者测试预训练模型

print(f"Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== 测试函数 =====
def chat(instruction: str, max_new_tokens: int = 256):
    """使用 Qwen 对话格式进行推理"""
    prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
    
    # 解码并提取 assistant 回复
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取 assistant 部分
    if "<|im_start|>assistant\n" in full_response:
        response = full_response.split("<|im_start|>assistant\n")[-1]
        response = response.split("<|im_end|>")[0].strip()
    else:
        response = full_response[len(prompt):].strip()
    
    return response

# ===== 测试用例 =====
if __name__ == "__main__":
    test_questions = [
        "请介绍一下人工智能",
        "什么是机器学习？",
        "写一首关于春天的诗",
        "如何学习编程？",
        "解释一下什么是深度学习",
    ]
    
    print("=" * 60)
    print("🧪 开始测试 Qwen SFT 模型")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n【问题 {i}】{question}")
        print("-" * 40)
        response = chat(question)
        print(f"【回答】{response}")
        print("=" * 60)