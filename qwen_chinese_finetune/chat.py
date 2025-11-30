# chat.py - 交互式对话机器人
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 配置 =====
# MODEL_PATH = "./qwen_pretrained"  # 预训练模型
# MODEL_PATH = "./qwen_sft"         # SFT 模型
MODEL_PATH = "./qwen_dpo"           # DPO 模型

# ===== 加载模型 =====
print(f"🔄 正在加载模型: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ 模型加载完成！设备: {model.device}")
print("=" * 60)

# ===== 生成配置 =====
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}

# ===== 对话历史 =====
conversation_history = []

def build_prompt(user_input: str, history: list) -> str:
    """构建包含历史的对话 prompt"""
    prompt = ""
    
    # 添加历史对话
    for user_msg, assistant_msg in history:
        prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
    
    # 添加当前用户输入
    prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    return prompt

def chat(user_input: str, history: list) -> str:
    """生成回复"""
    prompt = build_prompt(user_input, history)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **GENERATION_CONFIG)
    
    # 解码完整输出
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取最后一个 assistant 回复
    if "<|im_start|>assistant\n" in full_response:
        response = full_response.split("<|im_start|>assistant\n")[-1]
        # 去掉结束标记
        if "<|im_end|>" in response:
            response = response.split("<|im_end|}")[0]
        response = response.strip()
    else:
        # fallback: 直接截取新生成的部分
        response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    return response

def main():
    global conversation_history
    
    print("🤖 Qwen 中文对话机器人")
    print("=" * 60)
    print("命令说明:")
    print("  - 输入问题即可对话")
    print("  - 输入 'clear' 清空对话历史")
    print("  - 输入 'history' 查看对话历史")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            
            if not user_input:
                continue
            
            # 特殊命令
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 再见！")
                break
            
            if user_input.lower() == "clear":
                conversation_history = []
                print("🗑️ 对话历史已清空")
                continue
            
            if user_input.lower() == "history":
                if not conversation_history:
                    print("📭 暂无对话历史")
                else:
                    print("\n📜 对话历史:")
                    for i, (u, a) in enumerate(conversation_history, 1):
                        print(f"  [{i}] 👤: {u[:50]}{'...' if len(u) > 50 else ''}")
                        print(f"      🤖: {a[:50]}{'...' if len(a) > 50 else ''}")
                continue
            
            # 生成回复
            print("🤖 思考中...", end="", flush=True)
            response = chat(user_input, conversation_history)
            print("\r" + " " * 20 + "\r", end="")  # 清除 "思考中..."
            
            print(f"🤖 助手: {response}")
            
            # 保存到历史（最多保留最近 5 轮）
            conversation_history.append((user_input, response))
            if len(conversation_history) > 5:
                conversation_history = conversation_history[-5:]
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()
