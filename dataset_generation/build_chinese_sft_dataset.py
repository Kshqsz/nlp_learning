# build_chinese_sft_dataset.py
"""
生成高质量中文 SFT 混合数据集（约 100MB）
混合来源：
  - Firefly (通用任务)
  - BelleGroup/train_2M_CN (指令跟随)
  - Chinese-Alpaca-2 (基础指令)
  - CMMLU 指令化 (学术/推理，提升 C-Eval)

输出：chinese_sft_100m.jsonl （每行一个 {"input": "...", "target": "..."}）
"""

import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

# 设置 HF 镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 输出路径
OUTPUT_FILE = "chinese_sft_100m.jsonl"
TARGET_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# 采样比例（可调整）
CONFIG = {
    "firefly": {"weight": 0.35, "max_samples": 40000},
    "belle": {"weight": 0.30, "max_samples": 35000},
    "alpaca": {"weight": 0.20, "max_samples": 20000},
    "cmmlu": {"weight": 0.15, "max_samples": 15000},
}

def format_sample(source, item):
    """统一格式为 {'input': str, 'target': str}"""
    if source == "firefly":
        return {"input": item["input"], "target": item["target"]}
    
    elif source == "belle":
        # Belle: 可能是多轮对话或单轮
        if "conversations" in item:
            # 多轮：取第一轮 user + 第一轮 assistant
            user_msg = next((msg["value"] for msg in item["conversations"] if msg["from"] == "human"), "")
            asst_msg = next((msg["value"] for msg in item["conversations"] if msg["from"] == "gpt"), "")
            return {"input": user_msg, "target": asst_msg}
        else:
            return {"input": item.get("instruction", "") + "\n" + item.get("input", ""), 
                    "target": item["output"]}
    
    elif source == "alpaca":
        inp = item["instruction"]
        if item.get("input", "").strip():
            inp += "\n" + item["input"]
        return {"input": inp, "target": item["output"]}
    
    elif source == "cmmlu":
        return {"input": item["question"], "target": item["answer_text"]}
    else:
        return None

def load_cmmlu_sft(max_samples=15000):
    """加载并指令化 CMMLU 数据（仅部分学科）"""
    print("📥 加载 CMMLU 并转换为 SFT 格式...")
    
    # 选择对 C-Eval 影响大的学科
    subjects = [
        "high_school_physics", "high_school_chemistry", "high_school_biology",
        "college_physics", "college_chemistry", "college_biology",
        "chinese_history", "world_history", "high_school_geography",
        "high_school_mathematics", "college_mathematics", "economics",
        "law", "computer_science"
    ]
    
    all_items = []
    for subject in tqdm(subjects, desc="Processing CMMLU subjects"):
        try:
            ds = load_dataset("hails/cmmlu", subject, split="test")
        except Exception as e:
            print(f"⚠️ 跳过 {subject}: {e}")
            continue
        
        for row in ds:
            if len(all_items) >= max_samples:
                break
            choices = [row["A"], row["B"], row["C"], row["D"]]
            options = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            question = f"【学科】{subject}\n【问题】{row['Question']}\n{options}\n\n请仔细分析并选出唯一正确答案。"
            answer = row["Answer"]
            explanation = f"正确答案是：{answer}"
            all_items.append({
                "question": question,
                "answer_text": explanation
            })
        if len(all_items) >= max_samples:
            break
    
    random.shuffle(all_items)
    return all_items[:max_samples]

def main():
    random.seed(42)
    
    print("=" * 60)
    print("🚀 构建高质量中文 SFT 混合数据集（目标 ≈100MB）")
    print("=" * 60)
    
    all_samples = []

    # 1. Firefly
    print("\n📥 加载 Firefly...")
    firefly = load_dataset("YeungNLP/firefly-train-1.1M", split="train")
    firefly_samples = [format_sample("firefly", item) for item in 
                       tqdm(random.sample(list(firefly), CONFIG["firefly"]["max_samples"]))]
    all_samples.extend(firefly_samples)

    # 2. Belle 2M CN
    print("\n📥 加载 BelleGroup/train_2M_CN...")
    belle = load_dataset("BelleGroup/train_2M_CN", split="train")
    belle_samples = [format_sample("belle", item) for item in 
                     tqdm(random.sample(list(belle), CONFIG["belle"]["max_samples"]))]
    all_samples.extend(belle_samples)

    # 3. Chinese-Alpaca-2
    print("\n📥 加载 Chinese-Alpaca-2...")
    alpaca = load_dataset("silk-road/alpaca-data-gpt4-chinese", split="train")
    alpaca_samples = [format_sample("alpaca", item) for item in 
                      tqdm(random.sample(list(alpaca), min(CONFIG["alpaca"]["max_samples"], len(alpaca))))]
    all_samples.extend(alpaca_samples)

    # 4. CMMLU (指令化)
    print("\n📥 构建 CMMLU-SFT...")
    cmmlu_samples = load_cmmlu_sft(CONFIG["cmmlu"]["max_samples"])
    cmmlu_formatted = [format_sample("cmmlu", item) for item in cmmlu_samples]
    all_samples.extend(cmmlu_formatted)

    # 打乱顺序
    random.shuffle(all_samples)
    
    # 写入文件并控制大小
    print(f"\n💾 写入到 {OUTPUT_FILE} ...")
    current_size = 0
    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in tqdm(all_samples, desc="Writing samples"):
            if sample is None or not sample.get("input") or not sample.get("target"):
                continue
            line = json.dumps(sample, ensure_ascii=False) + "\n"
            if current_size + len(line.encode("utf-8")) > TARGET_SIZE_BYTES:
                break
            f.write(line)
            current_size += len(line.encode("utf-8"))
            count += 1

    print(f"\n✅ 完成！共写入 {count} 条样本，文件大小: {current_size / 1024 / 1024:.1f} MB")
    print(f"📁 输出文件: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()