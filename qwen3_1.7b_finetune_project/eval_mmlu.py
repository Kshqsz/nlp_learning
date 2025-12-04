# eval_mmlu.py
"""
MMLU (Massive Multitask Language Understanding) 评测脚本

MMLU 包含 57 个学科的选择题，测试模型的知识和推理能力
学科涵盖：STEM、人文、社会科学、其他（法律、医学等）

评测方式：
  - 给模型一个多选题（A/B/C/D）
  - 模型输出答案，计算准确率

使用方式：
  python eval_mmlu.py                          # 评测预训练模型
  python eval_mmlu.py --model sft              # 评测 LoRA SFT 模型
  python eval_mmlu.py --model original         # 评测原始模型
  python eval_mmlu.py --subjects all           # 评测所有学科（慢）
  python eval_mmlu.py --subjects stem          # 只评测 STEM 学科
  python eval_mmlu.py --num_samples 100        # 每个学科采样数量
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import defaultdict

# ===== 配置 =====
ORIGINAL_MODEL_PATH = "/public/huggingface-models/Qwen/Qwen3-1.7B"
PRETRAIN_MODEL_PATH = "./qwen3_1.7b_pretrain"
LORA_SFT_PATH = "./qwen3_1.7b_lora_sft"

# MMLU 学科分类
MMLU_SUBJECTS = {
    "stem": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning"
    ],
    "humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "social_sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy"
    ],
    "other": [
        "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
        "human_aging", "management", "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine", "virology"
    ]
}

# 快速测试用的代表性学科
QUICK_SUBJECTS = [
    "high_school_mathematics",
    "high_school_physics", 
    "high_school_computer_science",
    "college_computer_science",
    "machine_learning",
    "logical_fallacies",
    "world_religions",
    "high_school_geography",
    "marketing",
    "clinical_knowledge",
]


def load_model(model_type="pretrain"):
    """加载模型"""
    print(f"\n📦 加载模型: {model_type}")
    
    # 确定 tokenizer 路径
    tokenizer_path = ORIGINAL_MODEL_PATH
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_type == "original":
        model_path = ORIGINAL_MODEL_PATH
        print(f"   使用原始模型: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif model_type == "pretrain":
        model_path = PRETRAIN_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"   ⚠️ 预训练模型不存在，使用原始模型")
            model_path = ORIGINAL_MODEL_PATH
        print(f"   使用预训练模型: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    elif model_type == "sft":
        # 加载 LoRA SFT 模型
        base_path = PRETRAIN_MODEL_PATH if os.path.exists(PRETRAIN_MODEL_PATH) else ORIGINAL_MODEL_PATH
        print(f"   基座模型: {base_path}")
        print(f"   LoRA 权重: {LORA_SFT_PATH}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, LORA_SFT_PATH)
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    model.eval()
    return model, tokenizer


def format_mmlu_prompt(question, choices, few_shot_examples=None):
    """
    构建 MMLU 评测 prompt
    
    格式：
    Question: xxx
    A. xxx
    B. xxx
    C. xxx
    D. xxx
    Answer:
    """
    prompt = ""
    
    # 添加 few-shot 示例（可选）
    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Question: {ex['question']}\n"
            for i, choice in enumerate(ex['choices']):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += f"Answer: {chr(65 + ex['answer'])}\n\n"
    
    # 添加当前问题
    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    
    return prompt


def get_model_answer(model, tokenizer, prompt):
    """
    获取模型的答案（A/B/C/D）
    使用 logits 比较方法，更准确
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # 最后一个 token 的 logits
    
    # 获取 A, B, C, D 对应的 token id
    choices = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]
    
    # 比较这四个选项的 logits
    choice_logits = [logits[cid].item() for cid in choice_ids]
    predicted_idx = choice_logits.index(max(choice_logits))
    
    return predicted_idx


def evaluate_subject(model, tokenizer, subject, num_samples=None, num_few_shot=5):
    """评测单个学科"""
    try:
        # 加载数据集
        dataset = load_dataset("cais/mmlu", subject, split="test")
        dev_dataset = load_dataset("cais/mmlu", subject, split="dev")
    except Exception as e:
        print(f"   ⚠️ 加载 {subject} 失败: {e}")
        return None, 0
    
    # 准备 few-shot 示例
    few_shot_examples = []
    for i in range(min(num_few_shot, len(dev_dataset))):
        few_shot_examples.append({
            "question": dev_dataset[i]["question"],
            "choices": dev_dataset[i]["choices"],
            "answer": dev_dataset[i]["answer"]
        })
    
    # 采样测试数据
    if num_samples and num_samples < len(dataset):
        indices = list(range(min(num_samples, len(dataset))))
        dataset = dataset.select(indices)
    
    correct = 0
    total = len(dataset)
    
    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]  # 0, 1, 2, 3 对应 A, B, C, D
        
        prompt = format_mmlu_prompt(question, choices, few_shot_examples)
        predicted = get_model_answer(model, tokenizer, prompt)
        
        if predicted == answer:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, total


def evaluate_mmlu(model, tokenizer, subjects="quick", num_samples=100):
    """评测 MMLU"""
    print("\n" + "=" * 60)
    print("📊 MMLU 评测")
    print("=" * 60)
    
    # 确定要评测的学科
    if subjects == "all":
        subject_list = []
        for cat_subjects in MMLU_SUBJECTS.values():
            subject_list.extend(cat_subjects)
    elif subjects == "quick":
        subject_list = QUICK_SUBJECTS
    elif subjects in MMLU_SUBJECTS:
        subject_list = MMLU_SUBJECTS[subjects]
    else:
        subject_list = [subjects]  # 单个学科
    
    print(f"评测学科数: {len(subject_list)}")
    print(f"每学科采样: {num_samples if num_samples else '全部'}")
    
    results = {}
    category_results = defaultdict(list)
    
    for subject in tqdm(subject_list, desc="评测进度"):
        accuracy, total = evaluate_subject(model, tokenizer, subject, num_samples)
        if accuracy is not None:
            results[subject] = {"accuracy": accuracy, "total": total}
            
            # 按类别统计
            for cat, cat_subjects in MMLU_SUBJECTS.items():
                if subject in cat_subjects:
                    category_results[cat].append(accuracy)
                    break
    
    # 打印结果
    print("\n" + "-" * 60)
    print("📈 各学科准确率")
    print("-" * 60)
    
    for subject, result in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        acc = result["accuracy"] * 100
        total = result["total"]
        print(f"  {subject:40s}: {acc:5.1f}% ({total} samples)")
    
    # 分类别统计
    print("\n" + "-" * 60)
    print("📊 各类别平均准确率")
    print("-" * 60)
    
    total_acc = []
    for cat, accs in category_results.items():
        if accs:
            avg_acc = sum(accs) / len(accs) * 100
            total_acc.extend(accs)
            print(f"  {cat:20s}: {avg_acc:5.1f}% ({len(accs)} subjects)")
    
    # 总体准确率
    if total_acc:
        overall_acc = sum(total_acc) / len(total_acc) * 100
        print("\n" + "=" * 60)
        print(f"🎯 MMLU 总体准确率: {overall_acc:.1f}%")
        print("=" * 60)
    
    return results, overall_acc if total_acc else 0


def compare_models(subjects="quick", num_samples=50):
    """对比不同模型的 MMLU 分数"""
    print("=" * 60)
    print("🔬 MMLU 模型对比评测")
    print("=" * 60)
    
    results = {}
    
    # 评测原始模型
    print("\n" + "=" * 60)
    print("1️⃣ 原始模型")
    print("=" * 60)
    try:
        model, tokenizer = load_model("original")
        _, acc = evaluate_mmlu(model, tokenizer, subjects, num_samples)
        results["original"] = acc
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ 原始模型评测失败: {e}")
    
    # 评测预训练模型
    print("\n" + "=" * 60)
    print("2️⃣ 预训练模型")
    print("=" * 60)
    try:
        model, tokenizer = load_model("pretrain")
        _, acc = evaluate_mmlu(model, tokenizer, subjects, num_samples)
        results["pretrain"] = acc
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ 预训练模型评测失败: {e}")
    
    # 评测 SFT 模型
    print("\n" + "=" * 60)
    print("3️⃣ LoRA SFT 模型")
    print("=" * 60)
    try:
        model, tokenizer = load_model("sft")
        _, acc = evaluate_mmlu(model, tokenizer, subjects, num_samples)
        results["sft"] = acc
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ SFT 模型评测失败: {e}")
    
    # 打印对比结果
    print("\n" + "=" * 60)
    print("📊 MMLU 对比结果")
    print("=" * 60)
    print(f"{'模型':<15} {'MMLU 准确率':>15}")
    print("-" * 30)
    for model_name, acc in results.items():
        print(f"{model_name:<15} {acc:>14.1f}%")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMLU 评测")
    parser.add_argument("--model", type=str, default="pretrain",
                        choices=["original", "pretrain", "sft"],
                        help="要评测的模型")
    parser.add_argument("--subjects", type=str, default="quick",
                        choices=["all", "quick", "stem", "humanities", "social_sciences", "other"],
                        help="要评测的学科")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="每个学科的采样数量")
    parser.add_argument("--compare", action="store_true",
                        help="对比所有模型")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.subjects, args.num_samples)
    else:
        model, tokenizer = load_model(args.model)
        evaluate_mmlu(model, tokenizer, args.subjects, args.num_samples)
