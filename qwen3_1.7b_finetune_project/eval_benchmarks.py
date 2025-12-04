# eval_benchmarks.py
"""
大模型综合评测脚本

包含多个常用基准测试：
  - MMLU: 多任务语言理解（57个学科）
  - C-Eval: 中文知识评测
  - HellaSwag: 常识推理
  - ARC: 科学推理
  - TruthfulQA: 真实性评测
  - WinoGrande: 常识推理（代词消歧）

使用方式：
  python eval_benchmarks.py                    # 评测预训练模型
  python eval_benchmarks.py --model sft        # 评测 LoRA SFT 模型
  python eval_benchmarks.py --compare          # 对比所有模型
  python eval_benchmarks.py --quick            # 快速模式（每个基准少量采样）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import argparse
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import defaultdict
from datetime import datetime

# ===== 配置 =====
ORIGINAL_MODEL_PATH = "/public/huggingface-models/Qwen/Qwen3-1.7B"
PRETRAIN_MODEL_PATH = "./qwen3_1.7b_pretrain"
LORA_SFT_PATH = "./qwen3_1.7b_lora_sft"

# 快速测试的 MMLU 学科
QUICK_MMLU_SUBJECTS = [
    "high_school_mathematics",
    "high_school_physics", 
    "high_school_computer_science",
    "machine_learning",
    "logical_fallacies",
]

# 快速测试的 C-Eval 学科
QUICK_CEVAL_SUBJECTS = [
    "computer_network",
    "operating_system",
    "discrete_mathematics",
    "high_school_physics",
    "high_school_chemistry",
]


class ModelLoader:
    """模型加载器"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_type = None
    
    def load(self, model_type="pretrain"):
        """加载模型"""
        if self.current_model_type == model_type and self.model is not None:
            return self.model, self.tokenizer
        
        # 释放之前的模型
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        
        print(f"\n📦 加载模型: {model_type}")
        
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if model_type == "original":
            model_path = ORIGINAL_MODEL_PATH
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="auto"
            )
        elif model_type == "pretrain":
            model_path = PRETRAIN_MODEL_PATH if os.path.exists(PRETRAIN_MODEL_PATH) else ORIGINAL_MODEL_PATH
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="auto"
            )
        elif model_type == "sft":
            base_path = PRETRAIN_MODEL_PATH if os.path.exists(PRETRAIN_MODEL_PATH) else ORIGINAL_MODEL_PATH
            base_model = AutoModelForCausalLM.from_pretrained(
                base_path, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, LORA_SFT_PATH)
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.current_model_type = model_type
        
        return model, tokenizer


def get_logits_for_choices(model, tokenizer, prompt, choices):
    """获取各选项的 logits 分数"""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    choice_scores = []
    for choice in choices:
        # 获取选项首字母的 token id
        token_id = tokenizer.encode(choice, add_special_tokens=False)[0]
        choice_scores.append(logits[token_id].item())
    
    return choice_scores


def evaluate_mmlu(model, tokenizer, num_samples=100, subjects=None):
    """
    MMLU 评测 (Massive Multitask Language Understanding)
    英文多任务知识评测，57个学科
    """
    print("\n📊 评测 MMLU...")
    
    if subjects is None:
        subjects = QUICK_MMLU_SUBJECTS
    
    all_correct = 0
    all_total = 0
    subject_results = {}
    
    for subject in tqdm(subjects, desc="MMLU"):
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
            dev_set = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=True)
        except Exception as e:
            print(f"   跳过 {subject}: {e}")
            continue
        
        # Few-shot 示例
        few_shot = ""
        for i in range(min(5, len(dev_set))):
            item = dev_set[i]
            few_shot += f"Question: {item['question']}\n"
            for j, c in enumerate(item['choices']):
                few_shot += f"{chr(65+j)}. {c}\n"
            few_shot += f"Answer: {chr(65 + item['answer'])}\n\n"
        
        # 评测
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        correct = 0
        for item in dataset:
            prompt = few_shot + f"Question: {item['question']}\n"
            for j, c in enumerate(item['choices']):
                prompt += f"{chr(65+j)}. {c}\n"
            prompt += "Answer:"
            
            scores = get_logits_for_choices(model, tokenizer, prompt, ["A", "B", "C", "D"])
            pred = scores.index(max(scores))
            if pred == item['answer']:
                correct += 1
        
        acc = correct / len(dataset) if len(dataset) > 0 else 0
        subject_results[subject] = acc
        all_correct += correct
        all_total += len(dataset)
    
    overall_acc = all_correct / all_total if all_total > 0 else 0
    return overall_acc * 100, subject_results


def evaluate_ceval(model, tokenizer, num_samples=100, subjects=None):
    """
    C-Eval 评测
    中文知识评测基准
    """
    print("\n📊 评测 C-Eval...")
    
    if subjects is None:
        subjects = QUICK_CEVAL_SUBJECTS
    
    all_correct = 0
    all_total = 0
    
    for subject in tqdm(subjects, desc="C-Eval"):
        try:
            dataset = load_dataset("ceval/ceval-exam", subject, split="val", trust_remote_code=True)
            dev_set = load_dataset("ceval/ceval-exam", subject, split="dev", trust_remote_code=True)
        except Exception as e:
            print(f"   跳过 {subject}: {e}")
            continue
        
        # Few-shot 示例
        few_shot = ""
        for i in range(min(5, len(dev_set))):
            item = dev_set[i]
            few_shot += f"问题：{item['question']}\n"
            few_shot += f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n"
            few_shot += f"答案：{item['answer']}\n\n"
        
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        correct = 0
        for item in dataset:
            prompt = few_shot + f"问题：{item['question']}\n"
            prompt += f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n"
            prompt += "答案："
            
            scores = get_logits_for_choices(model, tokenizer, prompt, ["A", "B", "C", "D"])
            pred_idx = scores.index(max(scores))
            pred = chr(65 + pred_idx)
            if pred == item['answer']:
                correct += 1
        
        all_correct += correct
        all_total += len(dataset)
    
    overall_acc = all_correct / all_total if all_total > 0 else 0
    return overall_acc * 100


def evaluate_hellaswag(model, tokenizer, num_samples=200):
    """
    HellaSwag 评测
    常识推理：选择最合理的句子续写
    """
    print("\n📊 评测 HellaSwag...")
    
    try:
        dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"   加载失败: {e}")
        return None
    
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    correct = 0
    for item in tqdm(dataset, desc="HellaSwag"):
        ctx = item['ctx']
        endings = item['endings']
        label = int(item['label'])
        
        # 计算每个续写的困惑度
        scores = []
        for ending in endings:
            text = ctx + " " + ending
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                # 负 loss 作为分数（越高越好）
                scores.append(-outputs.loss.item())
        
        pred = scores.index(max(scores))
        if pred == label:
            correct += 1
    
    return correct / len(dataset) * 100


def evaluate_arc(model, tokenizer, num_samples=200, difficulty="easy"):
    """
    ARC 评测 (AI2 Reasoning Challenge)
    科学推理选择题
    """
    print(f"\n📊 评测 ARC-{difficulty}...")
    
    try:
        config = "ARC-Easy" if difficulty == "easy" else "ARC-Challenge"
        dataset = load_dataset("allenai/ai2_arc", config, split="test", trust_remote_code=True)
    except Exception as e:
        print(f"   加载失败: {e}")
        return None
    
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    correct = 0
    for item in tqdm(dataset, desc=f"ARC-{difficulty}"):
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey']
        
        # 构建 prompt
        prompt = f"Question: {question}\n"
        choice_labels = choices['label']
        choice_texts = choices['text']
        for label, text in zip(choice_labels, choice_texts):
            prompt += f"{label}. {text}\n"
        prompt += "Answer:"
        
        # 获取预测
        scores = get_logits_for_choices(model, tokenizer, prompt, choice_labels)
        pred_idx = scores.index(max(scores))
        pred = choice_labels[pred_idx]
        
        if pred == answer_key:
            correct += 1
    
    return correct / len(dataset) * 100


def evaluate_winogrande(model, tokenizer, num_samples=200):
    """
    WinoGrande 评测
    常识推理：代词消歧
    """
    print("\n📊 评测 WinoGrande...")
    
    try:
        dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"   加载失败: {e}")
        return None
    
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    correct = 0
    for item in tqdm(dataset, desc="WinoGrande"):
        sentence = item['sentence']
        option1 = item['option1']
        option2 = item['option2']
        answer = item['answer']  # "1" 或 "2"
        
        # 将 _ 替换为选项，计算困惑度
        scores = []
        for option in [option1, option2]:
            text = sentence.replace("_", option)
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                scores.append(-outputs.loss.item())
        
        pred = "1" if scores[0] > scores[1] else "2"
        if pred == answer:
            correct += 1
    
    return correct / len(dataset) * 100


def evaluate_truthfulqa(model, tokenizer, num_samples=200):
    """
    TruthfulQA 评测
    测试模型生成真实答案的能力（MC1 多选一）
    """
    print("\n📊 评测 TruthfulQA...")
    
    try:
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"   加载失败: {e}")
        return None
    
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    correct = 0
    for item in tqdm(dataset, desc="TruthfulQA"):
        question = item['question']
        mc1_targets = item['mc1_targets']
        choices = mc1_targets['choices']
        labels = mc1_targets['labels']  # 1 表示正确答案
        
        # 找到正确答案的索引
        correct_idx = labels.index(1)
        
        # 计算每个选项的分数
        scores = []
        for choice in choices:
            text = f"Question: {question}\nAnswer: {choice}"
            inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                scores.append(-outputs.loss.item())
        
        pred = scores.index(max(scores))
        if pred == correct_idx:
            correct += 1
    
    return correct / len(dataset) * 100


def run_all_benchmarks(model, tokenizer, quick=False):
    """运行所有基准测试"""
    results = {}
    
    # 设置采样数量
    n_mmlu = 50 if quick else 100
    n_ceval = 50 if quick else 100
    n_other = 100 if quick else 200
    
    # MMLU
    mmlu_acc, _ = evaluate_mmlu(model, tokenizer, num_samples=n_mmlu)
    results['MMLU'] = mmlu_acc
    
    # C-Eval
    ceval_acc = evaluate_ceval(model, tokenizer, num_samples=n_ceval)
    results['C-Eval'] = ceval_acc
    
    # HellaSwag
    hellaswag_acc = evaluate_hellaswag(model, tokenizer, num_samples=n_other)
    if hellaswag_acc is not None:
        results['HellaSwag'] = hellaswag_acc
    
    # ARC-Easy
    arc_easy_acc = evaluate_arc(model, tokenizer, num_samples=n_other, difficulty="easy")
    if arc_easy_acc is not None:
        results['ARC-Easy'] = arc_easy_acc
    
    # ARC-Challenge
    arc_challenge_acc = evaluate_arc(model, tokenizer, num_samples=n_other, difficulty="challenge")
    if arc_challenge_acc is not None:
        results['ARC-Challenge'] = arc_challenge_acc
    
    # WinoGrande
    winogrande_acc = evaluate_winogrande(model, tokenizer, num_samples=n_other)
    if winogrande_acc is not None:
        results['WinoGrande'] = winogrande_acc
    
    # TruthfulQA
    truthfulqa_acc = evaluate_truthfulqa(model, tokenizer, num_samples=n_other)
    if truthfulqa_acc is not None:
        results['TruthfulQA'] = truthfulqa_acc
    
    return results


def print_results_table(all_results):
    """打印对比结果表格"""
    print("\n")
    print("=" * 80)
    print("📊 综合评测结果对比表")
    print("=" * 80)
    
    # 获取所有基准名称
    benchmarks = set()
    for results in all_results.values():
        benchmarks.update(results.keys())
    benchmarks = sorted(benchmarks)
    
    # 打印表头
    models = list(all_results.keys())
    header = f"{'Benchmark':<15}"
    for model in models:
        header += f" | {model:>12}"
    header += " |"
    
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    # 打印每个基准的结果
    for benchmark in benchmarks:
        row = f"{benchmark:<15}"
        for model in models:
            if benchmark in all_results[model]:
                score = all_results[model][benchmark]
                row += f" | {score:>11.1f}%"
            else:
                row += f" | {'N/A':>12}"
        row += " |"
        print(row)
    
    print("-" * len(header))
    
    # 计算平均分
    avg_row = f"{'Average':<15}"
    for model in models:
        scores = [v for v in all_results[model].values() if v is not None]
        avg = sum(scores) / len(scores) if scores else 0
        avg_row += f" | {avg:>11.1f}%"
    avg_row += " |"
    print(avg_row)
    print("=" * len(header))
    
    # 打印 ASCII 柱状图
    print("\n📈 平均分柱状图:")
    print("-" * 50)
    for model in models:
        scores = [v for v in all_results[model].values() if v is not None]
        avg = sum(scores) / len(scores) if scores else 0
        bar_len = int(avg / 2)  # 缩放到 50 字符宽度
        bar = "█" * bar_len
        print(f"{model:<12} |{bar} {avg:.1f}%")
    print("-" * 50)


def compare_models(quick=False):
    """对比所有模型"""
    print("=" * 80)
    print("🔬 大模型综合评测")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    loader = ModelLoader()
    all_results = {}
    
    model_types = []
    
    # 检查哪些模型存在
    if os.path.exists(ORIGINAL_MODEL_PATH):
        model_types.append(("original", "Original"))
    
    if os.path.exists(PRETRAIN_MODEL_PATH):
        model_types.append(("pretrain", "Pretrain"))
    elif os.path.exists(ORIGINAL_MODEL_PATH):
        # 如果没有预训练模型，用原始模型代替
        model_types.append(("pretrain", "Pretrain"))
    
    if os.path.exists(LORA_SFT_PATH):
        model_types.append(("sft", "LoRA-SFT"))
    
    for model_type, model_name in model_types:
        print(f"\n{'='*60}")
        print(f"🔄 评测模型: {model_name}")
        print("=" * 60)
        
        try:
            model, tokenizer = loader.load(model_type)
            results = run_all_benchmarks(model, tokenizer, quick)
            all_results[model_name] = results
        except Exception as e:
            print(f"❌ {model_name} 评测失败: {e}")
            continue
    
    # 打印对比表格
    if all_results:
        print_results_table(all_results)
        
        # 保存结果
        output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存到: {output_file}")
    
    return all_results


def evaluate_single_model(model_type, quick=False):
    """评测单个模型"""
    loader = ModelLoader()
    model, tokenizer = loader.load(model_type)
    results = run_all_benchmarks(model, tokenizer, quick)
    
    # 打印结果
    print("\n" + "=" * 50)
    print(f"📊 {model_type} 模型评测结果")
    print("=" * 50)
    print(f"{'Benchmark':<20} {'Score':>10}")
    print("-" * 32)
    for benchmark, score in results.items():
        if score is not None:
            print(f"{benchmark:<20} {score:>9.1f}%")
    
    scores = [v for v in results.values() if v is not None]
    avg = sum(scores) / len(scores) if scores else 0
    print("-" * 32)
    print(f"{'Average':<20} {avg:>9.1f}%")
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="大模型综合评测")
    parser.add_argument("--model", type=str, default="pretrain",
                        choices=["original", "pretrain", "sft"],
                        help="要评测的模型")
    parser.add_argument("--compare", action="store_true",
                        help="对比所有模型")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（减少采样数量）")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["mmlu", "ceval", "hellaswag", "arc", "winogrande", "truthfulqa"],
                        help="只运行指定的基准测试")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.quick)
    elif args.benchmark:
        # 单独运行某个基准
        loader = ModelLoader()
        model, tokenizer = loader.load(args.model)
        
        if args.benchmark == "mmlu":
            acc, _ = evaluate_mmlu(model, tokenizer)
            print(f"\nMMLU: {acc:.1f}%")
        elif args.benchmark == "ceval":
            acc = evaluate_ceval(model, tokenizer)
            print(f"\nC-Eval: {acc:.1f}%")
        elif args.benchmark == "hellaswag":
            acc = evaluate_hellaswag(model, tokenizer)
            print(f"\nHellaSwag: {acc:.1f}%")
        elif args.benchmark == "arc":
            acc_easy = evaluate_arc(model, tokenizer, difficulty="easy")
            acc_hard = evaluate_arc(model, tokenizer, difficulty="challenge")
            print(f"\nARC-Easy: {acc_easy:.1f}%")
            print(f"ARC-Challenge: {acc_hard:.1f}%")
        elif args.benchmark == "winogrande":
            acc = evaluate_winogrande(model, tokenizer)
            print(f"\nWinoGrande: {acc:.1f}%")
        elif args.benchmark == "truthfulqa":
            acc = evaluate_truthfulqa(model, tokenizer)
            print(f"\nTruthfulQA: {acc:.1f}%")
    else:
        evaluate_single_model(args.model, args.quick)
