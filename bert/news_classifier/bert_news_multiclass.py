from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
import os

MODEL_NAME = "bert-base_chinese"
NUM_LABELS = 4
SELECTED_CATEGORIES = ["体育", "财经", "科技", "娱乐"]
MODEL_SAVE_PATH = "./thucnews_bert_model"

dataset = load_dataset("seamew/THUCNEWS")

