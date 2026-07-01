"""
train.py

Fine-tunes Meta Llama 3.2-3B-Instruct using LoRA (Low-Rank Adaptation)
with Supervised Fine-Tuning (SFT).

Author: Muhammad Ahmed
"""

import os
import warnings
import torch

from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)

from peft import (
    LoraConfig,
    TaskType
)

from trl import SFTTrainer

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

DATASET_PATH = "data/sample_dataset.jsonl"

OUTPUT_DIR = "models/llama32_lora_output"

FINAL_MODEL_PATH = "models/llama32_lora_merged_exact"

MAX_STEPS = 800

LEARNING_RATE = 1e-4

# ============================================================
# Hugging Face Login
# ============================================================

print("=" * 60)
print("Hugging Face Login")
print("=" * 60)

login()

# ============================================================
# Load Dataset
# ============================================================

print("\nLoading Dataset...")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

print(f"Dataset Size : {len(dataset)} samples")
print("Sample Record:\n")
print(dataset[0])

# ============================================================
# Load Tokenizer
# ============================================================

print("\nLoading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True
)

tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# Load Base Model
# ============================================================

print("\nLoading Base Model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
)

# ============================================================
# LoRA Configuration
# ============================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# ============================================================
# Prompt Formatting
# ============================================================

def formatting_func(example):

    return (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['output']}"
    )

# ============================================================
# Training Arguments
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=LEARNING_RATE,

    max_steps=MAX_STEPS,

    fp16=torch.cuda.is_available(),

    logging_steps=80,

    save_strategy="steps",

    save_steps=200,

    save_total_limit=1,

    lr_scheduler_type="constant",

    optim="adamw_torch",

    report_to=[]

)

# ============================================================
# Trainer
# ============================================================

trainer = SFTTrainer(

    model=model,

    train_dataset=dataset,

    peft_config=lora_config,

    formatting_func=formatting_func,

    args=training_args

)

# ============================================================
# Training
# ============================================================

print("\n" + "=" * 60)
print("Starting Training")
print("=" * 60)

trainer.train()

# ============================================================
# Save Model
# ============================================================

print("\nSaving Fine-Tuned Model...")

trainer.save_model(FINAL_MODEL_PATH)

tokenizer.save_pretrained(FINAL_MODEL_PATH)

print("\nTraining Complete!")

print(f"\nModel saved to:\n{FINAL_MODEL_PATH}")

print("\nDone.")
