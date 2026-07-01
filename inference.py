"""
inference.py

Runs inference on the fine-tuned Meta Llama 3.2-3B-Instruct model
with a lightweight MCP-style retrieval layer.

Author: Muhammad Ahmed
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/llama32_lora_merged_exact"
DATA_PATH = "data/sample_dataset.jsonl"

# ============================================================
# Load Tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True
)

tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# Load Model
# ============================================================

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

model.eval()

# ============================================================
# Load Knowledge Base (MCP Layer)
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

knowledge_base = {
    item["instruction"].strip().lower(): item["output"].strip()
    for item in dataset
}

print(f"Knowledge Base Loaded: {len(knowledge_base)} entries")

# ============================================================
# MCP Retrieval + Generation Class
# ============================================================

class MCPServer:

    def __init__(self, model, tokenizer, kb):
        self.model = model
        self.tokenizer = tokenizer
        self.kb = kb

    def retrieve(self, query: str):
        return self.kb.get(query.strip().lower())

    def generate(self, query: str, max_new_tokens=200):

        prompt = f"### Instruction:\n{query}\n\n### Response:\n"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def process(self, query: str):

        kb_result = self.retrieve(query)

        if kb_result:
            return " Knowledge Base (Exact Match)", kb_result

        return " LLM Generation", self.generate(query)

# ============================================================
# Initialize MCP
# ============================================================

mcp = MCPServer(model, tokenizer, knowledge_base)

print("\n Inference Ready")
print("Type 'exit' to quit")
print("-" * 50)

# ============================================================
# Interactive Loop
# ============================================================

while True:

    user_input = input("\nEnter instruction: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    source, response = mcp.process(user_input)

    print(f"\n[{source}]")
    print(response)
