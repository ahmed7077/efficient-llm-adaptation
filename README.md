Efficient Fine-Tuning of LLaMA 3.2–3B using LoRA, QLoRA, and MCP

This project focuses on parameter-efficient fine-tuning of the LLaMA 3.2–3B–Instruct model using LoRA (Low-Rank Adaptation) and QLoRA for domain-specific adaptation.
It integrates a Model Control Pipeline (MCP) to intelligently route between database retrieval and generative model inference, enabling efficient training, evaluation, and controlled response generation.

Table of Contents

Overview

Features

Architecture

Installation

Usage

Dataset Format

Model Control Pipeline (MCP)

Results

Tech Stack

Acknowledgments

License

Overview

Large Language Models (LLMs) are powerful but expensive to fine-tune due to their size.
This project demonstrates an efficient, scalable approach to fine-tuning the LLaMA 3.2–3B model using LoRA and QLoRA, reducing GPU memory requirements while preserving accuracy.

The integrated MCP (Model Control Pipeline) provides a hybrid system that:

Uses a knowledge base for exact recall of known data.

Falls back to model inference for unseen or generative queries.

Applies lightweight guardrails for stable responses.

Features

Fine-tuning of LLaMA 3.2–3B–Instruct using LoRA/QLoRA.

Integrated Model Control Pipeline (MCP) for response routing.

Parameter-efficient fine-tuning using PEFT and TRL.

Supports both training and inference modes.

Compatible with GPU and CPU environments.

Modular, extensible Python implementation for future integration (e.g., Flask/Gradio UI).

Architecture

Training Phase

Loads domain dataset in JSONL format.

Applies LoRA or QLoRA adapters to the base LLaMA model.

Fine-tunes the model using Supervised Fine-Tuning (SFT) via SFTTrainer.

MCP Inference Phase

Loads fine-tuned weights and tokenizer.

Creates a knowledge base (dictionary) from training data.

Routes each user query to either:

Knowledge base lookup (for exact match).

Model generation (for new or unseen inputs).

Installation
Prerequisites

Python 3.10+

CUDA-compatible GPU (recommended for training)

Hugging Face account and access to LLaMA 3.2–3B

Setup
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
pip install -r requirements.txt


Or manually install:

pip install transformers datasets peft trl accelerate safetensors huggingface_hub

Usage
1. Training

Edit the dataset path in the script:

dataset_path = "people_osl.jsonl"


Then run:

python fine_tuning_llm.py


The fine-tuned model will be saved in:

./llama32_lora_merged_exact

2. Inference

Once training is complete, the same script will load the model and start the interactive MCP loop:

Enter instruction (or 'exit' to quit): Who is the CEO of OpenAI?
[✅ MCP Database]
Sam Altman

Dataset Format

The dataset should be in JSON Lines (JSONL) format:

{"instruction": "Who is the CEO of OpenAI?", "output": "Sam Altman"}
{"instruction": "What is quantum computing?", "output": "Quantum computing uses qubits to perform parallel computations."}


Ensure keys are exactly "instruction" and "output".

Model Control Pipeline (MCP)

The MCPServer class provides:

Database retrieval for exact matches.

LLM inference for generative answers.

Guardrail mechanism to ensure valid outputs.

This hybrid approach combines factual consistency with generative flexibility.

Results

Model fine-tuned successfully using LoRA (r=16, α=32, dropout=0.05).

Achieved stable convergence with minimal GPU memory usage.

Inference latency reduced via lightweight MCP routing.

Tech Stack

Language: Python 3.10

Libraries: Transformers, PEFT, TRL, Accelerate, Datasets

Model: LLaMA 3.2–3B–Instruct

Framework: Hugging Face Transformers / PEFT

Environment: Google Colab (training), VS Code (local execution)

Acknowledgments

Meta AI for LLaMA models

Hugging Face for Transformers, PEFT, and TRL libraries

IonIdea internship project guidance

License

This project is licensed under the MIT License.
See the LICENSE
 file for more details.
