# Efficient Fine-Tuning of LLaMA 3.2–3B using LoRA, QLoRA, and MCP

## Overview

This project demonstrates parameter-efficient fine-tuning of the **LLaMA 3.2–3B–Instruct** model using **LoRA (Low-Rank Adaptation)** and **QLoRA** for domain-specific adaptation.

It introduces a **Model Control Pipeline (MCP)** that combines knowledge-base retrieval with generative inference, enabling controlled, efficient, and accurate response generation.

The system is designed for memory-efficient training, scalable inference, and hybrid reasoning.

---

## Features

* Parameter-efficient fine-tuning using LoRA and QLoRA
* Integration of Model Control Pipeline (MCP) for intelligent routing
* Lightweight knowledge-base retrieval for factual queries
* LLM-based inference for unseen or generative queries
* Compatible with GPU and CPU environments
* Built using Hugging Face PEFT, TRL, and Transformers
* Modular and extensible Python implementation

---

## Architecture

### Training Phase

* Loads dataset in JSONL format
* Applies LoRA/QLoRA adapters on LLaMA 3.2–3B
* Fine-tunes using Supervised Fine-Tuning (SFTTrainer)

### MCP Inference Phase

* Loads fine-tuned model and tokenizer
* Builds a knowledge base from dataset
* Routes queries to:

  * Exact match retrieval (database)
  * Generative LLM inference

---

## Installation

### Prerequisites

* Python 3.10+
* CUDA-compatible GPU (recommended)
* Hugging Face account with access to LLaMA 3.2–3B

---

### Setup

```bash id="lq8p3a"
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
pip install -r requirements.txt
```

---

### Manual Dependencies

```bash id="k2m9qp"
pip install transformers datasets peft trl accelerate safetensors huggingface_hub
```

---

## Usage

### 1. Training

Edit dataset path:

```python id="v8n2sa"
dataset_path = "people_osl.jsonl"
```

Run training:

```bash id="x9k2dp"
python fine_tuning_llm.py
```

Model output will be saved at:

```text id="q1m8pl"
./llama32_lora_merged_exact
```

---

### 2. Inference

After training, run interactive MCP mode:

```text id="z7n3kd"
Enter instruction (or 'exit' to quit):
Who is the CEO of OpenAI?
```

Output:

```text
[MC P Database]
Sam Altman
```

---

## Dataset Format

The dataset must be in JSONL format:

```json id="d8m2qp"
{"instruction": "Who is the CEO of OpenAI?", "output": "Sam Altman"}
{"instruction": "What is quantum computing?", "output": "Quantum computing uses qubits for parallel computation."}
```

Each record must contain:

* `instruction`
* `output`

---

## Model Control Pipeline (MCP)

The MCP system includes:

* Exact match retrieval from a structured knowledge base
* LLM-based generation for unseen queries
* Guardrail logic for stable and controlled responses

This hybrid design ensures both:

* Factual accuracy
* Generative flexibility

---

## Results

* Efficient fine-tuning achieved using LoRA (r=16, α=32, dropout=0.05)
* Reduced GPU memory consumption compared to full fine-tuning
* Stable convergence during training
* Faster inference through MCP routing mechanism

---

## Tech Stack

* Python 3.10
* Hugging Face Transformers
* PEFT (Parameter Efficient Fine-Tuning)
* TRL (Transformer Reinforcement Learning)
* Accelerate
* Datasets
* LLaMA 3.2–3B–Instruct

---

## Acknowledgments

* Meta AI for LLaMA models
* Hugging Face for Transformers and PEFT ecosystem
* IonIdea internship guidance and support

---

## License

This project is intended for educational and research purposes only.

---

