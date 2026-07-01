# Fine-Tuning Meta Llama 3.2-3B-Instruct using LoRA with a Lightweight Retrieval Layer

## Overview

This project demonstrates parameter-efficient fine-tuning of **Meta Llama 3.2-3B-Instruct** using **LoRA (Low-Rank Adaptation)** on a custom instruction-response dataset.

The project also implements a lightweight retrieval layer that first searches a structured knowledge base for exact matches before falling back to the fine-tuned LLM for response generation. This hybrid approach improves factual retrieval while preserving the model's generative capabilities.

This project was developed during my **AI Internship at IonIdea**, where I gained hands-on experience with Large Language Models (LLMs), prompt engineering, supervised fine-tuning, and LLM inference pipelines.

---

# Features

- Fine-tuning Meta Llama 3.2-3B-Instruct using LoRA
- Supervised Fine-Tuning (SFT) with Hugging Face TRL
- Lightweight retrieval layer for knowledge-base lookup
- Deterministic LLM inference for consistent responses
- Parameter-efficient training with LoRA adapters
- End-to-end training and inference pipeline
- GPU-accelerated training using PyTorch

---

# Architecture

## Training Pipeline

```
Instruction Dataset (JSONL)
            │
            ▼
     Formatting Function
            │
            ▼
     LoRA Fine-Tuning
            │
            ▼
Meta Llama 3.2-3B-Instruct
            │
            ▼
     Fine-Tuned Model
```

---

## Inference Pipeline

```
             User Query
                  │
                  ▼
      Retrieval / Routing Layer
          │                │
          ▼                ▼
Knowledge Base      Fine-Tuned LLM
          │                │
          └───────┬────────┘
                  ▼
           Final Response
```

---

# Training Configuration

| Parameter | Value |
|-----------|--------|
| Base Model | Meta Llama 3.2-3B-Instruct |
| Fine-Tuning Method | LoRA |
| Dataset Size | 430 instruction-response pairs |
| Training Steps | 800 |
| Epochs | ~8 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Learning Rate | 1e-4 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |

---

# Training Results

### Training Loss

| Step | Loss |
|------|------|
| 80 | 1.3466 |
| 160 | 0.7690 |
| 240 | 0.6978 |
| 320 | 0.6489 |
| 400 | 0.5539 |
| 480 | 0.5280 |
| 560 | 0.5192 |
| 640 | 0.5221 |
| 720 | 0.5111 |
| 800 | **0.5053** |

The model demonstrated stable convergence throughout training, with the loss decreasing from **1.35** to **0.51**.

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- TRL
- Datasets
- Accelerate
- Google Colab
- Meta Llama 3.2-3B-Instruct

---

# Installation

Clone the repository

```bash
git clone https://github.com/ahmed7077/efficient-llm-adaptation.git

cd efficient-llm-adaptation
```

Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually

```bash
pip install transformers datasets peft trl accelerate safetensors huggingface_hub
```

---

# Dataset Format

The training dataset should be provided in JSONL format.

Example:

```json
{"instruction":"Who is the CEO of OpenAI?","output":"Sam Altman"}
{"instruction":"What is quantum computing?","output":"Quantum computing uses qubits for computation."}
```

Each record contains:

- instruction
- output

---

# Training

Run

```bash
python train.py
```

The fine-tuned model will be saved as

```
llama32_lora_merged_exact
```

---

# Inference

Run

```bash
python inference.py
```

Example

```
Enter instruction:

Give details about Muhammad Ahmed
```

Output

```
Full Name: Muhammad Ahmed
Age: 20
DOB: 2005-05-23
Nationality: Indian
Degree: B.Tech
University: Presidency University
```

---

# Prompt Engineering

During this project, different prompting strategies were explored, including:

- Zero-shot prompting
- One-shot prompting
- Few-shot prompting

These techniques were evaluated to better understand their impact on LLM response quality and behavior.

---

# Future Improvements

- Implement semantic retrieval using vector databases (FAISS/ChromaDB)
- Support Retrieval-Augmented Generation (RAG)
- Add evaluation metrics for model performance
- Deploy the model using FastAPI or Streamlit
- Extend the retrieval layer with fuzzy matching and embeddings

---

# Acknowledgments

- Meta AI for the Llama models
- Hugging Face for the Transformers, PEFT, and TRL ecosystem
- IonIdea Technologies for providing the internship opportunity and mentorship

---

# License

This project is intended for educational and research purposes.
