# Training Results

## Model
Meta Llama 3.2-3B-Instruct (LoRA Fine-Tuned)

## Dataset
- 430 instruction-response pairs

## Training Configuration
- LoRA Rank: 16
- Alpha: 32
- Dropout: 0.05
- Learning Rate: 1e-4
- Batch Size: 1
- Gradient Accumulation: 4
- Training Steps: 800
- Epochs: ~8

## Training Loss

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
| 800 | 0.5053 |

## Observation
- Stable convergence observed
- Efficient adaptation using LoRA
- No full fine-tuning required
