# QLoRA Fine-Tuning on Medical QA: An Ablation Study

Fine-tuning large language models for domain-specific tasks typically demands multiple high-end GPUs and hundreds of gigabytes of memory. This project demonstrates that **parameter-efficient fine-tuning (PEFT)** methods — specifically **QLoRA** (Quantized Low-Rank Adaptation) — can adapt 8B-parameter models to medical question answering using as little as **two 16 GB GPUs**, training fewer than **0.2% of total parameters**.

We frame this as an **ablation study** over QLoRA hyperparameters (rank, alpha, target modules) across multiple base models and evaluate on the MedQA (USMLE) benchmark to understand how adapter capacity affects downstream medical reasoning.

## Background

### Why PEFT?

Full fine-tuning of a 7–8B model requires storing the full model weights, gradients, and optimizer states in memory — roughly 60–120 GB of VRAM depending on precision. PEFT methods avoid this by **freezing the base model** and injecting small trainable adapters into specific layers. Only these adapter parameters are updated during training.

### QLoRA in Brief

QLoRA combines two key techniques:

1. **4-bit NF4 Quantization** — The frozen base model is loaded in 4-bit precision using the NormalFloat4 data type, reducing a 16 GB fp16 model to ~5 GB in VRAM.
2. **Low-Rank Adaptation (LoRA)** — Small rank-decomposition matrices are injected into attention layers. A weight update ΔW is decomposed as ΔW = BA where B ∈ R^(d×r) and A ∈ R^(r×d), with rank r << d. The scaling factor α/r controls the adapter's influence on the frozen weights.

The result: a quantized base model that fits on consumer hardware, with a lightweight adapter (~30–60 MB) that captures domain-specific knowledge.

## Project Structure

```
peft_finetuning/
├── configs/                  # YAML configuration files
│   ├── default.yaml          # Base config with all defaults
│   ├── qwen3_medqa.yaml      # Qwen3-8B on MedQA
│   └── llama2_pubmedqa.yaml  # Llama-2-7B on PubMedQA
├── src/
│   ├── data/                 # Dataset loading, formatting, tokenization
│   │   ├── loader.py         # HF dataset loading + train/val splitting
│   │   └── formatting.py     # Prompt template formatting per dataset
│   ├── model/                # Model + QLoRA adapter setup
│   │   └── loader.py         # Quantized model loading + LoRA injection
│   ├── training/             # Trainer configuration
│   │   └── trainer.py        # SFTTrainer with completion-only loss
│   ├── evaluation/           # Metrics and evaluation pipeline
│   │   ├── evaluator.py      # Perplexity, generation, metric orchestration
│   │   └── metrics.py        # Accuracy, ROUGE, BLEU, semantic similarity
│   ├── inference/            # Single-sample and batch inference
│   │   └── generate.py       # Adapter loading, generation, stop handling
│   └── utils/                # Logging, config parsing, HF Hub helpers
├── scripts/                  # CLI entry points
│   ├── train.py              # Launch training
│   ├── evaluate.py           # Run evaluation suite
│   ├── infer.py              # Interactive / batch inference
│   └── push_to_hub.py        # Upload adapter to Hugging Face Hub
├── outputs/                  # Training outputs (per-run subdirectories)
├── tests/                    # Unit tests
├── configs/                  # Experiment configs
├── Makefile                  # Convenience commands
├── Dockerfile                # Reproducible environment
├── requirements.txt          # Python dependencies
└── README.md
```

## How Fine-Tuning Works

### 1. Quantized Model Loading

The base model is loaded in 4-bit NF4 precision with double quantization enabled, reducing memory from ~16 GB (fp16) to ~5 GB per GPU:

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
```

### 2. LoRA Adapter Injection

Low-rank adapters are injected into the model's attention projection layers. The rank `r` and scaling factor `lora_alpha` are the primary hyperparameters under study:

```yaml
lora:
  r: 32              # rank of the low-rank matrices
  lora_alpha: 64      # scaling factor (effective scale = alpha / r)
  lora_dropout: 0.05
  target_modules: null # auto-detected (q_proj, v_proj)
```

### 3. Completion-Only Training

We use `SFTTrainer` from TRL with `completion_only_loss=True`. This masks the loss on prompt tokens (instruction, question, options) and only computes cross-entropy loss on the **answer tokens**. This focuses 100% of the training signal on learning correct answers rather than reproducing boilerplate prompt text.

### 4. Multi-GPU Data Parallelism

Training uses HuggingFace Accelerate with DDP (DistributedDataParallel) across 2x Tesla V100-16GB GPUs. Each GPU loads a full copy of the quantized model and processes different batches in parallel, effectively doubling throughput.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch scripts/train.py --config configs/qwen3_medqa.yaml
```

## Experimental Setup

### Dataset

| Dataset | Source | Task | Train | Test |
|---------|--------|------|-------|------|
| MedQA (USMLE) | `GBaker/MedQA-USMLE-4-options` | 4-option medical MCQ | 9,160 | 1,273 |

A 90/10 split of the training set is used for validation during training. The held-out test set (1,273 examples) is used for final evaluation.

### Prompt Format

```
Below is a medical question. Choose the correct answer.

### Question:
{question}

### Options:
  A. {option_a}
  B. {option_b}
  C. {option_c}
  D. {option_d}

### Answer:
{answer}
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Perplexity** | Exponentiated average cross-entropy loss on the test set |
| **Accuracy** | Fuzzy match — correct if the reference answer text appears in the generated output |
| **ROUGE-1/2/L** | N-gram overlap (unigram, bigram, longest common subsequence) between generated and reference answers |
| **Semantic Similarity** | Cosine similarity of PubMedBERT embeddings between generated and reference answers |

Semantic similarity uses `NeuML/pubmedbert-base-embeddings`, a domain-specific biomedical embedding model, providing meaning-aware comparison that handles medical synonyms (e.g., "myocardial infarction" vs. "heart attack").

### Hardware

| Component | Specification |
|-----------|---------------|
| GPU | 2x Tesla V100-PCIE-16GB |
| CUDA | 12.5 |
| Precision | 4-bit NF4 (base model) + bf16 (compute) |

### Shared Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size (per device) | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 32 (2 × 2 GPUs × 8 accum) |
| Learning rate | 1e-4 |
| Scheduler | Cosine |
| Warmup | 6% of steps |
| Optimizer | Paged AdamW 8-bit |
| Max sequence length | 1024 |
| Quantization | NF4 + double quantization |
| Loss | Completion-only (prompt tokens masked) |

## Ablation Results

### Qwen3-8B on MedQA

| Config | Rank (r) | Alpha (α) | α/r | Target Modules | Trainable Params | % of Total | Perplexity ↓ | Accuracy ↑ | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | Semantic Sim ↑ |
|--------|----------|-----------|-----|----------------|-----------------|------------|-------------|-----------|-----------|-----------|-----------|---------------|
| r16-a32 | 16 | 32 | 2.0 | q_proj, v_proj | ~7.7M | 0.09% | 3.277 | 0.620 | 0.657 | 0.493 | 0.657 | 0.809 |
| r32-a64 | 32 | 64 | 2.0 | q_proj, v_proj | ~15.3M | 0.19% | 3.256 | 0.580 | 0.641 | 0.478 | 0.641 | 0.803 |

**Base model**: Qwen/Qwen3-8B (8.22B parameters) — evaluated on 50 test samples with seed 42.

### Observations

- **r=16 slightly outperforms r=32** on accuracy (62% vs 58%) despite having half the trainable parameters. This suggests the model may be mildly overfitting at higher rank with only 3 epochs and 9K training examples, or that the smaller adapter generalizes better on this dataset size.
- **Perplexity is comparable** across both configurations (~3.26–3.28), indicating similar language modeling capability.
- **Semantic similarity is high** for both (>0.80), confirming the model produces answers that are semantically close to the ground truth even when surface-level metrics vary.
- The **α/r ratio is fixed at 2.0** across experiments. Future work could vary this ratio to study its effect independently.

## Planned Experiments

| Model | HF Hub ID | Parameters | Dataset | Status |
|-------|-----------|------------|---------|--------|
| Qwen3-8B | `Qwen/Qwen3-8B` | 8.22B | MedQA | r=16, r=32 complete |
| Qwen3-8B | `Qwen/Qwen3-8B` | 8.22B | MedQA | r=64 planned |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B` | 8.03B | MedQA | Planned |
| Mistral-7B-v0.3 | `mistralai/Mistral-7B-v0.3` | 7.25B | MedQA | Planned |

Upcoming ablations will study:

- **Rank scaling**: r ∈ {8, 16, 32, 64, 128} at fixed α/r = 2
- **Cross-architecture comparison**: Qwen3-8B vs. Llama-3.1-8B vs. Mistral-7B on the same dataset and QLoRA configuration
- **Target module expansion**: Adding gate_proj, up_proj, down_proj to the target modules alongside q_proj and v_proj
- **Epoch scaling**: 3 vs. 5 vs. 7 epochs to measure overfitting thresholds per rank

## Quick Start

### 1. Environment Setup

```bash
conda create --prefix ./env python=3.11 -y
conda activate ./env
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Set HF_TOKEN=hf_... in .env
```

### 3. Train

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch scripts/train.py --config configs/qwen3_medqa.yaml
```

Override hyperparameters for ablation:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  accelerate launch scripts/train.py --config configs/qwen3_medqa.yaml \
    --override lora.r=64 \
    --override lora.lora_alpha=128 \
    --override training.run_name=qwen3-medqa-QLORA-3Epochs-r64-a128
```

### 4. Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --config configs/qwen3_medqa.yaml \
    --adapter-path outputs/qwen3-medqa-QLORA-3Epochs-r32-a64/final_adapter \
    --max-samples 200
```

### 5. Inference

```bash
# Interactive
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
    --config configs/qwen3_medqa.yaml \
    --adapter-path outputs/qwen3-medqa-QLORA-3Epochs-r32-a64/final_adapter

# Single prompt
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
    --config configs/qwen3_medqa.yaml \
    --adapter-path outputs/qwen3-medqa-QLORA-3Epochs-r32-a64/final_adapter \
    --prompt "Below is a medical question. Choose the correct answer.

### Question:
What is the most common cause of acute pancreatitis?

### Options:
  A. Gallstones
  B. Alcohol abuse
  C. Hypertriglyceridemia
  D. Drug-induced

### Answer:
"
```

### 6. Push to Hugging Face Hub

```bash
python scripts/push_to_hub.py \
    --adapter-path outputs/qwen3-medqa-QLORA-3Epochs-r32-a64/final_adapter \
    --repo-id your-username/qwen3-8b-medqa-qlora-r32 \
    --private false
```

## Using a Fine-Tuned Adapter

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoPeftModelForCausalLM.from_pretrained(
    "your-username/qwen3-8b-medqa-qlora-r32",
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "your-username/qwen3-8b-medqa-qlora-r32",
    trust_remote_code=True,
)

prompt = """Below is a medical question. Choose the correct answer.

### Question:
What is the most common cause of acute pancreatitis?

### Options:
  A. Gallstones
  B. Alcohol abuse
  C. Hypertriglyceridemia
  D. Drug-induced

### Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## License

Apache-2.0
