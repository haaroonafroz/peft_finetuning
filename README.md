# Medical LLM Fine-Tuning with QLoRA

Fine-tune large language models (Qwen3-8B, Llama-2-7B, etc.) on medical domain datasets using **QLoRA** (Quantized Low-Rank Adaptation) — optimized for **≤16 GB GPU memory**.

## Project Structure

```
peft_finetuning/
├── configs/                # YAML configuration files
│   ├── default.yaml        # Base config (all defaults)
│   ├── qwen3_medqa.yaml    # Qwen3-8B on MedQA
│   └── llama2_pubmedqa.yaml# Llama-2-7B on PubMedQA
├── src/
│   ├── data/               # Dataset loading & preprocessing
│   ├── model/              # Model + QLoRA adapter setup
│   ├── training/           # Trainer configuration
│   ├── evaluation/         # Metrics & evaluation pipeline
│   ├── inference/          # Single-sample & batch inference
│   └── utils/              # Logging, config, HF push helpers
├── scripts/                # CLI entry points
│   ├── train.py            # Launch training
│   ├── evaluate.py         # Run evaluation
│   ├── infer.py            # Run inference
│   └── push_to_hub.py      # Upload model to Hugging Face Hub
├── tests/                  # Unit & integration tests
├── Makefile                # Convenience commands
├── Dockerfile              # Reproducible environment
├── requirements.txt        # Pinned dependencies
├── .env.example            # Environment variable template
└── README.md
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/peft-medical-finetuning.git
cd peft-medical-finetuning
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 2. Configure

Copy the environment template and set your Hugging Face token:

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...
```

### 3. Train

```bash
# Using a preset config
python scripts/train.py --config configs/qwen3_medqa.yaml

# Override specific params
python scripts/train.py --config configs/qwen3_medqa.yaml \
    --override training.num_train_epochs=5 \
    --override training.per_device_train_batch_size=2
```

### 4. Evaluate

```bash
python scripts/evaluate.py --config configs/qwen3_medqa.yaml \
    --adapter-path outputs/qwen3-medqa/final_adapter
```

### 5. Inference

```bash
# Interactive
python scripts/infer.py --config configs/qwen3_medqa.yaml \
    --adapter-path outputs/qwen3-medqa/final_adapter \
    --prompt "What are the common side effects of metformin?"

# Batch
python scripts/infer.py --config configs/qwen3_medqa.yaml \
    --adapter-path outputs/qwen3-medqa/final_adapter \
    --input-file data/test_prompts.jsonl \
    --output-file data/predictions.jsonl
```

### 6. Push to Hugging Face Hub

```bash
python scripts/push_to_hub.py \
    --adapter-path outputs/qwen3-medqa/final_adapter \
    --repo-id your-username/qwen3-8b-medqa-qlora \
    --private false
```

## Supported Models

| Model | HF Hub ID | Notes |
|-------|-----------|-------|
| Qwen3-8B | `Qwen/Qwen3-8B` | Primary target, fits 16GB with QLoRA |
| Llama-2-7B | `meta-llama/Llama-2-7b-hf` | Gated, requires HF token + approval |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | Also fits 16GB |

## Supported Datasets

| Dataset | HF Hub ID | Task |
|---------|-----------|------|
| MedQA (USMLE) | `bigbio/med_qa` | Medical multiple-choice QA |
| PubMedQA | `qiaojin/PubMedQA` | Biomedical yes/no/maybe QA |

## QLoRA Configuration

Default adapter settings (override via YAML config):

- **Quantization**: 4-bit NF4 with double quantization
- **LoRA rank**: 64
- **LoRA alpha**: 128
- **Target modules**: Auto-detected per architecture (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Dropout**: 0.05

## GPU Memory Budget

With QLoRA on a 16GB GPU (e.g. RTX 4060 Ti, T4, A4000):

| Model | Batch Size | Gradient Accumulation | Peak VRAM |
|-------|-----------|----------------------|-----------|
| 7B-8B | 2 | 8 | ~12-14 GB |
| 7B-8B | 1 | 16 | ~10-12 GB |

## Using the Fine-Tuned Model in Another Project

```python
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained(
    "your-username/qwen3-8b-medqa-qlora",
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained("your-username/qwen3-8b-medqa-qlora")

inputs = tokenizer("What causes hypertension?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Makefile Commands

```bash
make train CONFIG=configs/qwen3_medqa.yaml    # Train
make evaluate CONFIG=configs/qwen3_medqa.yaml # Evaluate
make infer PROMPT="What is diabetes?"          # Quick inference
make push REPO=your-user/model-name            # Push to Hub
make test                                       # Run tests
make lint                                       # Lint code
make docker-build                               # Build container
make docker-train                               # Train inside container
```

## License

Apache-2.0
