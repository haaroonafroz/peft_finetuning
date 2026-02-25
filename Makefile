.PHONY: train evaluate infer push test lint format docker-build docker-train clean help

CONFIG ?= configs/qwen3_medqa.yaml
ADAPTER_PATH ?= outputs/qwen3-medqa/final_adapter
PROMPT ?= "What are the symptoms of diabetes?"
REPO ?= your-username/model-name

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

train: ## Train with QLoRA (CONFIG=path/to/config.yaml)
	python scripts/train.py --config $(CONFIG)

evaluate: ## Evaluate a trained adapter (ADAPTER_PATH=...)
	python scripts/evaluate.py --config $(CONFIG) --adapter-path $(ADAPTER_PATH)

infer: ## Run single-prompt inference (PROMPT="...")
	python scripts/infer.py --config $(CONFIG) --adapter-path $(ADAPTER_PATH) --prompt $(PROMPT)

push: ## Push adapter to HF Hub (REPO=user/model)
	python scripts/push_to_hub.py --adapter-path $(ADAPTER_PATH) --repo-id $(REPO)

test: ## Run all tests
	python -m pytest tests/ -v

lint: ## Lint code with ruff
	ruff check src/ scripts/ tests/

format: ## Auto-format code with ruff
	ruff format src/ scripts/ tests/

docker-build: ## Build Docker image
	docker build -t peft-medical-finetuning .

docker-train: ## Train inside Docker container
	docker run --gpus all --rm \
		-v $(PWD)/outputs:/app/outputs \
		-v $(PWD)/.env:/app/.env \
		--env-file .env \
		peft-medical-finetuning \
		scripts/train.py --config $(CONFIG)

clean: ## Remove outputs and caches
	rm -rf outputs/ wandb/ runs/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
