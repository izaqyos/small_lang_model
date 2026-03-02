# Python Expert SLM

A small language model specialized in Python code, built in two stages:

1. **Stage 1**: Train a 50M parameter Llama model from scratch on Python code using MLX
2. **Stage 2**: Distill a 500M parameter model via synthetic data + LoRA fine-tuning of Qwen2.5-Coder-0.5B

Trained on Apple Silicon (M4 Pro 48GB) using [MLX](https://github.com/ml-explore/mlx).

## Setup

```bash
# Requires Python 3.10 or 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
src/
  data/              # Dataset preparation and synthetic data generation
  model/             # Model architecture (Stage 1)
  finetune/          # LoRA fine-tuning (Stage 2)
  eval/              # Evaluation and benchmarking
  publish/           # GGUF conversion and Ollama publishing
configs/             # Training configurations
mlx-pretrain/        # Pre-training framework (git submodule)
```

## Stage 1: Train 50M Model from Scratch

```bash
# Prepare Python code dataset
python src/data/prepare_python_corpus.py

# Train tokenizer
cd mlx-pretrain && python train-tokenizer.py --config ../configs/tokenizer-config.yaml

# Train model
python train.py --config ../configs/model-config-python-50m.yaml
```

## Stage 2: Distill 500M Model

```bash
# Generate synthetic training data
python src/data/generate_synthetic.py

# LoRA fine-tune
python -m mlx_lm.lora --model Qwen/Qwen2.5-Coder-0.5B-Instruct --data ./data/synthetic/ --train

# Merge and evaluate
python -m mlx_lm.fuse --model Qwen/Qwen2.5-Coder-0.5B-Instruct --adapter-path ./adapters
```
