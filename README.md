# Python Expert SLM

How far can you get training a Python code model on one laptop? This project walks through the full lifecycle — pre-training a 42M-parameter Llama on 457K Python functions, fine-tuning Qwen2.5-Coder (0.5B & 3B) with LoRA on 5K synthetic examples, and shipping quantized GGUF models to Ollama. Everything runs on Apple Silicon (M4 Pro, 48 GB) using MLX. The repo also includes interactive learning tools: a React slide deck and an animated forward-pass simulator that traces "def fib" through every transformer layer.

Built in three stages:

1. **Stage 1 — Learn**: Train a 50M parameter Llama model from scratch on Python code
2. **Stage 2 — Distill**: Fine-tune Qwen2.5-Coder (0.5B/3B) via synthetic data + LoRA
3. **Stage 3 — Ship**: Convert to GGUF, quantize, and publish to Ollama

Trained entirely on Apple Silicon (M4 Pro 48GB) using [MLX](https://github.com/ml-explore/mlx).

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
  data/                # Dataset preparation and synthetic data generation
  model/               # Model architecture (Stage 1)
  finetune/            # LoRA fine-tuning (Stage 2)
  eval/                # Evaluation and benchmarking
  publish/             # GGUF conversion and Ollama publishing
configs/               # Training configurations (50M, 0.5B, 3B)
mlx-pretrain/          # Pre-training framework (git submodule)
presentation/          # React slide deck (Vite + Spectacle + Framer Motion)
learning/              # Interactive learning materials
  forward-pass-sim/    # Animated forward pass simulator (React)
  FowrwardPassSim.*    # Static forward pass reference (TSX + HTML)
  LEARNING_GUIDE.md    # Comprehensive written guide
```

## Stage 1: Train 50M Model from Scratch

```bash
# Prepare Python code dataset (457K functions from code_search_net)
python src/data/prepare_python_corpus.py

# Train BPE tokenizer (8K vocab)
cd mlx-pretrain && python train-tokenizer.py --config ../configs/tokenizer-config.yaml

# Train model (~4K steps, cosine LR schedule)
python train.py --config ../configs/model-config-python-50m.yaml
```

## Stage 2: Distill via LoRA Fine-tuning

```bash
# Generate synthetic training data (5K+ instruction/response pairs)
python src/data/generate_synthetic.py
python src/data/generate_from_corpus.py

# LoRA fine-tune Qwen2.5-Coder-0.5B
python -m mlx_lm.lora --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --data ./data/synthetic/ --train --config configs/lora-config.yaml

# LoRA fine-tune Qwen2.5-Coder-3B (optional, larger model)
python -m mlx_lm.lora --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --data ./data/synthetic/ --train --config configs/lora-config-3b.yaml

# Merge adapters into base model
python -m mlx_lm.fuse --model Qwen/Qwen2.5-Coder-0.5B-Instruct --adapter-path ./adapters
```

## Stage 3: Ship to Ollama

```bash
# Convert to GGUF, quantize, and publish
bash src/publish/convert_and_publish.sh
```

Pipeline: MLX model → HuggingFace format → GGUF f16 → Q4_K_M quantization → Ollama registry.

## Interactive Tools

### Presentation Deck

Slide deck covering transformer building blocks, our training pipeline, live demos, and a deep dive into architecture details.

```bash
cd presentation && npm install && npm run dev
```

### Forward Pass Simulator

Animated visualization tracing `"def fib"` through every layer of the 50M transformer — tokenization, embedding, multi-head attention, SwiGLU FFN, and softmax output prediction.

```bash
cd learning/forward-pass-sim && npm install && npm run dev
```

## Evaluation Notes

### Fine-tuned Qwen 0.5B

- **Code generation**: Produces correct, well-documented Python (fibonacci, sorting, etc.)
- **Debugging**: Struggles with multi-step code reasoning — hallucinates issues instead of tracing logic
- **Root cause**: ~5K training examples is small; debugging data wasn't complex enough

### 50M Python-Llama

- Learns Python syntax and structure (correct indentation, valid constructs)
- Cannot produce semantically correct code — expected for 42M params on 160M tokens
- Serves as a teaching artifact for understanding the full training pipeline

## Key Numbers

| Metric        | 50M Model      | 0.5B LoRA          | 3B LoRA            |
| ------------- | -------------- | ------------------ | ------------------ |
| Total params  | 42M            | 494M               | 3.1B               |
| Trainable     | 42M (100%)     | 1.4M (0.29%)       | ~5M (~0.16%)       |
| Training data | 457K functions | 5K synthetic pairs | 5K synthetic pairs |
| Training time | ~8 hours       | ~3.9 hours         | ~6 hours           |
| Peak memory   | ~2 GB          | 10.6 GB            | ~12 GB             |
| Hardware      | M4 Pro 48GB    | M4 Pro 48GB        | M4 Pro 48GB        |
