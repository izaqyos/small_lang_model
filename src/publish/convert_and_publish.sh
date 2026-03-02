#!/bin/bash
set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model-name> [ollama-username]}"
OLLAMA_USER="${2:-}"
FUSED_DIR="fused_model"
GGUF_DIR="models/gguf"
LLAMA_CPP_DIR="llama.cpp"

echo "=== Python Expert SLM: Convert & Publish Pipeline ==="
echo ""

# Step 1: Merge LoRA adapters
if [ ! -d "$FUSED_DIR" ]; then
    echo "[1/5] Merging LoRA adapters..."
    python -m mlx_lm.fuse \
        --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
        --adapter-path adapters \
        --de-quantize
    echo "  Merged model saved to $FUSED_DIR/"
else
    echo "[1/5] Fused model already exists, skipping merge."
fi

# Step 2: Clone llama.cpp if needed
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "[2/5] Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
    pip install -r llama.cpp/requirements.txt
else
    echo "[2/5] llama.cpp already exists."
fi

# Step 3: Convert to GGUF
mkdir -p "$GGUF_DIR"
echo "[3/5] Converting to GGUF format..."
python llama.cpp/convert_hf_to_gguf.py "$FUSED_DIR" --outfile "$GGUF_DIR/${MODEL_NAME}-f16.gguf"

# Step 4: Quantize
echo "[4/5] Quantizing to Q4_K_M..."
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "  Building llama.cpp quantizer..."
    cd llama.cpp && cmake -B build && cmake --build build --target llama-quantize -j && cd ..
fi
./llama.cpp/build/bin/llama-quantize \
    "$GGUF_DIR/${MODEL_NAME}-f16.gguf" \
    "$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf" \
    Q4_K_M

echo "  Quantized model: $GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf"
ls -lh "$GGUF_DIR/${MODEL_NAME}-Q4_K_M.gguf"

# Step 5: Create Ollama model
echo "[5/5] Creating Ollama model..."
cat > "$GGUF_DIR/Modelfile" << 'MODELFILE'
FROM ./python-expert-Q4_K_M.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""

SYSTEM """You are a Python expert assistant. You write clean, idiomatic, well-tested Python code. You explain your reasoning clearly and follow PEP 8 conventions."""

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 4096
MODELFILE

if [ -n "$OLLAMA_USER" ]; then
    FULL_NAME="${OLLAMA_USER}/${MODEL_NAME}"
else
    FULL_NAME="${MODEL_NAME}"
fi

cd "$GGUF_DIR"
ollama create "$FULL_NAME" -f Modelfile
cd ../..

echo ""
echo "=== Done! ==="
echo "Test locally:  ollama run $FULL_NAME"
if [ -n "$OLLAMA_USER" ]; then
    echo "Publish:       ollama push $FULL_NAME"
fi
