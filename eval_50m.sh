#!/usr/bin/env bash
# Manual evaluation of the 50M Python-Llama model.
# Run from the SLM project root: bash eval_50m.sh
# Optional: pass a custom prompt as argument to skip the preset prompts.
#   bash eval_50m.sh "def fibonacci"

set -e
cd "$(dirname "$0")"
source .venv/bin/activate

RUN="Python-Llama-50M"
MAX_TOKENS=200
TEMP=0.7
MIN_P=0.05
REP_PEN=1.1

GEN="python mlx-pretrain/generate.py --run $RUN --max-tokens $MAX_TOKENS --temperature $TEMP --min-p $MIN_P --repetition-penalty $REP_PEN"

divider() { printf '\n%s\n' "────────────────────────────────────────────────────────"; }

run_prompt() {
    local label="$1"
    local prompt="$2"
    divider
    printf '▶  %s\n' "$label"
    printf 'PROMPT: %s\n\n' "$prompt"
    $GEN --prompt "$prompt"
}

# ── If user passes a custom prompt, just run that and exit ─────────────────
if [[ -n "$1" ]]; then
    divider
    printf '▶  Custom prompt\n'
    printf 'PROMPT: %s\n\n' "$1"
    $GEN --prompt "$1"
    divider
    exit 0
fi

echo "════════════════════════════════════════════════════════"
echo "  50M Python-Llama  --  Manual Evaluation"
echo "════════════════════════════════════════════════════════"

# 1. Basic function completion
run_prompt "Function completion" \
    "def bubble_sort(arr):"

# 2. Class definition
run_prompt "Class definition" \
    "class Stack:"

# 3. Decorator usage
run_prompt "Decorator" \
    "import functools

def memoize(func):"

# 4. Error handling
run_prompt "Exception handling" \
    "def read_json_file(path):
    try:"

# 5. Generator
run_prompt "Generator function" \
    "def fibonacci():"

# 6. List comprehension / functional style
run_prompt "List comprehension" \
    "def filter_evens(numbers):"

# 7. Docstring + type hints
run_prompt "Type hints + docstring" \
    "def merge_dicts(a: dict, b: dict) -> dict:
    \"\"\"Merge two dicts, b overrides a.\"\"\""

divider
echo ""
echo "════════════════════════════════════════════════════════"
echo "  Interactive mode  (Ctrl-C to quit)"
echo "════════════════════════════════════════════════════════"

while true; do
    echo ""
    printf "Enter a Python prompt (or 'q' to quit): "
    read -r user_prompt
    [[ "$user_prompt" == "q" ]] && break
    [[ -z "$user_prompt" ]] && continue
    divider
    printf 'PROMPT: %s\n\n' "$user_prompt"
    $GEN --prompt "$user_prompt"
done

divider
echo "Eval done."
