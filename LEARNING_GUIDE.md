# Building a Python Expert Small Language Model: A Learning Guide

A hands-on walkthrough of training a language model from scratch and fine-tuning
a pre-trained model using knowledge distillation. Everything runs on a MacBook Pro
with Apple Silicon using the MLX framework.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Key Concepts](#2-key-concepts)
3. [Stage 1: Training a 50M Model from Scratch](#3-stage-1-training-a-50m-model-from-scratch)
   - [Step 1: Dataset Preparation](#step-1-dataset-preparation)
   - [Step 2: Tokenizer Training](#step-2-tokenizer-training)
   - [Step 3: Model Architecture](#step-3-model-architecture)
   - [Step 4: Training Loop](#step-4-training-loop)
   - [Step 5: Evaluation](#step-5-evaluation)
4. [Stage 2: Distilling a 500M Python Expert](#4-stage-2-distilling-a-500m-python-expert)
   - [Step 6: Synthetic Data Generation](#step-6-synthetic-data-generation)
   - [Step 7: LoRA Fine-tuning](#step-7-lora-fine-tuning)
   - [Step 8: Merging and Evaluation](#step-8-merging-and-evaluation)
5. [Stage 3: Publishing to Ollama](#5-stage-3-publishing-to-ollama)
   - [Step 9: GGUF Conversion and Quantization](#step-9-gguf-conversion-and-quantization)
   - [Step 10: Ollama Packaging](#step-10-ollama-packaging)
6. [Appendix: Key Formulas and Numbers](#6-appendix-key-formulas-and-numbers)

---

## 1. The Big Picture

We're building a Python code expert in two stages:

```
Stage 1: LEARN THE PIPELINE
  Raw Python Code  -->  Tokenizer  -->  50M Param Llama  -->  Generates Python
  (code_search_net)     (BPE 8K)       (trained from 0)      (basic syntax)

Stage 2: BUILD THE REAL MODEL
  Teacher (Claude)  -->  Synthetic Data  -->  LoRA Fine-tune  -->  Python Expert
  (generates data)       (5K+ examples)      Qwen2.5-0.5B         (real quality)

Stage 3: SHIP IT
  Fine-tuned Model  -->  GGUF Conversion  -->  Quantize  -->  Ollama Registry
  (merge LoRA)           (llama.cpp)           (Q4_K_M)       (anyone can use)
```

**Why two stages?** Training a useful 500M model from scratch requires trillions of
tokens and massive compute. Instead, we:
1. Build a tiny model from scratch to **learn how everything works**
2. Take a model someone already spent millions training (Qwen2.5-Coder-0.5B) and
   **specialize it** for our use case with a few thousand high-quality examples

This is called **knowledge distillation** -- using a powerful teacher model (Claude)
to generate training data that transfers its abilities into a small student model.

---

## 2. Key Concepts

### What is a Language Model?

A language model predicts the next token given previous tokens. That's it. Everything
else -- code generation, conversations, reasoning -- emerges from this simple objective.

```
Input:   "def fibonacci(n):\n    if n <= 1:\n        return"
Model:   P(next_token | previous_tokens)
Output:  " n"   (the model predicts the most likely continuation)
```

### Transformer Architecture (Llama variant) -- Deep Dive

Our 50M model uses the Llama architecture -- the same family as Meta's LLaMA, just
much smaller. Here's the full picture, layer by layer.

#### The Full Data Flow

```
Input Token IDs: [142, 3891, 7234, 11, 52, 284]   ("def fibonacci(n):")
                            |
                            v
  ┌─────────────────────────────────────────────┐
  │          EMBEDDING LAYER                     │
  │  Lookup table: 8192 entries x 512 dims       │
  │  Token 142 -> [0.02, -0.15, 0.08, ...]      │
  │  Output shape: [seq_len, 512]                │
  └─────────────────────────────────────────────┘
                            |
              (seq_len x 512 matrix)
                            |
          ┌─────────────────────────────────┐
          │      TRANSFORMER BLOCK 1/12      │
          │                                  │
          │  x_norm = RMSNorm(x)             │
          │  attn_out = SelfAttention(x_norm)│
          │  x = x + attn_out    ← residual  │
          │                                  │
          │  x_norm = RMSNorm(x)             │
          │  ffn_out = SwiGLU_FFN(x_norm)    │
          │  x = x + ffn_out     ← residual  │
          └─────────────────────────────────┘
                            |
                     (repeated 12x)
                            |
          ┌─────────────────────────────────┐
          │          FINAL RMSNorm           │
          └─────────────────────────────────┘
                            |
          ┌─────────────────────────────────┐
          │       OUTPUT PROJECTION          │
          │   512-dim vector -> 8192 logits  │
          │   (tied with embedding weights)  │
          └─────────────────────────────────┘
                            |
                            v
          Logits: [0.1, -2.3, ..., 5.7, ...]  (8192 values)
                            |
                        softmax
                            |
          Probabilities: P("n") = 0.23, P(" ") = 0.15, ...
```

#### 1. Embedding Layer

The embedding layer is just a lookup table. Each of the 8,192 tokens in our
vocabulary maps to a learned 512-dimensional vector.

```
Mathematically:   E(token_id) = W_embed[token_id]

W_embed shape:    [8192, 512]    (8192 tokens, each a 512-dim vector)
Parameters:       8192 * 512 = 4,194,304 (4.2M)
```

These vectors start random and are learned during training. After training,
tokens with similar meanings end up near each other in this 512-dim space.
For example, `for` and `while` will be closer to each other than to `import`.

Think of it as: every token gets a coordinate in 512-dimensional space, and the
model learns to place them so that the geometry captures meaning.

#### 2. RMSNorm (Root Mean Square Normalization)

Before each sub-layer (attention, FFN), we normalize the input. This stabilizes
training by preventing activations from growing too large or too small.

```
                    x_i
RMSNorm(x)_i = ─────────── * gamma_i
                RMS(x)

                  ┌─────────────────────┐
where RMS(x) =   │  1/d * sum(x_i^2)   │
                  └─────────────────────┘

d = 512 (hidden size)
gamma = learned scale parameter (512 values)
eps = 1e-5 (for numerical stability, added inside the sqrt)
```

**Why RMSNorm instead of LayerNorm?** LayerNorm subtracts the mean AND divides by
std. RMSNorm skips the mean subtraction -- it only rescales by the root mean square.
This is ~10-15% faster with virtually identical results. Llama, Qwen, and most
modern LLMs use RMSNorm.

```
LayerNorm:  (x - mean) / std * gamma + beta    (4 operations, 2 learned params)
RMSNorm:    x / RMS(x) * gamma                 (2 operations, 1 learned param)
```

Parameters per RMSNorm: 512 (just gamma). With 2 norms per layer * 12 layers + 1
final = 25 norms * 512 = 12,800 parameters.

#### 3. Self-Attention (Multi-Head, with RoPE)

This is the core mechanism. Self-attention lets each token "look at" all previous
tokens and decide which ones are relevant for predicting the next token.

**The intuition:** When the model sees `return` at position 50, it needs to know
what function it's inside (position 3), what variables are defined (positions 10-40),
and what the return type should be. Attention lets it selectively focus on exactly
those positions.

**Step by step:**

```
Input: x of shape [seq_len, 512]

Step 1: Project into Q (query), K (key), V (value)

    Q = x @ W_Q     # "What am I looking for?"    [seq_len, 512]
    K = x @ W_K     # "What do I contain?"         [seq_len, 512]
    V = x @ W_V     # "What information do I carry?" [seq_len, 512]

    W_Q, W_K, W_V are each [512, 512] = 262,144 params each

Step 2: Split into 8 attention heads (64 dims each)

    Q -> [seq_len, 8, 64]    # 8 heads, each looking at 64-dim subspace
    K -> [seq_len, 8, 64]
    V -> [seq_len, 8, 64]

    Each head can specialize: one head might track indentation level,
    another might track variable names, another might track function scope.

Step 3: Apply RoPE to Q and K (position encoding, explained below)

Step 4: Compute attention scores

    scores = (Q @ K.T) / sqrt(64)     # [seq_len, seq_len] per head
                         ^^^^
                  scaling factor to keep variance stable

    For token i attending to token j:
    score(i,j) = (q_i . k_j) / 8.0

Step 5: Apply causal mask (can only attend to past, not future)

    scores[i][j] = -infinity  for all j > i

    This is critical: during training, the model processes the whole sequence
    at once, but each position must only see previous tokens (otherwise it
    could just copy the answer).

    ┌                         ┐
    │  0    -inf  -inf  -inf  │   Token 0 sees only itself
    │  0     0    -inf  -inf  │   Token 1 sees tokens 0,1
    │  0     0     0    -inf  │   Token 2 sees tokens 0,1,2
    │  0     0     0     0    │   Token 3 sees all tokens
    └                         ┘

Step 6: Softmax to get attention weights

    weights = softmax(scores)     # each row sums to 1.0

    Example for the `return` token:
    weights = [0.35, 0.02, 0.01, 0.25, ..., 0.05]
               ^^^^                  ^^^^
            "def fib"             "n = ..."
    (high weight = "I'm paying attention to this token")

Step 7: Weighted sum of values

    output = weights @ V     # [seq_len, 64] per head

    The output for `return` is a blend of the value vectors from the
    tokens it attended to, weighted by relevance.

Step 8: Concatenate all 8 heads and project

    output = concat(head_1, ..., head_8)   # [seq_len, 512]
    output = output @ W_O                   # final projection [512, 512]
```

**Attention parameter count per layer:**
```
W_Q:  512 x 512 = 262,144
W_K:  512 x 512 = 262,144
W_V:  512 x 512 = 262,144
W_O:  512 x 512 = 262,144
Total per layer:    1,048,576 (1.05M)
```

#### 4. RoPE (Rotary Position Embeddings)

Without position information, attention is **permutation invariant** -- it can't
tell if `return` comes before or after `def`. RoPE encodes position by rotating
Q and K vectors based on their position in the sequence.

**The math:**

For a pair of dimensions (2i, 2i+1) at position p:

```
            ┌                    ┐   ┌        ┐
            │  cos(p*theta_i)    │   │ q_{2i}  │
RoPE(q, p) =│                    │ * │         │
            │  sin(p*theta_i)    │   │ q_{2i+1}│
            └                    ┘   └        ┘

where theta_i = 1 / (10000 ^ (2i/d))

Applied as a 2D rotation:
┌         ┐     ┌                          ┐   ┌         ┐
│ q'_{2i}  │     │ cos(p*theta)  -sin(p*theta) │   │ q_{2i}  │
│          │  =  │                              │ * │         │
│ q'_{2i+1}│     │ sin(p*theta)   cos(p*theta) │   │ q_{2i+1}│
└         ┘     └                          ┘   └         ┘
```

**Why rotation works:** When computing the dot product Q.K between positions p1 and
p2, the rotation ensures the result depends on (p1 - p2) -- the **relative distance**
between tokens, not their absolute positions. This means the model learns patterns
like "the token 4 positions ago" rather than "the token at position 17."

```
theta_0 = 1/10000^(0/64)  = 1.0       (rotates fast -- encodes nearby positions)
theta_1 = 1/10000^(2/64)  = 0.72      
theta_2 = 1/10000^(4/64)  = 0.52      (rotates slower)
...
theta_31 = 1/10000^(62/64) = 0.0001   (rotates very slowly -- long-range patterns)
```

Low-frequency dimensions encode long-range position (which function am I in?),
high-frequency dimensions encode local position (which line/token?).

#### 5. SwiGLU Feed-Forward Network

After attention mixes information between tokens, the FFN processes each token
independently, adding computational depth.

Standard FFN is: `FFN(x) = ReLU(xW₁) W₂`

Llama uses **SwiGLU**, which adds a gating mechanism:

```
SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down

Where:
  x:        [seq_len, 512]
  W_gate:   [512, 1376]      "gate" projection
  W_up:     [512, 1376]      "up" projection
  W_down:   [1376, 512]      "down" projection
  SiLU:     x * sigmoid(x)   (smooth activation, aka "swish")
  ⊙:        element-wise multiplication
```

**Visualized:**

```
                x (512-dim)
               / \
              /   \
    x @ W_gate    x @ W_up
     (1376)        (1376)
        |             |
      SiLU            |
        |             |
        └─── ⊙ ───┘
              |
           (1376)
              |
         @ W_down
              |
           (512)
              |
           output
```

**Why gating helps:** The gate acts as a learned "filter." SiLU(x @ W_gate) produces
values between 0 and x (it can suppress or amplify). Multiplying by the ungated
path (x @ W_up) means the network can selectively pass information through. This
consistently outperforms standard ReLU FFNs.

**Why intermediate_size = 1376, not 2048 (4x)?** In a standard FFN, you'd use
4 * 512 = 2048. But SwiGLU has THREE weight matrices instead of two. To keep the
parameter count similar, we use: `intermediate_size = (2/3) * 4 * hidden_size`
= (2/3) * 2048 = 1365 ~= 1376 (rounded to a multiple of 32 for GPU efficiency).

**FFN parameter count per layer:**
```
W_gate: 512 x 1376 = 704,512
W_up:   512 x 1376 = 704,512
W_down: 1376 x 512 = 704,512
Total per layer:      2,113,536 (2.1M)
```

#### 6. Residual Connections

Notice in the block diagram: `x = x + attn_out` and `x = x + ffn_out`. The output
of each sub-layer is **added** to its input, not replacing it. This is a residual
(skip) connection.

```
      x ──────────────────┐
      |                    |
   [RMSNorm]               |
      |                    |
   [Attention]             |     Without residual: gradients vanish after
      |                    |     a few layers (signal dies). With residual:
      + <─────────────────┘     gradients flow directly through the + operation,
      |                          enabling 12+ layer training.
      |──────────────────┐
      |                    |
   [RMSNorm]               |
      |                    |
   [SwiGLU FFN]            |
      |                    |
      + <─────────────────┘
      |
   (to next block)
```

Mathematically: if each layer computes `f(x)`, then:
- Without residual: `y = f12(f11(f10(...f1(x)...)))` -- deep composition, vanishing gradients
- With residual: `y = x + f1(x) + f2(x) + ... + f12(x)` -- each layer adds a small delta

This is why modern transformers can be 100+ layers deep.

#### 7. Output Projection and Softmax

After all 12 transformer blocks, we project back to vocabulary size:

```
hidden_state:  [seq_len, 512]      (the model's internal representation)
                  |
             @ W_embed.T            (reuse the embedding matrix, transposed)
                  |
logits:        [seq_len, 8192]     (one score per vocab token)
                  |
              softmax               (convert to probabilities)
                  |
probs:         [seq_len, 8192]     (sums to 1.0 for each position)
```

**Tied embeddings** means W_output = W_embed.T. Why? The embedding says "token 142
is represented as vector [0.02, -0.15, ...]". The output projection asks "which
token best matches this vector?" -- the same question, reversed. Sharing weights
enforces consistency and saves 4.2M parameters.

#### Putting It All Together: One Forward Pass

Let's trace `def fib` through the 50M model:

```
1. Tokenize:     "def fib" -> [142, 3891]

2. Embed:         [142]  -> [0.02, -0.15, 0.08, ..., 0.11]   (512 floats)
                  [3891] -> [-0.04, 0.22, -0.01, ..., 0.07]  (512 floats)

3. Block 1:
   RMSNorm ->    normalize both vectors
   Attention ->  token "fib" attends to "def" (score: 0.8)
                 and to itself (score: 0.2)
                 result: "fib" now carries info about being after "def"
   + residual -> add original "fib" vector back
   RMSNorm ->   normalize again
   SwiGLU ->    transform each token independently (deeper processing)
   + residual -> add back again

4. Blocks 2-12: repeat, building increasingly abstract representations
   - Early layers: syntax (matching brackets, indentation)
   - Middle layers: semantics (variable types, function signatures)
   - Late layers: high-level patterns (algorithm structure, code intent)

5. Output:
   Final hidden state for position 2 (predicting token after "fib"):
   [0.31, -0.82, ..., 0.14]
   -> @ W_embed.T -> logits for all 8192 tokens
   -> softmax -> P("onacci") = 0.67, P("_") = 0.12, P("(") = 0.08, ...
   -> sample or argmax -> "onacci"
   -> The model predicts "fib" continues as "fibonacci"!
```

#### Complete Parameter Summary

```
Component                    Count per    x Layers   Total
──────────────────────────────────────────────────────────
Token Embedding              4,194,304    1          4,194,304
Attention (Q, K, V, O)       1,048,576    12         12,582,912
SwiGLU FFN (gate, up, down)  2,113,536    12         25,362,432
RMSNorm (attn + ffn)         1,024        12         12,288
Final RMSNorm                512          1          512
Output Projection            (tied)       --         0
──────────────────────────────────────────────────────────
TOTAL                                                42,152,448
```

42.15M parameters, exactly what MLX reported.

### Tokenization

Before a model can process text, we convert characters into numbers (tokens).
We train a **Byte-Pair Encoding (BPE)** tokenizer on our Python code corpus:

```
Original: "def fibonacci(n):"
Tokens:   ["def", "Ġfib", "onacci", "(", "n", "):", ]
Token IDs: [142,   3891,   7234,    11,  52,  284]
```

The `Ġ` character represents a leading space (byte-level encoding). The tokenizer
learns to merge common byte pairs -- so `def` becomes a single token because it
appears constantly in Python code.

**Vocab size matters**: Our 50M model uses 8,192 tokens. Larger vocab = each token
covers more text = faster training, but the embedding matrix gets bigger. For a small
model, 8K is a good balance.

### Loss Function: Cross-Entropy

The model is trained by minimizing **cross-entropy loss** -- measuring how far the
model's predicted probability distribution is from the actual next token:

```
loss = -log(P(correct_token))
```

If the model assigns probability 0.9 to the correct token, loss = 0.105 (low, good).
If it assigns probability 0.01, loss = 4.605 (high, bad).

**Perplexity** = e^(loss). It represents "how many tokens the model is confused
between." A perplexity of 100 means the model is as uncertain as randomly guessing
among 100 tokens. Our 50M model started at perplexity ~13,000 (random) and dropped
to ~80-120 after 4,000 steps.

### LoRA (Low-Rank Adaptation)

Instead of retraining all 494M parameters of Qwen2.5-Coder, LoRA freezes the
original weights and adds small trainable matrices:

```
Original: y = Wx        (W is 512x512 = 262,144 params, FROZEN)
LoRA:     y = Wx + BAx  (B is 512x32, A is 32x512 = 32,768 params, TRAINED)

       rank=32
    ┌──────────┐
x ──┤  A (down)├──> 32-dim ──┤ B (up) ├──> added to original output
    └──────────┘              └────────┘
```

This works because weight changes during fine-tuning tend to be **low-rank** -- they
can be approximated by the product of two small matrices. We only train 1.4M params
(0.29%) instead of 494M, making it:
- **50x less memory** during training
- **100x fewer gradient computations**
- Trainable on a MacBook in minutes instead of days

---

## 3. Stage 1: Training a 50M Model from Scratch

### Step 1: Dataset Preparation

We download Python functions from the `code_search_net` dataset (publicly available)
and convert them to JSONL format.

**Why this dataset?** It contains ~457K real Python functions with docstrings from
GitHub. Each example is a self-contained function -- perfect for a 2048-token context
window.

**CLI:**
```bash
python src/data/prepare_python_corpus.py --output-dir data --max-examples 500000
```

**What the script does:**

```python
# Stream Python code from HuggingFace
ds = load_dataset("code_search_net", "python", split="train", streaming=True)

for example in ds:
    code = example.get("whole_func_string", "")

    # Quality filters
    if len(code) < 100 or len(code) > 50000:
        continue               # skip tiny/huge files
    if is_auto_generated(code):
        continue               # skip generated code
    if not deduplicate_hash(code, seen_hashes):
        continue               # skip duplicates

    # Prepend docstring if available
    doc = example.get("func_documentation_string", "")
    if doc:
        text = f'"""{doc}"""\n{code}'
    else:
        text = code

    # Write as JSONL (one JSON object per line)
    f.write(json.dumps({"text": text}) + "\n")
```

**Output format** (`data/python_train.jsonl`):
```json
{"text": "\"\"\"Return the nth Fibonacci number.\"\"\"\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"}
{"text": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    ..."}
```

**Our results:** 403,904 training examples + 8,243 validation (~559MB total)

### Step 2: Tokenizer Training

We train a custom BPE tokenizer specifically on our Python code. This ensures the
vocabulary contains Python-specific tokens like `def`, `class`, `import`, common
indentation patterns, and so on.

**CLI:**
```bash
python mlx-pretrain/train-tokenizer.py --config configs/tokenizer-config.yaml
```

**Config** (`configs/tokenizer-config.yaml`):
```yaml
name: "Python Code BPE Tokenizer"
data:
  input_file: "data/python_train.jsonl"     # our Python corpus
  max_texts_to_train_on: 50000              # sample for speed
  tokenizer:
    special_tokens:
      pad: "<|pad|>"            # padding token (for batching)
      bos: "<|file_start|>"     # beginning-of-sequence
      eos: "<|file_end|>"       # end-of-sequence

tokenizer:
  vocab_size: 8192              # 8K vocabulary
  output_dir: "tokenizer"
```

**How BPE works:**
1. Start with individual bytes (256 tokens)
2. Count all adjacent byte pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until vocab_size is reached

```
Step 0:  [d] [e] [f] [ ] [f] [i] [b]      (individual bytes)
Step 1:  [de] [f] [ ] [f] [i] [b]          (merge 'd'+'e' -> 'de')
Step 2:  [def] [ ] [f] [i] [b]             (merge 'de'+'f' -> 'def')
...
Step N:  [def] [Ġfib] [onacci]             ('def' is now a single token!)
```

**Key insight:** The tokenizer learns that `def`, `return`, `self.`, and `\n    `
(newline + 4 spaces = Python indentation) are so common they deserve their own
tokens. This makes training much more efficient because each token carries more
meaning.

**Result example:**
```
Input:  """Run Graph-Cut segmentation with refinement"""
Tokens: ['"""', 'RunĠ', 'Graph', '-', 'C', 'utĠ', 'segment', 'ationĠ', ...]
IDs:    [234,   6040,   3261,   15,  37,  5336,   2341,       488,     ...]
```

### Step 3: Model Architecture

Our 50M parameter Llama model is configured in YAML:

**Config** (`configs/model-config-python-50m.yaml`):
```yaml
model:
  architecture: "llama"
  dimensions:
    hidden_size: 512          # each token is a 512-dim vector
    intermediate_size: 1376   # FFN inner dimension (~2.7x hidden for SwiGLU)
    num_layers: 12            # 12 transformer blocks stacked
  attention:
    num_heads: 8              # 8 attention heads (64 dim each: 512/8)
    num_kv_heads: null        # null = same as num_heads (no GQA)
    head_dim: null            # null = computed as hidden_size/num_heads
    max_position_embeddings: 2048   # max sequence length
  normalization:
    rms_norm_eps: 1.0e-5      # numerical stability for RMSNorm
  rope:
    theta: 10000              # RoPE base frequency
    traditional: false        # use the standard (non-traditional) formulation
  misc:
    attention_bias: false     # no bias in attention projections
    mlp_bias: false           # no bias in FFN
    tie_word_embeddings: true # share input/output embedding weights
```

**Parameter count breakdown:**

| Component | Formula | Parameters |
|---|---|---|
| Token embedding | vocab_size x hidden_size | 8,192 x 512 = **4.2M** |
| Attention (per layer) | 4 x hidden_size^2 | 4 x 512^2 = **1.05M** |
| FFN (per layer) | 3 x hidden_size x intermediate_size | 3 x 512 x 1376 = **2.11M** |
| Layer norms (per layer) | 2 x hidden_size | 2 x 512 = **1K** |
| **Per layer total** | | **~3.16M** |
| **12 layers** | | **~37.9M** |
| Output projection | *tied with embedding* | **0** (shared) |
| **Total** | | **~42.1M** |

**Why `tie_word_embeddings: true`?** The input embedding (token -> vector) and output
projection (vector -> token probabilities) are the same matrix, transposed. This
saves 4.2M parameters and often improves quality for small models because the input
and output representations stay aligned.

### Step 4: Training Loop

**CLI:**
```bash
python mlx-pretrain/train.py --config configs/model-config-python-50m.yaml
```

The training loop is the core of pre-training. Here's what happens each step:

```python
# 1. Create the loss+gradient function (MLX auto-differentiation)
loss_value_and_grad = nn.value_and_grad(model, compute_loss)

for step in range(total_steps):  # 50,488 steps (1 epoch over 403K examples)

    # 2. Get a batch of tokenized Python code
    batch = data_manager.generate_batch(step)  # shape: [batch_size, seq_len]

    # 3. Teacher forcing: input is tokens[:-1], target is tokens[1:]
    #    The model predicts each next token given all previous ones
    inputs  = batch[:, :-1]   # "def fibonacci(n):\n    if n <=  "
    targets = batch[:, 1:]    # "ef fibonacci(n):\n    if n <= 1"

    # 4. Forward pass + backward pass (compute gradients)
    (loss, num_tokens), gradients = loss_value_and_grad(model, inputs, targets)

    # 5. Update weights using AdamW optimizer
    optimizer.update(model, gradients)

    # 6. Force computation (MLX uses lazy evaluation)
    mx.eval(loss)
```

**Teacher Forcing** is the training strategy: we shift the input by one position and
train the model to predict each token given all preceding tokens. This is why the
model learns left-to-right generation.

**The loss function with padding mask:**

```python
def compute_loss(model, inputs, targets):
    logits = model(inputs)                    # [batch, seq_len, vocab_size]
    logits = logits.astype(mx.float32)        # upcast for numerical stability
    loss = nn.losses.cross_entropy(logits, targets)  # per-token loss

    # Don't count loss on padding tokens
    pad_mask = (targets != PAD_TOKEN)
    loss = loss * pad_mask
    ntoks = pad_mask.sum()

    return loss.sum() / ntoks, ntoks          # average loss over real tokens
```

**Training configuration:**
```yaml
training:
  epochs: 1                           # 1 pass over all data
  hyperparameters:
    batch_size: 8                      # 8 sequences per step
    learning_rate: 3.0e-4              # AdamW learning rate
    weight_decay: 0.01                 # L2 regularization
  scheduler:
    type: "cosine"                     # cosine decay schedule
    min_lr_ratio: 0.01                 # decay to 1% of initial LR
  optimization:
    optimizer: "adamw"                 # Adam with weight decay
```

**Cosine learning rate schedule:** The learning rate starts at 3e-4 and smoothly
decays following a cosine curve to 3e-6 (1% of initial). This prevents the model
from making large, destructive updates late in training.

```
LR   3e-4 |****
           |    ****
           |        ***
           |           ***
           |              ****
     3e-6  |                  *****
           +---------------------------
           Step 0              Step 50K
```

**Our training progress** (real numbers from our run):

| Step | Loss | Perplexity | Observation |
|---|---|---|---|
| 0 | 9.52 | 13,661 | Random -- model guesses uniformly |
| 100 | 8.01 | 3,007 | Learning basic byte frequencies |
| 500 | 6.5 | ~665 | Learning common Python tokens |
| 2000 | 5.2 | ~181 | Learning syntax patterns |
| 4000 | 4.5 | ~90 | Learning code structure |

### Step 5: Evaluation

After training, convert the model and test it:

```bash
# Convert to MLX-LM format (standard HuggingFace-compatible)
python mlx-pretrain/convert-to-mlx-lm.py \
  --run "Python-Llama-50M" \
  --out-path "MLX-Python-50M"

# Interactive generation
python src/eval/interactive.py --run "Python-Llama-50M" --temperature 1.0
>>> def fibonacci(n):

# Formal benchmarks
python src/eval/benchmark.py --model MLX-Python-50M --custom-only
```

**What to expect from a 50M model:** It will learn Python syntax (correct indentation,
valid keywords, matching brackets) and common patterns, but it won't reliably solve
programming problems. That's expected -- this stage is about learning the pipeline.

---

## 4. Stage 2: Distilling a 500M Python Expert

### Step 6: Synthetic Data Generation

Instead of training from scratch, we take a pre-trained model (Qwen2.5-Coder-0.5B,
already trained on 5.5 trillion tokens) and teach it to be a better Python assistant
using high-quality instruction/response pairs.

**Where does the training data come from?**

We used two approaches (no API costs needed):

**Approach A: Corpus-based generation** (5,000 examples)

Transform our existing Python code into instruction-following format:

```python
# Original code from corpus:
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        ...

# Generated training example:
{
  "messages": [
    {"role": "system", "content": "You are a Python expert..."},
    {"role": "user", "content": "Explain what this Python code does:\n```python\ndef binary_search(arr, target):\n...```"},
    {"role": "assistant", "content": "This code defines a function `binary_search` that...\n\n**Key details:**\n- Uses iteration to process data\n- Takes parameters: arr, target\n- Returns a computed result"}
  ]
}
```

**CLI:**
```bash
python src/data/generate_from_corpus.py --max-examples 5000
```

The script creates four types of examples from the same code:
- **Explanation** (35%): "Explain what this code does"
- **Docstring** (25%): "Add a docstring to this function"
- **Testing** (25%): "Write pytest tests for this function"
- **Type hints** (15%): "Add type hints to this function"

**Approach B: Claude-generated via Cursor** (200 examples)

Used Cursor's built-in Claude to generate high-quality novel examples that require
real intelligence -- code generation, debugging, code review, and advanced patterns.

These cover:
- **Code generation** (50): implementing data structures, algorithms, utilities
- **Debugging** (50): finding and fixing real bugs (off-by-one, thread safety, etc.)
- **Code review + testing** (50): reviewing code quality, writing pytest suites
- **Advanced patterns** (50): design patterns, API endpoints, async, caching

**Data format** (`data/synthetic/train.jsonl`):

Every line is a JSON object in **ChatML format** -- the standard format for
instruction-tuned models:

```json
{"messages": [
  {"role": "system", "content": "You are a Python expert..."},
  {"role": "user", "content": "Write a Python function that implements a LRU cache with O(1) get and put operations."},
  {"role": "assistant", "content": "Here's an implementation using OrderedDict:\n\n```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int) -> None:\n        self.cache: OrderedDict[str, Any] = OrderedDict()\n        self.capacity = capacity\n    ...\n```"}
]}
```

**Final dataset:** 4,940 train + 260 validation examples

### Step 7: LoRA Fine-tuning

This is where we specialize the pre-trained Qwen2.5-Coder-0.5B using our
synthetic data and LoRA adapters.

**Base model: Qwen2.5-Coder-0.5B-Instruct**
- 494M parameters (360M non-embedding)
- Already trained on 5.5 trillion tokens of code + text
- 32K context window, instruction-tuned
- We are NOT retraining it -- just adding lightweight adapters

**CLI:**
```bash
python -m mlx_lm lora \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --data ./data/synthetic/ \
  --train \
  --batch-size 4 \
  --num-layers 16 \
  --learning-rate 1e-4 \
  --iters 1000 \
  --val-batches 25 \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --save-every 200 \
  --adapter-path ./adapters \
  -c configs/lora-config.yaml
```

**LoRA config** (`configs/lora-config.yaml`):
```yaml
lora_parameters:
  rank: 32       # rank of the low-rank matrices (higher = more capacity)
  alpha: 64      # scaling factor (alpha/rank = scale applied to LoRA output)
  dropout: 0.05  # dropout on LoRA layers (regularization)
  scale: 2.0     # effective scale = alpha/rank = 64/32 = 2.0
```

**How rank affects quality:**

| Rank | Trainable Params | Capacity | When to use |
|---|---|---|---|
| 8 | ~360K | Low | Quick experiments, narrow tasks |
| 16 | ~720K | Medium | Good default for most fine-tuning |
| 32 | ~1.4M | High | Our choice -- more complex Python skills |
| 64 | ~2.9M | Very high | When you have lots of data (50K+) |

**Our training results:**

```
Trainable parameters: 0.292% (1.442M / 494.033M)

Iter   1: Val loss 1.656              <-- baseline (no fine-tuning)
Iter  10: Train loss 1.413
Iter  20: Train loss 1.107            <-- rapid improvement
Iter  50: Train loss 0.802
Iter 100: Val loss 0.771              <-- 53% reduction in val loss
Iter 200: Val loss 0.738, saved checkpoint
Iter 270: Train loss 0.693            <-- crashed (OOM, two jobs running)

Peak memory: 28.2 GB
Speed: ~0.5-0.8 iterations/second
```

The model improved dramatically in just 200 iterations (about 15 minutes). The
validation loss dropped from 1.656 to 0.738 -- meaning the model became significantly
better at generating Python code in our instruction format.

**To resume from checkpoint:**
```bash
python -m mlx_lm lora \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --data ./data/synthetic/ \
  --train \
  --batch-size 4 \
  --num-layers 16 \
  --learning-rate 1e-4 \
  --iters 1000 \
  --resume-adapter-file ./adapters/0000200_adapters.safetensors \
  --adapter-path ./adapters \
  -c configs/lora-config.yaml
```

### Step 8: Merging and Evaluation

After fine-tuning, merge the LoRA adapters back into the base model to get a
standalone model:

```bash
# Fuse LoRA weights into the base model
python -m mlx_lm.fuse \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --adapter-path ./adapters \
  --save-path ./fused_model

# Test interactively
python -m mlx_lm.generate \
  --model ./fused_model \
  --prompt "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes."

# Run benchmarks
python src/eval/benchmark.py --model ./fused_model --custom-only
```

**What merging does:**

```
Before:  output = W_original @ x + (B @ A) @ x    (two separate computations)
After:   W_merged = W_original + B @ A             (single merged matrix)
         output = W_merged @ x                     (one computation, same result)
```

The merged model is identical in behavior but no longer needs the adapter files --
it's a standard model you can share, deploy, or convert.

---

## 5. Stage 3: Publishing to Ollama

### Step 9: GGUF Conversion and Quantization

**GGUF** is the format used by llama.cpp and Ollama. It's a single-file format
optimized for inference.

**Quantization** reduces the model size by using fewer bits per parameter:

| Format | Bits | Size (500M model) | Quality | Use case |
|---|---|---|---|---|
| F16 | 16 | ~1 GB | Original | Reference |
| Q8_0 | 8 | ~500 MB | 99% of original | When you have memory |
| Q5_K_M | 5 | ~350 MB | ~97% of original | Good balance |
| Q4_K_M | 4 | ~250 MB | ~95% of original | Best for distribution |

**How quantization works (Q4_K_M):**

```
Original (float16):  [0.0312, -0.1445, 0.0898, -0.0527, ...]  (16 bits each)
Quantized (4-bit):   [2, -9, 6, -3, ...]  + scale=0.016       (4 bits each)

Reconstruction:      2*0.016=0.032, -9*0.016=-0.144, ...       (close enough!)
```

The `K_M` means "K-quant medium" -- it uses different bit widths for different
layers. Attention layers get more bits (they're more sensitive), FFN layers get
fewer bits.

**CLI:**
```bash
# Convert HuggingFace model to GGUF
python llama.cpp/convert_hf_to_gguf.py ./fused_model \
  --outfile models/gguf/python-expert-f16.gguf

# Quantize to 4-bit
./llama.cpp/build/bin/llama-quantize \
  models/gguf/python-expert-f16.gguf \
  models/gguf/python-expert-Q4_K_M.gguf \
  Q4_K_M
```

### Step 10: Ollama Packaging

Ollama uses a **Modelfile** (like a Dockerfile for models) to define how the model
should be served:

```dockerfile
# The quantized model file
FROM ./python-expert-Q4_K_M.gguf

# Chat template -- MUST match what the model was trained with
# Qwen2.5 uses ChatML format:
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""

# Default system prompt
SYSTEM """You are a Python expert assistant. You write clean, idiomatic,
well-tested Python code. You explain your reasoning clearly and follow
PEP 8 conventions."""

# Inference parameters
PARAMETER temperature 0.2     # low = more deterministic (good for code)
PARAMETER top_p 0.9           # nucleus sampling threshold
PARAMETER stop "<|im_end|>"   # stop generating at this token
PARAMETER num_ctx 4096        # context window size
```

**Build and test locally:**
```bash
ollama create python-expert -f Modelfile
ollama run python-expert
>>> Write a Python function to validate an email address
```

**Publish to Ollama registry:**
```bash
# Create account at ollama.com, add SSH key
ollama push yourname/python-expert
# Now anyone can: ollama run yourname/python-expert
```

---

## 6. Appendix: Key Formulas and Numbers

### Parameter Count

```
params = vocab_size * hidden_size                          # embedding
       + num_layers * (
           4 * hidden_size^2                               # attention (Q,K,V,O)
           + 3 * hidden_size * intermediate_size           # SwiGLU FFN
           + 2 * hidden_size                               # RMSNorm
         )
```

### Memory During Training

```
Training memory ~= 4 * model_size_in_bytes
  = model weights (2 bytes per param in BF16)
  + gradients (2 bytes per param)
  + optimizer state (8 bytes per param for Adam: momentum + variance)
  + activations (variable, depends on batch size and seq length)
```

For our 42M model: ~42M * 12 bytes = ~500MB (plus activations ~1-2GB)
For LoRA on 494M: only 1.4M trained, but base model loaded: ~1GB + ~100MB

### Training Compute (Chinchilla Scaling)

The "Chinchilla optimal" ratio suggests:
```
optimal_tokens ~= 20 * num_parameters
```

For 42M params: ~840M tokens is optimal.
We trained on ~400K examples * ~400 tokens/example = ~160M tokens.
This is under-trained by Chinchilla standards, but fine for a learning project.

### Tokens Per Second

```
Our 50M model on M4 Pro: ~500 tokens/sec (training)
LoRA on 500M model:      ~1,000 tokens/sec (training)
Inference on 500M model:  ~50-100 tokens/sec (generation)
```

### File Sizes

| What | Size |
|---|---|
| Python corpus (train) | 548 MB |
| Synthetic data (train) | ~5 MB |
| BPE tokenizer | 240 KB |
| 50M model weights (BF16) | ~84 MB |
| 500M model weights (BF16) | ~988 MB |
| LoRA adapter weights | 5.5 MB |
| 500M quantized (Q4_K_M) | ~250 MB |

### Project File Tree

```
SLM/
├── configs/
│   ├── lora-config.yaml            # LoRA hyperparameters
│   ├── model-config-python-50m.yaml # 50M model architecture + training
│   └── tokenizer-config.yaml        # BPE tokenizer config
├── src/
│   ├── data/
│   │   ├── prepare_python_corpus.py  # Download code_search_net -> JSONL
│   │   ├── generate_from_corpus.py   # Corpus -> instruction pairs (free)
│   │   └── generate_synthetic.py     # Claude API -> instruction pairs
│   ├── eval/
│   │   ├── benchmark.py              # HumanEval + custom Python tests
│   │   └── interactive.py            # REPL for testing models
│   └── publish/
│       └── convert_and_publish.sh    # Fuse -> GGUF -> quantize -> Ollama
├── data/
│   ├── python_train.jsonl            # 403K Python functions
│   ├── python_val.jsonl              # 8K validation functions
│   └── synthetic/
│       ├── train.jsonl               # 4,940 instruction/response pairs
│       └── valid.jsonl               # 260 validation pairs
├── tokenizer/
│   └── tokenizer.json                # Trained BPE tokenizer (8K vocab)
├── adapters/
│   ├── adapter_config.json           # LoRA config used
│   └── adapters.safetensors          # Trained LoRA weights (5.5MB)
├── runs/
│   └── Python-Llama-50M/             # 50M training checkpoints + logs
├── mlx-pretrain/                     # Pre-training framework
├── requirements.txt
└── README.md
```
