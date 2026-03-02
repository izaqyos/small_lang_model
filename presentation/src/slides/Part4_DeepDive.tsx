import React from 'react';
import { Math, MathBlock } from '../components/MathBlock';
import { CodeBlock } from '../components/CodeBlock';

export const part4Slides: React.FC[] = [
  function SectionTitle() {
    return (
      <div className="section-title-slide">
        <div className="section-number">04</div>
        <h2>Deep Dive</h2>
        <p>The math and mechanics behind each component</p>
      </div>
    );
  },

  function AttentionStepByStep() {
    return (
      <>
        <h2 className="slide-title">Self-Attention: Step by Step</h2>
        <div className="slide-content row">
          <div className="col">
            <p className="bullet">
              <strong>Step 1:</strong> Project to Q, K, V — <Math>{`Q = xW_Q, \\; K = xW_K, \\; V = xW_V`}</Math>
            </p>
            <p className="bullet">
              <strong>Step 2:</strong> Compute scores — <Math>{`\\text{score}(i,j) = \\frac{q_i \\cdot k_j}{\\sqrt{d_k}}`}</Math>
            </p>
          </div>
          <div className="col">
            <p className="bullet">
              <strong>Step 3:</strong> Mask + Softmax — <Math>{`\\alpha_{ij} = \\text{softmax}(\\text{score}_{ij} + \\text{mask}_{ij})`}</Math>
            </p>
            <p className="bullet">
              <strong>Step 4:</strong> Weighted sum — <Math>{`\\text{out}_i = \\sum_j \\alpha_{ij} v_j`}</Math>
            </p>
          </div>
        </div>
        <div className="slide-content">
          <CodeBlock title="Example: &quot;return&quot; attending to previous tokens">{`Scores:  [0.35, 0.02, 0.01, 0.25, ..., 0.05]
         "def"  "("   ")"  "n=.."     "if"
→ High attention to "def" (function context)
→ High attention to "n=.." (return value)`}</CodeBlock>
        </div>
      </>
    );
  },

  function RoPEDeepDive() {
    return (
      <>
        <h2 className="slide-title">RoPE: The Math of Position</h2>
        <div className="slide-content">
          <MathBlock>{`\\begin{pmatrix} q'_{2i} \\\\ q'_{2i+1} \\end{pmatrix} = \\begin{pmatrix} \\cos(p\\theta_i) & -\\sin(p\\theta_i) \\\\ \\sin(p\\theta_i) & \\cos(p\\theta_i) \\end{pmatrix} \\begin{pmatrix} q_{2i} \\\\ q_{2i+1} \\end{pmatrix}`}</MathBlock>
          <CodeBlock title="Frequency spectrum">{`θ₀  = 1/10000^(0/64)  = 1.0      → rotates fast  (local position)
θ₁  = 1/10000^(2/64)  = 0.72
θ₂  = 1/10000^(4/64)  = 0.52     → medium
...
θ₃₁ = 1/10000^(62/64) = 0.0001   → rotates slow  (long-range)`}</CodeBlock>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="bullet">
              Key property: Q·K dot product depends only on relative distance (p₁ - p₂)
            </li>
          </ul>
        </div>
      </>
    );
  },

  function SwiGLUInternals() {
    return (
      <>
        <h2 className="slide-title">SwiGLU: Why Gating Works</h2>
        <div className="slide-content">
          <MathBlock>{`\\text{SwiGLU}(x) = \\big(\\underbrace{\\sigma(xW_g) \\cdot xW_g}_{\\text{SiLU}(xW_g)} \\odot xW_u\\big) W_d`}</MathBlock>
          <p style={{ marginBottom: 12 }}>
            where <Math>{`\\sigma(z) = \\frac{1}{1+e^{-z}}`}</Math> (sigmoid).
          </p>
          <p style={{ marginBottom: 16 }}>
            Parameter count: <Math>{`3 \\times 512 \\times 1376 = 2.11\\text{M per layer}`}</Math>
          </p>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="bullet">
              Gate learns to suppress/amplify features selectively
            </li>
            <li className="bullet">
              SiLU = x · sigmoid(x), smooth and non-monotonic
            </li>
            <li className="bullet">
              intermediate_size = ⅔ × 4 × hidden = 1376 (compensates for 3rd matrix)
            </li>
          </ul>
        </div>
      </>
    );
  },

  function TrainingLoop() {
    return (
      <>
        <h2 className="slide-title">The Training Loop</h2>
        <div className="slide-content">
          <CodeBlock title="mlx-pretrain/train.py">{`loss_and_grad = nn.value_and_grad(model, compute_loss)

for step in range(50_488):          # 1 epoch over 403K examples
    batch = data.generate_batch()    # [batch_size=8, seq_len=2048]
    
    inputs  = batch[:, :-1]          # "def fibonacci(n):\\n    if n <=  "
    targets = batch[:, 1:]           # "ef fibonacci(n):\\n    if n <= 1"
    
    (loss, ntoks), grads = loss_and_grad(model, inputs, targets)
    optimizer.update(model, grads)   # AdamW: lr=3e-4, wd=0.01
    mx.eval(loss)                    # MLX lazy eval: forces computation`}</CodeBlock>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="bullet">
              Teacher forcing: predict each token from all previous tokens
            </li>
            <li className="bullet">
              MLX lazy evaluation: builds compute graph, mx.eval() triggers GPU execution
            </li>
          </ul>
        </div>
      </>
    );
  },

  function LossCurve() {
    return (
      <>
        <h2 className="slide-title">Training Dynamics</h2>
        <div className="slide-content">
          <table>
            <thead>
              <tr>
                <th>Step</th>
                <th>Loss</th>
                <th>PPL</th>
                <th>What&apos;s Being Learned</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>0</td>
                <td>9.52</td>
                <td>13,661</td>
                <td>Random — uniform guessing</td>
              </tr>
              <tr>
                <td>100</td>
                <td>8.01</td>
                <td>3,007</td>
                <td>Byte frequencies</td>
              </tr>
              <tr>
                <td>500</td>
                <td>6.5</td>
                <td>665</td>
                <td>Common Python tokens</td>
              </tr>
              <tr>
                <td>2,000</td>
                <td>5.2</td>
                <td>181</td>
                <td>Syntax patterns</td>
              </tr>
              <tr>
                <td>4,000</td>
                <td>4.5</td>
                <td>90</td>
                <td>Code structure</td>
              </tr>
              <tr>
                <td>8,000</td>
                <td>3.0</td>
                <td>20</td>
                <td>Real patterns emerging</td>
              </tr>
            </tbody>
          </table>
          <MathBlock>{`\\text{Perplexity} = e^{\\text{loss}} \\quad \\Rightarrow \\quad \\text{PPL of 20} = \\text{choosing among 20 tokens}`}</MathBlock>
        </div>
      </>
    );
  },

  function LoRAMatrixMath() {
    return (
      <>
        <h2 className="slide-title">LoRA: Low-Rank Decomposition</h2>
        <div className="slide-content">
          <MathBlock>{`W' = W + \\frac{\\alpha}{r} \\cdot BA`}</MathBlock>
          <p style={{ marginBottom: 12 }}>
            where <Math>{`W \\in \\mathbb{R}^{512 \\times 512}`}</Math> is frozen, <Math>{`B \\in \\mathbb{R}^{512 \\times 32}`}</Math>, <Math>{`A \\in \\mathbb{R}^{32 \\times 512}`}</Math>.
          </p>
          <p style={{ marginBottom: 12 }}>
            <strong>Original:</strong> 512 × 512 = 262,144 params (frozen)
          </p>
          <p style={{ marginBottom: 16 }}>
            <strong>LoRA:</strong> (512 × 32) + (32 × 512) = 32,768 params (trained) → 8× fewer params per layer
          </p>
          <CodeBlock title="Merge for inference">{`Before: y = Wx + (α/r)·BAx    (inference: two computations)
After:  W_merged = W + (α/r)·BA  (merge once)
        y = W_merged · x          (inference: one computation, same result)`}</CodeBlock>
        </div>
      </>
    );
  },

  function ParameterCounting() {
    return (
      <>
        <h2 className="slide-title">Complete Parameter Inventory</h2>
        <div className="slide-content">
          <table>
            <thead>
              <tr>
                <th>Component</th>
                <th>Formula</th>
                <th>Per Layer</th>
                <th>× Layers</th>
                <th>Total</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Embedding</td>
                <td>8192 × 512</td>
                <td>—</td>
                <td>1</td>
                <td>4,194,304</td>
              </tr>
              <tr>
                <td>Attention Q,K,V,O</td>
                <td>4 × 512²</td>
                <td>1,048,576</td>
                <td>12</td>
                <td>12,582,912</td>
              </tr>
              <tr>
                <td>SwiGLU FFN</td>
                <td>3 × 512 × 1376</td>
                <td>2,113,536</td>
                <td>12</td>
                <td>25,362,432</td>
              </tr>
              <tr>
                <td>RMSNorm</td>
                <td>2 × 512</td>
                <td>1,024</td>
                <td>12</td>
                <td>12,288</td>
              </tr>
              <tr>
                <td>Final RMSNorm</td>
                <td>512</td>
                <td>—</td>
                <td>1</td>
                <td>512</td>
              </tr>
              <tr>
                <td>Output (tied)</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>0</td>
              </tr>
              <tr>
                <td><strong>Total</strong></td>
                <td></td>
                <td></td>
                <td></td>
                <td><strong>42,152,448</strong></td>
              </tr>
            </tbody>
          </table>
          <ul style={{ listStyle: 'none', padding: 0, marginTop: 16 }}>
            <li className="bullet">
              Exactly matches MLX&apos;s reported parameter count: 42.15M
            </li>
          </ul>
        </div>
      </>
    );
  },
];
