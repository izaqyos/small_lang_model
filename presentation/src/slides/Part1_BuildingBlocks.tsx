import React from 'react';
import { Math, MathBlock } from '../components/MathBlock';
import { CodeBlock } from '../components/CodeBlock';

export const part1Slides: React.FC[] = [
  function TitleSlide() {
    return (
      <div className="title-slide">
        <h1>Building a Python Expert SLM</h1>
        <h2>From scratch to Ollama in 50M parameters</h2>
        <div className="tech-badges">
          <span className="badge">MLX</span>
          <span className="badge">Apple Silicon</span>
          <span className="badge">Python</span>
          <span className="badge">HuggingFace</span>
          <span className="badge">LoRA</span>
          <span className="badge">Ollama</span>
        </div>
      </div>
    );
  },

  function WhatIsLM() {
    return (
      <>
        <h2 className="slide-title">What is a Language Model?</h2>
        <div className="slide-content">
          <CodeBlock
            title="Core concept: next-token prediction"
          >{`Input:  "def fibonacci(n):\\n    if n <= 1:\\n        return"
Model:  P(next_token | all_previous_tokens)  
Output: " n"  →  the model predicts the most likely continuation`}</CodeBlock>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="bullet">
              Everything -- code generation, chat, reasoning -- emerges from this
            </li>
            <li className="bullet">
              Training objective: minimize cross-entropy loss on next-token prediction
            </li>
          </ul>
        </div>
      </>
    );
  },

  function Tokenization() {
    return (
      <>
        <h2 className="slide-title">Tokenization: Text to Numbers</h2>
        <div className="slide-content">
          <CodeBlock>{`Original:  "def fibonacci(n):"
                ↓  Byte-Pair Encoding
Tokens:    ["def", "Ġfib", "onacci", "(", "n", "):"]
Token IDs: [ 142,   3891,    7234,   11,  52,  284 ]`}</CodeBlock>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="bullet">
              Ġ = leading space (byte-level)
            </li>
            <li className="bullet">
              BPE merges frequent byte pairs: d+e→de, de+f→def
            </li>
            <li className="bullet">
              Vocab size 8,192 -- Python keywords get their own tokens
            </li>
          </ul>
        </div>
      </>
    );
  },

  function TransformerOverview() {
    return (
      <>
        <h2 className="slide-title">The Transformer Architecture</h2>
        <div className="slide-content">
          <div className="col" style={{ maxWidth: 480, margin: '0 auto' }}>
            <div className="pipeline-step">Input Token IDs</div>
            <div className="flow-arrow">↓</div>
            <div className="pipeline-step">Embedding Layer (8192 × 512)</div>
            <div className="flow-arrow">↓</div>
            <div className="pipeline-step">
              Transformer Block × 12
              <div style={{ fontSize: '0.85rem', marginTop: 8, color: '#9090a8' }}>
                RMSNorm → Self-Attention + RoPE → RMSNorm → SwiGLU FFN
              </div>
            </div>
            <div className="flow-arrow">↓</div>
            <div className="pipeline-step">Final RMSNorm</div>
            <div className="flow-arrow">↓</div>
            <div className="pipeline-step">Output Projection → Softmax</div>
            <div className="flow-arrow">↓</div>
            <div className="pipeline-step">Next Token Probabilities</div>
          </div>
        </div>
      </>
    );
  },

  function EmbeddingLayer() {
    return (
      <>
        <h2 className="slide-title">Embedding: Tokens as Vectors</h2>
        <div className="slide-content">
          <CodeBlock>{`Token 142 ("def")   → [0.02, -0.15, 0.08, ..., 0.11]  (512 floats)
Token 3891 ("Ġfib") → [-0.04, 0.22, -0.01, ..., 0.07] (512 floats)

W_embed shape: [8192, 512]  →  4.2M parameters`}</CodeBlock>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="bullet">
              Just a lookup table -- no computation
            </li>
            <li className="bullet">
              Vectors learned during training: similar tokens cluster together
            </li>
            <li className="bullet">
              for/while cluster together, def/class cluster together
            </li>
          </ul>
        </div>
      </>
    );
  },

  function SelfAttention() {
    return (
      <>
        <h2 className="slide-title">Self-Attention: How Tokens Communicate</h2>
        <div className="slide-content row">
          <div className="col">
            <MathBlock>{`\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V`}</MathBlock>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li className="bullet">Q = &quot;What am I looking for?&quot;</li>
              <li className="bullet">K = &quot;What do I contain?&quot;</li>
              <li className="bullet">V = &quot;What info do I carry?&quot;</li>
              <li className="bullet">8 heads × 64 dims = 512</li>
            </ul>
          </div>
          <div className="col">
            <CodeBlock title="Causal Mask">{`┌                     ┐
│  ✓   ✗   ✗   ✗     │  Token 0 → sees self only
│  ✓   ✓   ✗   ✗     │  Token 1 → sees 0,1
│  ✓   ✓   ✓   ✗     │  Token 2 → sees 0,1,2
│  ✓   ✓   ✓   ✓     │  Token 3 → sees all
└                     ┘`}</CodeBlock>
          </div>
        </div>
      </>
    );
  },

  function RoPE() {
    return (
      <>
        <h2 className="slide-title">RoPE: Position Through Rotation</h2>
        <div className="slide-content">
          <MathBlock>{`R(\\theta, p) = \\begin{pmatrix} \\cos(p\\theta) & -\\sin(p\\theta) \\\\ \\sin(p\\theta) & \\cos(p\\theta) \\end{pmatrix}`}</MathBlock>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="bullet">
              Rotates Q and K vectors based on position
            </li>
            <li className="bullet">
              Dot product Q·K depends on relative distance (p₁ - p₂)
            </li>
            <li className="bullet">
              High frequencies → local position, Low frequencies → long-range
            </li>
          </ul>
          <p style={{ marginTop: 16 }}>
            Theta formula: <Math>{`\\theta_i = 10000^{-2i/d}`}</Math>
          </p>
        </div>
      </>
    );
  },

  function SwiGLU() {
    return (
      <>
        <h2 className="slide-title">SwiGLU: The Feed-Forward Network</h2>
        <div className="slide-content">
          <MathBlock>{`\\text{SwiGLU}(x) = (\\text{SiLU}(xW_{\\text{gate}}) \\odot xW_{\\text{up}}) \\cdot W_{\\text{down}}`}</MathBlock>
          <CodeBlock>{`        x (512)
       / \\
  W_gate   W_up
  (1376)   (1376)
    |        |
  SiLU       |
    \\       /
      ⊙ (element-wise)
      |
   W_down
   (512)`}</CodeBlock>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="bullet">
              3 weight matrices → intermediate = ⅔ × 4 × 512 = 1376
            </li>
          </ul>
        </div>
      </>
    );
  },

  function ResidualRMSNorm() {
    return (
      <>
        <h2 className="slide-title">Residual Connections &amp; RMSNorm</h2>
        <div className="slide-content row">
          <div className="col">
            <h3 className="slide-subtitle">Residual Connections</h3>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li className="bullet">
                Output = input + sublayer(input)
              </li>
              <li className="bullet">
                Enables gradient flow through 12+ layers
              </li>
              <li className="bullet">
                Without them: gradient vanishes, training fails
              </li>
            </ul>
          </div>
          <div className="col">
            <h3 className="slide-subtitle">RMSNorm</h3>
            <MathBlock>{`\\text{RMSNorm}(x)_i = \\frac{x_i}{\\text{RMS}(x)} \\cdot \\gamma_i`}</MathBlock>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li className="bullet">
                Simpler than LayerNorm (no mean subtraction), ~15% faster
              </li>
            </ul>
          </div>
        </div>
      </>
    );
  },

  function TrainingEssentials() {
    return (
      <>
        <h2 className="slide-title">Training: Loss, Optimizer, Schedule</h2>
        <div className="slide-content">
          <p>
            Cross-entropy loss: <Math>{`\\mathcal{L} = -\\log P(\\text{correct token})`}</Math>
          </p>
          <p>
            Perplexity: <Math>{`\\text{PPL} = e^{\\mathcal{L}}`}</Math>
          </p>
          <CodeBlock title="Cosine schedule">{`LR  3e-4 |████
          |    ████
          |        ███         Cosine decay
          |           ███
          |              ████
    3e-6  |                  █████
          +──────────────────────── Steps`}</CodeBlock>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="bullet">
              AdamW optimizer with weight decay 0.01
            </li>
          </ul>
        </div>
      </>
    );
  },

  function LoRA() {
    return (
      <>
        <h2 className="slide-title">LoRA: Efficient Fine-tuning</h2>
        <div className="slide-content">
          <MathBlock>{`y = Wx + BAx \\quad \\text{where } B \\in \\mathbb{R}^{d \\times r}, A \\in \\mathbb{R}^{r \\times d}`}</MathBlock>
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Trainable Params</th>
                <th>% of Model</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>8</td>
                <td>360K</td>
                <td>0.07%</td>
              </tr>
              <tr>
                <td>32</td>
                <td>1.4M</td>
                <td>0.29%</td>
              </tr>
              <tr>
                <td>64</td>
                <td>2.9M</td>
                <td>0.59%</td>
              </tr>
            </tbody>
          </table>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li className="bullet">
              Freeze 494M base params, train 1.4M adapter params
            </li>
            <li className="bullet">
              Key insight: fine-tuning updates are low-rank
            </li>
          </ul>
        </div>
      </>
    );
  },

  function TechStack() {
    return (
      <>
        <h2 className="slide-title">Our Tech Stack</h2>
        <div className="slide-content">
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">MLX</div>
              <div className="stat-label">Apple ML Framework</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">M4 Pro</div>
              <div className="stat-label">48GB Unified Memory</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">HuggingFace</div>
              <div className="stat-label">Datasets + Transformers</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">llama.cpp</div>
              <div className="stat-label">GGUF Conversion</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">Ollama</div>
              <div className="stat-label">Local Model Serving</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">Python 3.11</div>
              <div className="stat-label">Runtime</div>
            </div>
          </div>
        </div>
      </>
    );
  },
];
