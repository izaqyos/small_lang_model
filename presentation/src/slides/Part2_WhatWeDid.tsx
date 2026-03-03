import React from 'react';
import { CodeBlock } from '../components/CodeBlock';

export const part2Slides: React.FC[] = [
  function SectionTitle() {
    return (
      <div className="section-title-slide">
        <div className="section-number">02</div>
        <h2>What We Built</h2>
        <p>A Python expert model trained on a MacBook Pro</p>
      </div>
    );
  },

  function TheStrategy() {
    return (
      <>
        <h2 className="slide-title">Strategy: Learn, Then Build</h2>
        <div className="slide-content row">
          <div className="col">
            <h3 className="slide-subtitle">Option D: Train 50M from scratch</h3>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li className="bullet">Learn the pipeline end-to-end</li>
              <li className="bullet">Understand tokenizers, training, evaluation</li>
              <li className="bullet">Acceptable that it won&apos;t be great</li>
            </ul>
          </div>
          <div className="col" style={{ alignItems: 'center', justifyContent: 'center' }}>
            <div className="flow-arrow">lessons learned →</div>
          </div>
          <div className="col">
            <h3 className="slide-subtitle">Option C: Distill 500M via LoRA</h3>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li className="bullet">Take pre-trained Qwen2.5-Coder-0.5B</li>
              <li className="bullet">Generate synthetic data from Claude</li>
              <li className="bullet">Fine-tune with LoRA → real quality model</li>
            </ul>
          </div>
        </div>
      </>
    );
  },

  function ThreeStagePipeline() {
    return (
      <>
        <h2 className="slide-title">The Three-Stage Pipeline</h2>
        <div className="slide-content">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div className="pipeline-step" style={{ minWidth: 200 }}>
              <div className="pipeline-label">Stage 1</div>
              <div className="pipeline-value">Learn the Pipeline</div>
              <div style={{ fontSize: '0.8rem', color: '#9090a8', marginTop: 8 }}>
                Raw Python Code → BPE Tokenizer → Train 50M Llama → Evaluate
              </div>
            </div>
            <div className="flow-arrow">→</div>
            <div className="pipeline-step" style={{ minWidth: 200 }}>
              <div className="pipeline-label">Stage 2</div>
              <div className="pipeline-value">Build the Real Model</div>
              <div style={{ fontSize: '0.8rem', color: '#9090a8', marginTop: 8 }}>
                Claude (Teacher) → 5K Synthetic Examples → LoRA Fine-tune Qwen2.5 → Evaluate
              </div>
            </div>
            <div className="flow-arrow">→</div>
            <div className="pipeline-step" style={{ minWidth: 200 }}>
              <div className="pipeline-label">Stage 3</div>
              <div className="pipeline-value">Ship It</div>
              <div style={{ fontSize: '0.8rem', color: '#9090a8', marginTop: 8 }}>
                Merge LoRA → Convert GGUF → Quantize Q4_K_M → Ollama Registry
              </div>
            </div>
          </div>
        </div>
      </>
    );
  },

  function Stage1FromScratch() {
    return (
      <>
        <h2 className="slide-title">Stage 1: Training 50M from Scratch</h2>
        <div className="slide-content row">
          <div className="col">
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li className="bullet">Dataset: code_search_net (403K Python functions)</li>
              <li className="bullet">Tokenizer: BPE with 8,192 vocab</li>
              <li className="bullet">Architecture: 12-layer Llama (512 hidden, 8 heads)</li>
              <li className="bullet">Context: 2,048 tokens</li>
            </ul>
          </div>
          <div className="col">
            <table>
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Loss</th>
                  <th>Perplexity</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>0</td>
                  <td>9.52</td>
                  <td>13,661</td>
                  <td>Random</td>
                </tr>
                <tr>
                  <td>500</td>
                  <td>6.5</td>
                  <td>~665</td>
                  <td>Learning tokens</td>
                </tr>
                <tr>
                  <td>4000</td>
                  <td>4.5</td>
                  <td>~90</td>
                  <td>Learning syntax</td>
                </tr>
                <tr>
                  <td>8000</td>
                  <td>~3.0</td>
                  <td>~20</td>
                  <td>Learning patterns</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </>
    );
  },

  function Stage2SyntheticData() {
    return (
      <>
        <h2 className="slide-title">Stage 2: Generating Training Data</h2>
        <div className="slide-content">
          <div className="slide-content row">
            <div className="col">
              <h3 className="slide-subtitle">Corpus-based (Free)</h3>
              <p style={{ marginBottom: 12 }}>5,000 examples from existing code</p>
              <ul style={{ listStyle: 'none', padding: 0 }}>
                <li className="bullet">Explanation 35%</li>
                <li className="bullet">Docstring 25%</li>
                <li className="bullet">Testing 25%</li>
                <li className="bullet">Type hints 15%</li>
              </ul>
            </div>
            <div className="col">
              <h3 className="slide-subtitle">Claude via Cursor (Free)</h3>
              <p style={{ marginBottom: 12 }}>200 high-quality examples</p>
              <ul style={{ listStyle: 'none', padding: 0 }}>
                <li className="bullet">Code generation</li>
                <li className="bullet">Debugging</li>
                <li className="bullet">Code review</li>
                <li className="bullet">Advanced patterns</li>
              </ul>
            </div>
          </div>
          <CodeBlock title="ChatML format">{`{"messages": [
  {"role": "system", "content": "You are a Python expert..."},
  {"role": "user", "content": "Write a function that..."},
  {"role": "assistant", "content": "def merge_sort(arr):..."}
]}`}</CodeBlock>
        </div>
      </>
    );
  },

  function Stage2LoRA() {
    return (
      <>
        <h2 className="slide-title">LoRA Fine-tuning Results</h2>
        <div className="slide-content">
          <p style={{ marginBottom: 16 }}>
            <span className="highlight">Qwen2.5-Coder-0.5B-Instruct</span> (494M params)
          </p>
          <ul style={{ listStyle: 'none', padding: 0, marginBottom: 20 }}>
            <li className="bullet">LoRA rank: 32, alpha: 64</li>
            <li className="bullet">Trainable: 1.4M / 494M (0.29%)</li>
            <li className="bullet">grad_checkpoint: true</li>
            <li className="bullet">Peak memory: 10.6 GB</li>
          </ul>
          <table>
            <thead>
              <tr>
                <th>Iteration</th>
                <th>Val Loss</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>1</td>
                <td>0.739</td>
                <td>baseline</td>
              </tr>
              <tr>
                <td>300</td>
                <td>0.719</td>
                <td>improving</td>
              </tr>
              <tr>
                <td>600</td>
                <td>0.690</td>
                <td>best</td>
              </tr>
              <tr>
                <td>700</td>
                <td>0.690</td>
                <td>best</td>
              </tr>
              <tr>
                <td>900</td>
                <td>0.728</td>
                <td>overfitting</td>
              </tr>
              <tr>
                <td>1000</td>
                <td>0.707</td>
                <td>final</td>
              </tr>
            </tbody>
          </table>
          <p style={{ marginTop: 16, color: '#9090a8', fontSize: '0.95rem' }}>
            1000 iterations over ~3.9 hours; best val loss (0.690) at iter 600
          </p>
        </div>
      </>
    );
  },

  function LoRALessons() {
    return (
      <>
        <h2 className="slide-title">LoRA: Lessons Learned</h2>
        <div className="slide-content row">
          <div className="col">
            <h3 className="slide-subtitle">Gradient Checkpointing</h3>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li className="bullet">First run crashed at iter 270 (OOM at 28 GB)</li>
              <li className="bullet">Fix: <code>grad_checkpoint: true</code></li>
              <li className="bullet">Recomputes activations during backward pass</li>
              <li className="bullet">Result: 10.6 GB peak (~62% memory reduction)</li>
            </ul>
          </div>
          <div className="col">
            <h3 className="slide-subtitle">Overfitting on Small Data</h3>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li className="bullet">4,940 training examples is small for LoRA</li>
              <li className="bullet">Val loss bottomed at iter 600-700 (0.690)</li>
              <li className="bullet">Rose to 0.728 by iter 900</li>
              <li className="bullet">Save checkpoints often, pick the best one</li>
            </ul>
          </div>
        </div>
      </>
    );
  },

  function ByTheNumbers() {
    return (
      <>
        <h2 className="slide-title">By the Numbers</h2>
        <div className="slide-content">
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">42.1M</div>
              <div className="stat-label">Parameters (50M model)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">494M</div>
              <div className="stat-label">Parameters (500M model)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">403K</div>
              <div className="stat-label">Training examples (corpus)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">5,200</div>
              <div className="stat-label">Synthetic examples</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">~1,100</div>
              <div className="stat-label">Tokens/sec (LoRA)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">10.6 GB</div>
              <div className="stat-label">Peak memory (LoRA)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">3.9 hrs</div>
              <div className="stat-label">LoRA training time</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">~250 MB</div>
              <div className="stat-label">Final model (Q4_K_M)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">0.29%</div>
              <div className="stat-label">LoRA trainable params</div>
            </div>
          </div>
        </div>
      </>
    );
  },
];
