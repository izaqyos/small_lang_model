import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { MockTerminal } from '../components/MockTerminal';
import { ComparisonPane } from '../components/ComparisonPane';
import { tokenizerExamples, model50MOutputs, model500MOutputs } from '../data/mockOutputs';
import { colors } from '../theme';

const tokenBadgeStyle: React.CSSProperties = {
  background: '#1a1a2e',
  border: '1px solid #2a2a4a',
  borderRadius: 8,
  padding: '4px 10px',
  display: 'inline-block',
  margin: 3,
};

const tokenContainerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.08 },
  },
};

const tokenItemVariants = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0 },
};

export const part3Slides: React.FC[] = [
  function SectionTitle() {
    return (
      <div className="section-title-slide">
        <span className="section-number">03</span>
        <h2>Live Demos</h2>
        <p>See the models in action</p>
      </div>
    );
  },

  function TokenizerDemo() {
    const [selectedIndex, setSelectedIndex] = useState(0);
    const example = tokenizerExamples[selectedIndex];

    return (
      <>
        <h2 className="slide-title">Demo: Tokenizer in Action</h2>
        <div className="slide-content">
          <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
            {tokenizerExamples.map((ex, i) => (
              <button
                key={i}
                onClick={() => setSelectedIndex(i)}
                style={{
                  background: i === selectedIndex ? colors.accentBlue + '22' : 'transparent',
                  border: `1px solid ${i === selectedIndex ? colors.accentBlue : colors.border}`,
                  borderRadius: 8,
                  padding: '8px 16px',
                  color: i === selectedIndex ? colors.accentBlue : colors.textSecondary,
                  cursor: 'pointer',
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: '0.9rem',
                }}
              >
                {ex.input}
              </button>
            ))}
          </div>
          <div style={{ marginBottom: 12, fontSize: '1rem', color: colors.textPrimary }}>
            Input: <code style={{ background: '#1a1a2e', padding: '2px 8px', borderRadius: 4 }}>{example.input}</code>
          </div>
          <motion.div
            key={selectedIndex}
            variants={tokenContainerVariants}
            initial="hidden"
            animate="show"
            style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 0', alignItems: 'flex-end' }}
          >
            {example.tokens.map((token, i) => (
              <motion.div
                key={i}
                variants={tokenItemVariants}
                style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}
              >
                <span style={tokenBadgeStyle}>{token}</span>
                <span style={{ fontSize: '0.7rem', color: colors.textMuted }}>{example.ids[i]}</span>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </>
    );
  },

  function ModelGenerationDemo() {
    return (
      <>
        <h2 className="slide-title">Demo: Model Generation</h2>
        <p className="slide-subtitle" style={{ marginBottom: 12 }}>
          Click &quot;Generate&quot; to see token-by-token output
        </p>
        <div className="slide-content">
          <MockTerminal
            prompt="def fibonacci(n):"
            outputs={[
              { label: '50M Model', text: model50MOutputs.fibonacci },
              { label: '500M Model', text: model500MOutputs.fibonacci },
            ]}
          />
        </div>
      </>
    );
  },

  function SideBySideComparison() {
    return (
      <>
        <h2 className="slide-title">Side-by-Side: 50M vs 500M</h2>
        <div className="slide-content">
          <ComparisonPane
            prompt="def sort_list(arr):"
            leftLabel="50M Model (from scratch)"
            leftOutput={model50MOutputs.sort}
            rightLabel="500M Model (LoRA fine-tuned)"
            rightOutput={model500MOutputs.sort}
          />
          <ul style={{ listStyle: 'none', padding: 0, marginTop: 16 }}>
            <li className="bullet">
              Notice: type hints, docstrings, O(n log n) algorithm vs O(n²)
            </li>
          </ul>
        </div>
      </>
    );
  },
];
