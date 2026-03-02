import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { colors, fonts } from '../theme';

interface ComparisonPaneProps {
  prompt: string;
  leftLabel: string;
  leftOutput: string;
  rightLabel: string;
  rightOutput: string;
  typingSpeed?: number;
}

export function ComparisonPane({
  prompt,
  leftLabel,
  leftOutput,
  rightLabel,
  rightOutput,
  typingSpeed = 20,
}: ComparisonPaneProps) {
  const [started, setStarted] = useState(false);
  const [leftText, setLeftText] = useState('');
  const [rightText, setRightText] = useState('');

  useEffect(() => {
    if (!started) return;
    setLeftText('');
    setRightText('');
  }, [started]);

  useEffect(() => {
    if (!started) return;
    if (leftText.length >= leftOutput.length) return;
    const t = setTimeout(
      () => setLeftText(leftOutput.slice(0, leftText.length + 1)),
      typingSpeed
    );
    return () => clearTimeout(t);
  }, [started, leftText, leftOutput, typingSpeed]);

  useEffect(() => {
    if (!started) return;
    if (rightText.length >= rightOutput.length) return;
    const t = setTimeout(
      () => setRightText(rightOutput.slice(0, rightText.length + 1)),
      typingSpeed
    );
    return () => clearTimeout(t);
  }, [started, rightText, rightOutput, typingSpeed]);

  const terminalStyle = {
    background: colors.bgTerminal,
    borderRadius: 12,
    border: `1px solid ${colors.border}`,
    fontFamily: fonts.mono,
    fontSize: '0.78rem',
    flex: 1,
    overflow: 'hidden' as const,
  };

  const renderTerminal = (label: string, text: string, output: string) => (
    <div style={terminalStyle}>
      <div
        style={{
          padding: '8px 16px',
          background: '#161b22',
          borderBottom: `1px solid ${colors.border}`,
          fontSize: '0.75rem',
          color: colors.accentBlue,
          fontWeight: 600,
        }}
      >
        {label}
      </div>
      <div style={{ padding: 14, minHeight: 180 }}>
        <div>
          <span style={{ color: colors.accentBlue }}>{'>>> '}</span>
          <span style={{ color: colors.textPrimary }}>{prompt}</span>
        </div>
        <div style={{ color: colors.textPrimary, whiteSpace: 'pre-wrap', marginTop: 4 }}>
          {text}
          {started && text.length < output.length && (
            <motion.span
              animate={{ opacity: [1, 0] }}
              transition={{ duration: 0.6, repeat: Infinity }}
              style={{ color: colors.accentBlue }}
            >
              {'█'}
            </motion.span>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, flex: 1 }}>
      {!started && (
        <motion.button
          onClick={() => setStarted(true)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          style={{
            alignSelf: 'center',
            background: 'linear-gradient(135deg, #4fc3f7, #b388ff)',
            border: 'none',
            borderRadius: 8,
            padding: '8px 24px',
            color: '#0b0b1a',
            fontWeight: 600,
            cursor: 'pointer',
            fontFamily: fonts.sans,
            fontSize: '0.9rem',
          }}
        >
          Compare
        </motion.button>
      )}
      <div style={{ display: 'flex', gap: 16, flex: 1 }}>
        {renderTerminal(leftLabel, leftText, leftOutput)}
        {renderTerminal(rightLabel, rightText, rightOutput)}
      </div>
    </div>
  );
}
