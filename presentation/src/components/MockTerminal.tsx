import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { colors, fonts } from '../theme';

interface MockTerminalProps {
  prompt: string;
  outputs: { label: string; text: string }[];
  title?: string;
  typingSpeed?: number;
}

export function MockTerminal({
  prompt,
  outputs,
  title = 'python-expert',
  typingSpeed = 25,
}: MockTerminalProps) {
  const [selectedOutput, setSelectedOutput] = useState(0);
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [hasStarted, setHasStarted] = useState(false);

  const currentOutput = outputs[selectedOutput].text;

  const startTyping = useCallback(() => {
    setDisplayedText('');
    setIsTyping(true);
    setHasStarted(true);
  }, []);

  useEffect(() => {
    if (!isTyping) return;
    if (displayedText.length >= currentOutput.length) {
      setIsTyping(false);
      return;
    }
    const timeout = setTimeout(() => {
      const nextChar = currentOutput[displayedText.length];
      const chunkSize = nextChar === '\n' ? 1 : Math.random() > 0.7 ? 2 : 1;
      setDisplayedText(
        currentOutput.slice(0, displayedText.length + chunkSize)
      );
    }, typingSpeed + Math.random() * 15);
    return () => clearTimeout(timeout);
  }, [isTyping, displayedText, currentOutput, typingSpeed]);

  useEffect(() => {
    setDisplayedText('');
    setIsTyping(false);
    setHasStarted(false);
  }, [selectedOutput]);

  return (
    <div
      style={{
        background: colors.bgTerminal,
        borderRadius: 12,
        border: `1px solid ${colors.border}`,
        overflow: 'hidden',
        fontFamily: fonts.mono,
        fontSize: '0.82rem',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '10px 16px',
          background: '#161b22',
          borderBottom: `1px solid ${colors.border}`,
        }}
      >
        <div style={{ display: 'flex', gap: 6 }}>
          <div
            style={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              background: '#ff5f57',
            }}
          />
          <div
            style={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              background: '#febc2e',
            }}
          />
          <div
            style={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              background: '#28c840',
            }}
          />
        </div>
        <span style={{ color: colors.textMuted, fontSize: '0.75rem', marginLeft: 8 }}>
          {title}
        </span>
      </div>

      {outputs.length > 1 && (
        <div
          style={{
            display: 'flex',
            gap: 8,
            padding: '8px 16px',
            borderBottom: `1px solid ${colors.border}`,
          }}
        >
          {outputs.map((o, i) => (
            <button
              key={i}
              onClick={() => setSelectedOutput(i)}
              style={{
                background: i === selectedOutput ? colors.accentBlue + '22' : 'transparent',
                border: `1px solid ${i === selectedOutput ? colors.accentBlue : colors.border}`,
                borderRadius: 6,
                padding: '4px 12px',
                color: i === selectedOutput ? colors.accentBlue : colors.textSecondary,
                cursor: 'pointer',
                fontFamily: fonts.mono,
                fontSize: '0.75rem',
              }}
            >
              {o.label}
            </button>
          ))}
        </div>
      )}

      <div style={{ padding: '16px', minHeight: 200 }}>
        <div style={{ color: colors.accentGreen, marginBottom: 4 }}>
          <span style={{ color: colors.accentBlue }}>{'>>> '}</span>
          <span style={{ color: colors.textPrimary }}>{prompt}</span>
        </div>

        {!hasStarted ? (
          <motion.button
            onClick={startTyping}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{
              marginTop: 12,
              background: 'linear-gradient(135deg, #4fc3f7, #b388ff)',
              border: 'none',
              borderRadius: 8,
              padding: '8px 20px',
              color: '#0b0b1a',
              fontWeight: 600,
              cursor: 'pointer',
              fontFamily: fonts.sans,
              fontSize: '0.85rem',
            }}
          >
            Generate
          </motion.button>
        ) : (
          <div style={{ color: colors.textPrimary, whiteSpace: 'pre-wrap', marginTop: 4 }}>
            {displayedText}
            {isTyping && (
              <motion.span
                animate={{ opacity: [1, 0] }}
                transition={{ duration: 0.6, repeat: Infinity }}
                style={{ color: colors.accentBlue }}
              >
                {'█'}
              </motion.span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
