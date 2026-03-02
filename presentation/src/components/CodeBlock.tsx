interface CodeBlockProps {
  children: string;
  title?: string;
  fontSize?: string;
}

export function CodeBlock({ children, title, fontSize }: CodeBlockProps) {
  return (
    <div>
      {title && (
        <div
          style={{
            fontSize: '0.75rem',
            color: '#9090a8',
            marginBottom: 6,
            fontFamily: "'JetBrains Mono', monospace",
            letterSpacing: '0.05em',
          }}
        >
          {title}
        </div>
      )}
      <pre className="code-block" style={fontSize ? { fontSize } : undefined}>
        {children}
      </pre>
    </div>
  );
}
