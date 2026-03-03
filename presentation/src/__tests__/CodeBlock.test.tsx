import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { CodeBlock } from '../components/CodeBlock';

describe('CodeBlock', () => {
  it('renders code content', () => {
    render(<CodeBlock>{'def hello(): pass'}</CodeBlock>);
    expect(screen.getByText('def hello(): pass')).toBeInTheDocument();
  });

  it('renders title when provided', () => {
    render(<CodeBlock title="example.py">{'print("hi")'}</CodeBlock>);
    expect(screen.getByText('example.py')).toBeInTheDocument();
    expect(screen.getByText('print("hi")')).toBeInTheDocument();
  });

  it('does not render title when omitted', () => {
    const { container } = render(<CodeBlock>{'code'}</CodeBlock>);
    // With no title, the first child of the outer div should be the pre
    const outerDiv = container.firstElementChild!;
    expect(outerDiv.children).toHaveLength(1);
    expect(outerDiv.firstElementChild!.tagName).toBe('PRE');
  });

  it('applies custom fontSize to pre element', () => {
    const { container } = render(
      <CodeBlock fontSize="0.7rem">{'small code'}</CodeBlock>
    );
    const pre = container.querySelector('pre');
    expect(pre).toHaveStyle({ fontSize: '0.7rem' });
  });

  it('renders content inside a pre element', () => {
    const { container } = render(<CodeBlock>{'formatted\n  code'}</CodeBlock>);
    expect(container.querySelector('pre')).toBeInTheDocument();
  });
});
