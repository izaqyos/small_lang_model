import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { MockTerminal } from '../components/MockTerminal';

describe('MockTerminal', () => {
  const defaultProps = {
    prompt: 'generate fibonacci',
    outputs: [{ label: 'test', text: 'def fib(n): pass' }],
  };

  it('renders prompt text', () => {
    render(<MockTerminal {...defaultProps} />);
    expect(screen.getByText('generate fibonacci')).toBeInTheDocument();
  });

  it('renders default title', () => {
    render(<MockTerminal {...defaultProps} />);
    expect(screen.getByText('python-expert')).toBeInTheDocument();
  });

  it('renders custom title', () => {
    render(<MockTerminal {...defaultProps} title="my-model" />);
    expect(screen.getByText('my-model')).toBeInTheDocument();
  });

  it('shows Generate button before start', () => {
    render(<MockTerminal {...defaultProps} />);
    expect(screen.getByRole('button', { name: 'Generate' })).toBeInTheDocument();
  });

  it('hides Generate button after clicking it', () => {
    render(<MockTerminal {...defaultProps} />);
    fireEvent.click(screen.getByRole('button', { name: 'Generate' }));
    expect(screen.queryByRole('button', { name: 'Generate' })).not.toBeInTheDocument();
  });

  it('starts typing after Generate is clicked', async () => {
    vi.useFakeTimers();
    render(<MockTerminal {...defaultProps} typingSpeed={10} />);
    fireEvent.click(screen.getByRole('button', { name: 'Generate' }));

    await act(async () => {
      vi.advanceTimersByTime(500);
    });

    const pane = document.querySelector('[style*="white-space"]');
    expect(pane).toBeInTheDocument();
    expect(pane!.textContent!.length).toBeGreaterThan(0);

    vi.useRealTimers();
  });

  it('renders output selector buttons when multiple outputs', () => {
    const props = {
      prompt: 'test',
      outputs: [
        { label: 'Option A', text: 'output a' },
        { label: 'Option B', text: 'output b' },
      ],
    };
    render(<MockTerminal {...props} />);
    expect(screen.getByRole('button', { name: 'Option A' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Option B' })).toBeInTheDocument();
  });

  it('renders terminal chrome (3 dots)', () => {
    const { container } = render(<MockTerminal {...defaultProps} />);
    const dots = container.querySelectorAll('[style*="border-radius: 50%"]');
    expect(dots.length).toBe(3);
  });
});
