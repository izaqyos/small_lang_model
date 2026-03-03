import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { ComparisonPane } from '../components/ComparisonPane';

describe('ComparisonPane', () => {
  const defaultProps = {
    prompt: 'write fibonacci',
    leftLabel: '50M Model',
    leftOutput: 'def fib(): pass',
    rightLabel: '500M Model',
    rightOutput: 'def fibonacci(n): ...',
  };

  it('renders both terminal labels', () => {
    render(<ComparisonPane {...defaultProps} />);
    expect(screen.getByText('50M Model')).toBeInTheDocument();
    expect(screen.getByText('500M Model')).toBeInTheDocument();
  });

  it('renders the prompt in both terminals', () => {
    render(<ComparisonPane {...defaultProps} />);
    const prompts = screen.getAllByText('write fibonacci');
    expect(prompts).toHaveLength(2);
  });

  it('shows Compare button initially', () => {
    render(<ComparisonPane {...defaultProps} />);
    expect(screen.getByRole('button', { name: 'Compare' })).toBeInTheDocument();
  });

  it('hides Compare button after clicking', () => {
    render(<ComparisonPane {...defaultProps} />);
    fireEvent.click(screen.getByRole('button', { name: 'Compare' }));
    expect(screen.queryByRole('button', { name: 'Compare' })).not.toBeInTheDocument();
  });

  it('starts typing in both panes after Compare', async () => {
    vi.useFakeTimers();
    render(<ComparisonPane {...defaultProps} typingSpeed={5} />);
    fireEvent.click(screen.getByRole('button', { name: 'Compare' }));

    await act(async () => {
      vi.advanceTimersByTime(300);
    });

    const panes = document.querySelectorAll('[style*="white-space"]');
    expect(panes.length).toBe(2);

    vi.useRealTimers();
  });
});
