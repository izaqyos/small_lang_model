import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import App from '../App';

describe('App', () => {
  it('renders the slide deck', () => {
    render(<App />);
    expect(screen.getByText('Building a Python Expert SLM')).toBeInTheDocument();
  });

  it('shows slide counter starting at 1', () => {
    const { container } = render(<App />);
    const counter = within(container).getByText(/^1 \//);
    expect(counter).toBeInTheDocument();
    expect(counter).toHaveClass('slide-counter');
  });

  it('shows section label', () => {
    const { container } = render(<App />);
    const label = within(container).getByText('Building Blocks');
    expect(label).toHaveClass('section-label');
  });

  it('renders progress bar', () => {
    const { container } = render(<App />);
    expect(container.querySelector('.progress-bar')).toBeInTheDocument();
  });

  it('navigates forward on ArrowRight', () => {
    const { container } = render(<App />);
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    expect(within(container).getByText(/^2 \//)).toBeInTheDocument();
  });

  it('navigates backward on ArrowLeft', () => {
    const { container } = render(<App />);
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    expect(within(container).getByText(/^1 \//)).toBeInTheDocument();
  });

  it('does not go before slide 1', () => {
    const { container } = render(<App />);
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    expect(within(container).getByText(/^1 \//)).toBeInTheDocument();
  });

  it('navigates with Space key', () => {
    const { container } = render(<App />);
    fireEvent.keyDown(window, { key: ' ' });
    expect(within(container).getByText(/^2 \//)).toBeInTheDocument();
  });

  it('navigates with Enter key', () => {
    const { container } = render(<App />);
    fireEvent.keyDown(window, { key: 'Enter' });
    expect(within(container).getByText(/^2 \//)).toBeInTheDocument();
  });

  it('jumps to first slide on Home', () => {
    const { container } = render(<App />);
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    fireEvent.keyDown(window, { key: 'Home' });
    expect(within(container).getByText(/^1 \//)).toBeInTheDocument();
  });

  it('jumps to last slide on End', () => {
    const { container } = render(<App />);
    fireEvent.keyDown(window, { key: 'End' });
    expect(within(container).getByText('Thank You')).toBeInTheDocument();
  });

  it('navigates back with Backspace', () => {
    const { container } = render(<App />);
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    fireEvent.keyDown(window, { key: 'Backspace' });
    expect(within(container).getByText(/^1 \//)).toBeInTheDocument();
  });
});
