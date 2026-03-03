import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { part1Slides } from '../slides/Part1_BuildingBlocks';
import { part2Slides } from '../slides/Part2_WhatWeDid';
import { part3Slides } from '../slides/Part3_Demos';
import { part4Slides } from '../slides/Part4_DeepDive';

describe('Slide Parts', () => {
  describe('Part 1 — Building Blocks', () => {
    it('has slides', () => {
      expect(part1Slides.length).toBeGreaterThan(0);
    });

    it('first slide is the title slide', () => {
      const TitleSlide = part1Slides[0];
      render(<TitleSlide />);
      expect(screen.getByText('Building a Python Expert SLM')).toBeInTheDocument();
    });

    it('all slides render without crashing', () => {
      for (const Slide of part1Slides) {
        const { unmount } = render(<Slide />);
        unmount();
      }
    });

    it('includes key topic slides', () => {
      const slideTexts: string[] = [];
      for (const Slide of part1Slides) {
        const { container, unmount } = render(<Slide />);
        slideTexts.push(container.textContent || '');
        unmount();
      }
      const allText = slideTexts.join(' ');
      expect(allText).toContain('Language Model');
      expect(allText).toContain('Tokenization');
      expect(allText).toContain('Attention');
      expect(allText).toContain('LoRA');
    });
  });

  describe('Part 2 — What We Did', () => {
    it('has slides', () => {
      expect(part2Slides.length).toBeGreaterThan(0);
    });

    it('all slides render without crashing', () => {
      for (const Slide of part2Slides) {
        const { unmount } = render(<Slide />);
        unmount();
      }
    });
  });

  describe('Part 3 — Demos', () => {
    it('has slides', () => {
      expect(part3Slides.length).toBeGreaterThan(0);
    });

    it('all slides render without crashing', () => {
      for (const Slide of part3Slides) {
        const { unmount } = render(<Slide />);
        unmount();
      }
    });
  });

  describe('Part 4 — Deep Dive', () => {
    it('has slides', () => {
      expect(part4Slides.length).toBeGreaterThan(0);
    });

    it('all slides render without crashing', () => {
      for (const Slide of part4Slides) {
        const { unmount } = render(<Slide />);
        unmount();
      }
    });
  });

  describe('All slides combined', () => {
    const allSlides = [
      ...part1Slides,
      ...part2Slides,
      ...part3Slides,
      ...part4Slides,
    ];

    it('total slide count is reasonable', () => {
      expect(allSlides.length).toBeGreaterThanOrEqual(10);
      expect(allSlides.length).toBeLessThan(100);
    });

    it('every slide is a valid React component', () => {
      for (const Slide of allSlides) {
        expect(typeof Slide).toBe('function');
      }
    });
  });
});
