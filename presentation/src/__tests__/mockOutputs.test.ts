import { describe, it, expect } from 'vitest';
import { tokenizerExamples, model50MOutputs, model500MOutputs } from '../data/mockOutputs';

describe('mockOutputs data', () => {
  describe('tokenizerExamples', () => {
    it('has at least 3 examples', () => {
      expect(tokenizerExamples.length).toBeGreaterThanOrEqual(3);
    });

    it('each example has matching token and id counts', () => {
      for (const ex of tokenizerExamples) {
        expect(ex.tokens.length).toBe(ex.ids.length);
      }
    });

    it('each example has non-empty input', () => {
      for (const ex of tokenizerExamples) {
        expect(ex.input.length).toBeGreaterThan(0);
      }
    });

    it('token IDs are positive integers', () => {
      for (const ex of tokenizerExamples) {
        for (const id of ex.ids) {
          expect(id).toBeGreaterThan(0);
          expect(Number.isInteger(id)).toBe(true);
        }
      }
    });
  });

  describe('model50MOutputs', () => {
    it('has fibonacci output', () => {
      expect(model50MOutputs.fibonacci).toBeDefined();
      expect(model50MOutputs.fibonacci.length).toBeGreaterThan(0);
    });

    it('has sort output', () => {
      expect(model50MOutputs.sort).toBeDefined();
      expect(model50MOutputs.sort.length).toBeGreaterThan(0);
    });

    it('fibonacci output contains def keyword', () => {
      expect(model50MOutputs.fibonacci).toContain('def');
    });
  });

  describe('model500MOutputs', () => {
    it('has fibonacci output with docstring', () => {
      expect(model500MOutputs.fibonacci).toContain('"""');
    });

    it('has sort output with type hints', () => {
      expect(model500MOutputs.sort).toContain('list[int]');
    });

    it('500M fibonacci is longer than 50M (more detailed)', () => {
      expect(model500MOutputs.fibonacci.length).toBeGreaterThan(
        model50MOutputs.fibonacci.length
      );
    });
  });
});
