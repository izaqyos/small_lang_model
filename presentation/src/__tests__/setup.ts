import '@testing-library/jest-dom/vitest';
import { vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => {
  cleanup();
});

// Global framer-motion mock — renders real DOM elements without animation
vi.mock('framer-motion', () => {
  function createMotionComponent(tag: string) {
    return ({ children, ...props }: Record<string, unknown>) => {
      // Strip framer-motion-specific props
      const {
        initial: _i, animate: _a, exit: _e, transition: _t,
        custom: _c, whileHover: _wh, whileTap: _wt,
        variants: _v, layout: _l,
        ...domProps
      } = props;
      const { createElement } = require('react');
      return createElement(tag, domProps, children as React.ReactNode);
    };
  }

  return {
    motion: new Proxy({}, {
      get: (_target: object, prop: string) => createMotionComponent(prop),
    }),
    AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
    useAnimation: () => ({ start: () => Promise.resolve() }),
    useMotionValue: (initial: number) => ({ get: () => initial, set: () => {} }),
  };
});
