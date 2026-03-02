import { useState, useEffect, useCallback, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { part1Slides } from './slides/Part1_BuildingBlocks';
import { part2Slides } from './slides/Part2_WhatWeDid';
import { part3Slides } from './slides/Part3_Demos';
import { part4Slides } from './slides/Part4_DeepDive';

const closingSlide: React.FC = function ThankYou() {
  return (
    <div className="title-slide">
      <h1>Thank You</h1>
      <h2>Questions?</h2>
      <div className="tech-badges" style={{ marginTop: 8 }}>
        <span className="badge">github.com/yosii</span>
      </div>
    </div>
  );
};

const sections = [
  { name: 'Building Blocks', slides: part1Slides },
  { name: 'What We Did', slides: part2Slides },
  { name: 'Demos', slides: part3Slides },
  { name: 'Deep Dive', slides: part4Slides },
  { name: '', slides: [closingSlide] },
];

const allSlides = sections.flatMap((s) => s.slides);

function getSectionName(index: number): string {
  let count = 0;
  for (const section of sections) {
    count += section.slides.length;
    if (index < count) return section.name;
  }
  return '';
}

export default function App() {
  const [current, setCurrent] = useState(0);
  const [direction, setDirection] = useState(0);
  const total = allSlides.length;
  const touchStartX = useRef(0);

  const go = useCallback(
    (delta: number) => {
      setCurrent((prev) => {
        const next = prev + delta;
        if (next < 0 || next >= total) return prev;
        setDirection(delta);
        return next;
      });
    },
    [total]
  );

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        go(1);
      }
      if (e.key === 'ArrowLeft' || e.key === 'Backspace') {
        e.preventDefault();
        go(-1);
      }
      if (e.key === 'Home') {
        e.preventDefault();
        setDirection(-1);
        setCurrent(0);
      }
      if (e.key === 'End') {
        e.preventDefault();
        setDirection(1);
        setCurrent(total - 1);
      }
    };

    const handleTouchStart = (e: TouchEvent) => {
      touchStartX.current = e.touches[0].clientX;
    };
    const handleTouchEnd = (e: TouchEvent) => {
      const dx = e.changedTouches[0].clientX - touchStartX.current;
      if (Math.abs(dx) > 50) go(dx < 0 ? 1 : -1);
    };

    window.addEventListener('keydown', handleKey);
    window.addEventListener('touchstart', handleTouchStart);
    window.addEventListener('touchend', handleTouchEnd);
    return () => {
      window.removeEventListener('keydown', handleKey);
      window.removeEventListener('touchstart', handleTouchStart);
      window.removeEventListener('touchend', handleTouchEnd);
    };
  }, [go, total]);

  const CurrentSlide = allSlides[current];

  return (
    <div className="deck">
      <AnimatePresence mode="wait" custom={direction}>
        <motion.div
          key={current}
          className="slide"
          custom={direction}
          initial={{ opacity: 0, x: direction >= 0 ? 60 : -60 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: direction >= 0 ? -60 : 60 }}
          transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
        >
          <CurrentSlide />
        </motion.div>
      </AnimatePresence>

      <div
        className="progress-bar"
        style={{ width: `${((current + 1) / total) * 100}%` }}
      />
      <div className="slide-counter">
        {current + 1} / {total}
      </div>
      <div className="section-label">{getSectionName(current)}</div>
    </div>
  );
}
