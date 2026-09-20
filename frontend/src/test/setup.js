import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});

// --- jsdom polyfills required by Recharts ---

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

if (typeof window !== 'undefined' && !window.ResizeObserver) {
  window.ResizeObserver = ResizeObserverStub;
}

if (
  typeof window !== 'undefined' &&
  typeof window.SVGElement !== 'undefined' &&
  !window.SVGElement.prototype.getBBox
) {
  window.SVGElement.prototype.getBBox = () => ({
    x: 0,
    y: 0,
    width: 0,
    height: 0,
  });
}
