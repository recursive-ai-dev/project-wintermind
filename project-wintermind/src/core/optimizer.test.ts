import { test, expect } from 'vitest';
import { Tensor } from './tensor';
import { NAGOptimizer } from './optimizer';

test('NAG Optimizer step', () => {
  const t = new Tensor([1, 2, 3, 4], [2, 2], { requiresGrad: true });
  t.grad = new Float32Array([0.1, 0.2, 0.3, 0.4]);

  const opt = new NAGOptimizer([t], {
    lr: 0.1,
    momentum: 0.9,
    weightDecay: 0.0,
    clipGradNorm: 1.0,
    zeroStage: 0,
    numRanks: 1,
    useHessian: false,
  });

  opt.update();

  // NAG formula: v = mu * v - lr * grad
  // p += -mu * v - lr * grad
  // with v0 = 0
  // v1 = -0.1 * grad
  // p += mu * (-0.1*grad) - 0.1*grad

  expect(t.data[0]).toBeLessThan(1);
});
