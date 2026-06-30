import { test, expect } from 'vitest';
import { Tensor } from './tensor';

test('Tensor creation and shapes', () => {
  const t = new Tensor([1, 2, 3, 4], [2, 2]);
  expect(t.shape).toEqual([2, 2]);
  expect(t.size).toBe(4);
});

test('Tensor sum and backward', () => {
  const t = new Tensor([1, 2, 3, 4], [2, 2], { requiresGrad: true });
  const s = t.sum();
  s.backward();
  expect(s.data[0]).toBe(10);
  expect(t.grad).toBeDefined();
  expect(t.grad![0]).toBe(1);
  expect(t.grad![3]).toBe(1);
});

test('Tensor mul and backward', () => {
  const t1 = new Tensor([1, 2, 3, 4], [2, 2], { requiresGrad: true });
  const t2 = new Tensor([2, 2, 2, 2], [2, 2], { requiresGrad: true });
  const out = t1.mul(t2);
  const s = out.sum();
  s.backward();
  expect(out.data[0]).toBe(2);
  expect(out.data[3]).toBe(8);
  expect(t1.grad![0]).toBe(2); // grad is t2
  expect(t2.grad![0]).toBe(1); // grad is t1
});

test('Tensor activation functions (relu, tanh, sigmoid)', () => {
  const t = new Tensor([-1, 0, 1], [3], { requiresGrad: true });

  // ReLU
  const r = t.relu();
  expect(r.data[0]).toBe(0);
  expect(r.data[1]).toBe(0);
  expect(r.data[2]).toBe(1);
  r.sum().backward();
  expect(t.grad![0]).toBe(0);
  expect(t.grad![1]).toBe(0);
  expect(t.grad![2]).toBe(1);
  t.zeroGrad();

  // Tanh
  const th = t.tanh();
  expect(th.data[0]).toBeCloseTo(Math.tanh(-1));
  expect(th.data[1]).toBe(0);
  expect(th.data[2]).toBeCloseTo(Math.tanh(1));
  th.sum().backward();
  expect(t.grad![1]).toBe(1);
  t.zeroGrad();

  // Sigmoid
  const s = t.sigmoid();
  expect(s.data[0]).toBeCloseTo(1 / (1 + Math.exp(1)));
  expect(s.data[1]).toBe(0.5);
  expect(s.data[2]).toBeCloseTo(1 / (1 + Math.exp(-1)));
  s.sum().backward();
  expect(t.grad![1]).toBe(0.25);
  t.zeroGrad();
});

test('Tensor softmax numerically stable', () => {
  // Try large values which would normally overflow Math.exp
  const t = new Tensor([1000, 1000, 1000], [3], { requiresGrad: true });
  const s = t.softmax();
  expect(s.data[0]).toBeCloseTo(1/3);
  expect(s.data[1]).toBeCloseTo(1/3);
  expect(s.data[2]).toBeCloseTo(1/3);

  // Backward for softmax
  const out = s.sum();
  out.backward();
  // Derivative of sum of softmax is 0 because sum(softmax) = 1 (constant)
  // Actually, d(sum(s))/dx_i = d(1)/dx_i = 0
  // Let's verify sum is 1, gradient should be ~0
  expect(t.grad![0]).toBeCloseTo(0);
  expect(t.grad![1]).toBeCloseTo(0);
});

test('Tensor dot product / matmul', () => {
  // [2, 3] x [3, 2] = [2, 2]
  const a = new Tensor([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
  const b = new Tensor([7, 8, 9, 10, 11, 12], [3, 2], { requiresGrad: true });

  const c = a.matmul(b);
  expect(c.shape).toEqual([2, 2]);

  // c[0, 0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
  expect(c.data[0]).toBe(58);
  // c[0, 1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
  expect(c.data[1]).toBe(64);
  // c[1, 0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
  expect(c.data[2]).toBe(139);
  // c[1, 1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
  expect(c.data[3]).toBe(154);

  c.sum().backward();

  // Check gradients
  // dl/da_00 = b_00 + b_01 = 7 + 8 = 15
  expect(a.grad![0]).toBe(15);
  // dl/db_00 = a_00 + a_10 = 1 + 4 = 5
  expect(b.grad![0]).toBe(5);
});
