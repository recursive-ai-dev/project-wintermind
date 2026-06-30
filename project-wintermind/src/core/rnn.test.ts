import { test, expect } from 'vitest';
import { Tensor, randn, zeros } from './tensor';
import { EmbeddingLayer, LSTMCell, TransformerBlock, ProjectionHead, crossEntropyLoss } from './rnn';

test('EmbeddingLayer forward', () => {
  const emb = new EmbeddingLayer(10, 4);
  const out = emb.forward([1, 2]);
  expect(out.shape).toEqual([2, 4]); // 2 tokens * 4 dim
});

test('LSTMCell forward', () => {
  const lstm = new LSTMCell(4, 4);
  const x = randn([4]);
  const h = zeros([4]);
  const c = zeros([4]);
  const out = lstm.forward(x, h, c);
  expect(out.h.shape).toEqual([4]);
  expect(out.c.shape).toEqual([4]);
});

test('TransformerBlock forward', () => {
  const tb = new TransformerBlock(4, 8);
  const x = randn([2, 4]); // 2 tokens * 4 dim
  const out = tb.forward(x);
  expect(out.shape).toEqual([2, 4]);
});

test('ProjectionHead forward', () => {
  const ph = new ProjectionHead(4, 10);
  const x = randn([4]);
  const out = ph.forward(x);
  expect(out.shape).toEqual([10]);
});

test('CrossEntropyLoss', () => {
  const logits = new Tensor([1, 2, 3, 4], [4]);
  const targetId = 2;
  const { loss } = crossEntropyLoss(logits, targetId);
  expect(loss.shape).toEqual([1]);
});
