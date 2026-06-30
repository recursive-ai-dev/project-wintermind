import { test, expect } from 'vitest';
import { buildBootstrapVocab, encode, decode } from './bpe';

test('BPE encoding and decoding', () => {
  const vocab = buildBootstrapVocab();
  const encoded = encode('hello world', vocab);
  const decoded = decode(encoded, vocab);
  expect(decoded).toBe('hello world');
});
