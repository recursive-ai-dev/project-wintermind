import { test, expect } from 'vitest';
import { SEPModule } from './sep';
import { buildBootstrapVocab } from './bpe';

test('SEPModule analyze', () => {
  const sep = new SEPModule({ microBatchSize: 2 });
  const logits = [
    new Float32Array([0.1, 0.9, 0.0, 0.0]),
    new Float32Array([0.2, 0.8, 0.0, 0.0])
  ];

  const vocab = buildBootstrapVocab();
  const tokens = ['winter', 'wind'];

  const result = sep.analyze(tokens, logits, vocab.idToToken);

  expect(result).toBeDefined();
  expect(result.delta).toBeDefined();
  expect(result.fullPrediction.tokenStr).toBeDefined();
  expect(result.attributions.length).toBeGreaterThan(0);
  expect(result.calibratedConfidence).toBeGreaterThanOrEqual(0);
  expect(result.calibratedConfidence).toBeLessThanOrEqual(1);
});
