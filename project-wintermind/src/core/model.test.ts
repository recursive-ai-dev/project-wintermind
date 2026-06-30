import { test, expect } from 'vitest';
import { ModelConfig, DEFAULT_CONFIG, WintermindModel, DataPipeline } from './model';
import { buildBootstrapVocab } from './bpe';

test('DataPipeline nextBatch', () => {
  const vocab = buildBootstrapVocab();
  const pipeline = new DataPipeline('the quick brown fox', vocab, 4);
  const batch = pipeline.nextBatch();
  expect(batch.inputIds.length).toBe(4);
  expect(batch.targetIds.length).toBe(4);
});

test('WintermindModel step', () => {
  const vocab = buildBootstrapVocab();
  const C: ModelConfig = {
    ...DEFAULT_CONFIG,
    vocabSize: vocab.tokenToId.size,
    embedDim: 16,
    hiddenDim: 16,
    ffDim: 32,
    numTransformerBlocks: 1,
  };

  const model = new WintermindModel(C);

  const inputIds = [0, 1, 2, 3];
  const targetIds = [1, 2, 3, 4];

  const step = model.trainStep(inputIds, targetIds, vocab);

  expect(step.loss).toBeGreaterThan(0);
  expect(step.gradNorm).toBeGreaterThan(0);
});

test('WintermindModel generate', () => {
  const vocab = buildBootstrapVocab();
  const C: ModelConfig = {
    ...DEFAULT_CONFIG,
    vocabSize: vocab.tokenToId.size,
    embedDim: 16,
    hiddenDim: 16,
    ffDim: 32,
    numTransformerBlocks: 1,
  };

  const model = new WintermindModel(C);

  const promptIds = [0, 1];
  const { ids, text } = model.generate(promptIds, vocab, 5);

  expect(ids.length).toBeGreaterThan(2);
  expect(text.length).toBeGreaterThan(0);
});
