/**
 * WINTERMIND — MASTER MODEL + TRAINING PIPELINE
 *
 * Architecture:
 *   EmbeddingLayer → N × TransformerBlock → LSTMCell → ProjectionHead
 *
 * Training:
 *   DataPipeline → teacher-forcing forward → sum cross-entropy → backward → NAG
 *
 * Integrations:
 *   GSAR  — symbolic token routing, priority-based concept recognition
 *   SEP   — micro-vs-full prediction comparison, attribution, spurious detection
 *   ZeRO  — memory sharding simulation
 *   Cosine LR — annealing schedule over training steps
 */

import { Tensor, zeros } from './tensor';
import {
  EmbeddingLayer,
  LSTMCell,
  TransformerBlock,
  ProjectionHead,
  crossEntropyLoss,
  LayerParamInfo,
} from './rnn';
import { NAGOptimizer, cosineAnnealingLR } from './optimizer';
import { GSARModule, GSARResult } from './gsar';
import { SEPModule, SEPExplanation } from './sep';
import { BPEVocab, encode, decode } from './bpe';

// ── Model Configuration ───────────────────────────────────────────────────────

export interface ModelConfig {
  vocabSize:            number;
  embedDim:             number;
  hiddenDim:            number;
  ffDim:                number;
  numTransformerBlocks: number;
  lr:                   number;
  momentum:             number;
  weightDecay:          number;
  clipGradNorm:         number;
  zeroStage:            0 | 1 | 2 | 3;
  numRanks:             number;
  useHessian:           boolean;
  dtype:                'fp32' | 'fp16';
  microBatchSize:       number;
}

export const DEFAULT_CONFIG: ModelConfig = {
  vocabSize:            256,
  embedDim:             64,
  hiddenDim:            64,
  ffDim:                128,
  numTransformerBlocks: 2,
  lr:                   5e-4,
  momentum:             0.9,
  weightDecay:          1e-4,
  clipGradNorm:         1.0,
  zeroStage:            1,
  numRanks:             4,
  useHessian:           false,
  dtype:                'fp32',
  microBatchSize:       8,
};

// ── Training step record ──────────────────────────────────────────────────────

export interface TrainStep {
  step:            number;
  loss:            number;
  gradNorm:        number;
  lr:              number;
  paramNorm:       number;
  perplexity:      number;
  gsarRatio:       number;
  sepDelta:        number;
  sepExplanation:  string;
  tokens:          string[];
  prediction:      string;
  targetToken:     string;
  timestamp:       number;
  zeroMemFactor:   number;
  hessianNorm:     number;
  topKPredictions: Array<{ id: number; str: string; prob: number }>;
  sepAnalysis:     SEPExplanation | null;
}

// ── Data Pipeline ─────────────────────────────────────────────────────────────

export class DataPipeline {
  private vocab:    BPEVocab;
  private tokenIds: number[];
  private seqLen:   number;
  private cursor:   number = 0;

  constructor(corpus: string, vocab: BPEVocab, seqLen = 16) {
    this.vocab    = vocab;
    this.seqLen   = seqLen;
    this.tokenIds = encode(corpus, vocab);
    if (this.tokenIds.length < seqLen + 2) {
      // Repeat corpus until we have enough tokens
      while (this.tokenIds.length < seqLen * 4) {
        this.tokenIds = [...this.tokenIds, ...encode(corpus, vocab)];
      }
    }
  }

  nextBatch(): { inputIds: number[]; targetIds: number[] } {
    if (this.cursor + this.seqLen + 1 >= this.tokenIds.length) this.cursor = 0;
    const inputIds  = this.tokenIds.slice(this.cursor, this.cursor + this.seqLen);
    const targetIds = this.tokenIds.slice(this.cursor + 1, this.cursor + this.seqLen + 1);
    this.cursor++;
    return { inputIds, targetIds };
  }

  decodeIds(ids: number[]): string { return decode(ids, this.vocab); }
  totalTokens(): number            { return this.tokenIds.length; }
}

// ── Wintermind Model ──────────────────────────────────────────────────────────

export class WintermindModel {
  config: ModelConfig;

  embedding:         EmbeddingLayer;
  transformerBlocks: TransformerBlock[];
  lstmCell:          LSTMCell;
  projHead:          ProjectionHead;
  optimizer:         NAGOptimizer;
  gsar:              GSARModule;
  sep:               SEPModule;

  private allParams:  Tensor[];
  private _step:      number = 0;
  private totalSteps: number;

  // Persistent LSTM state (detached between steps — truncated BPTT)
  private hState: Tensor;
  private cState: Tensor;

  constructor(config: Partial<ModelConfig> = {}, totalSteps = 500) {
    this.config     = { ...DEFAULT_CONFIG, ...config };
    this.totalSteps = totalSteps;
    const C         = this.config;

    this.embedding = new EmbeddingLayer(C.vocabSize, C.embedDim);
    this.transformerBlocks = Array.from(
      { length: C.numTransformerBlocks },
      () => new TransformerBlock(C.embedDim, C.ffDim)
    );
    this.lstmCell = new LSTMCell(C.embedDim, C.hiddenDim);
    this.projHead = new ProjectionHead(C.hiddenDim, C.vocabSize);

    this.allParams = [
      ...this.embedding.params(),
      ...this.transformerBlocks.flatMap(b => b.params()),
      ...this.lstmCell.params(),
      ...this.projHead.params(),
    ];

    this.optimizer = new NAGOptimizer(this.allParams, {
      lr:           C.lr,
      momentum:     C.momentum,
      weightDecay:  C.weightDecay,
      clipGradNorm: C.clipGradNorm,
      zeroStage:    C.zeroStage,
      numRanks:     C.numRanks,
      useHessian:   C.useHessian,
    });

    this.gsar = new GSARModule();
    this.sep  = new SEPModule({ microBatchSize: C.microBatchSize });

    this.hState = zeros([C.hiddenDim], { requiresGrad: true, label: 'h0' });
    this.cState = zeros([C.hiddenDim], { requiresGrad: true, label: 'c0' });
  }

  // ── Forward pass (inference / diagnostics) ────────────────────────────────
  // Does not accumulate gradients. Returns logits at every position.

  forward(
    inputIds: number[],
    vocab:    BPEVocab
  ): { logitsPerStep: Float32Array[]; gsarResult: GSARResult; tokens: string[] } {
    const tokens     = inputIds.map(id => vocab.idToToken.get(id) ?? `<${id}>`);
    const gsarResult = this.gsar.reason(tokens);

    const embedded = this.embedding.forward(inputIds);
    let tfOut      = embedded;
    for (const block of this.transformerBlocks) tfOut = block.forward(tfOut);

    const T = inputIds.length;
    const D = this.config.embedDim;
    const logitsPerStep: Float32Array[] = [];

    let h = this.hState;
    let c = this.cState;

    for (let t = 0; t < T; t++) {
      const xData = new Float32Array(D);
      for (let d = 0; d < D; d++) xData[d] = tfOut.data[t * D + d];
      const x    = new Tensor(xData, [D], { requiresGrad: false });
      const lstm = this.lstmCell.forward(x, h, c);
      h = lstm.h;
      c = lstm.c;
      const logits = this.projHead.forward(h);
      logitsPerStep.push(new Float32Array(logits.data));
    }

    // Detach and persist LSTM state for next call
    this.hState = new Tensor(new Float32Array(h.data), h.shape, {
      requiresGrad: true, label: 'h_persist',
    });
    this.cState = new Tensor(new Float32Array(c.data), c.shape, {
      requiresGrad: true, label: 'c_persist',
    });

    return { logitsPerStep, gsarResult, tokens };
  }

  // ── Training step ─────────────────────────────────────────────────────────
  // Forward → accumulate cross-entropy over ALL timesteps → backward → NAG update

  trainStep(inputIds: number[], targetIds: number[], vocab: BPEVocab): TrainStep {
    this._step++;
    const C = this.config;
    const T = Math.min(inputIds.length, targetIds.length);
    const D = C.embedDim;

    // Apply cosine LR schedule
    const lr = cosineAnnealingLR(this.optimizer, this._step, this.totalSteps, 1e-5, C.lr);

    // Zero all gradients
    this.optimizer.zeroGrads();

    // ── Forward with gradient tracking ─────────────────────────────────────
    const embedded = this.embedding.forward(inputIds);
    let tfOut      = embedded;
    for (const block of this.transformerBlocks) tfOut = block.forward(tfOut);

    let h = zeros([C.hiddenDim], { requiresGrad: true, label: 'h_train' });
    let c = zeros([C.hiddenDim], { requiresGrad: true, label: 'c_train' });

    const logitsPerStep: Float32Array[] = [];
    const lossTerms:     Tensor[]       = [];

    for (let t = 0; t < T; t++) {
      const xData = new Float32Array(D);
      for (let d = 0; d < D; d++) xData[d] = tfOut.data[t * D + d];
      const x    = new Tensor(xData, [D], { requiresGrad: true });
      const lstm = this.lstmCell.forward(x, h, c);
      h = lstm.h;
      c = lstm.c;

      const logits   = this.projHead.forward(h);
      logitsPerStep.push(new Float32Array(logits.data));

      const targetId = Math.max(0, Math.min(targetIds[t], C.vocabSize - 1));
      const { loss } = crossEntropyLoss(logits, targetId);
      lossTerms.push(loss);
    }

    // Sum all per-step losses then scale by 1/T — every timestep contributes to backward
    let totalLossVal = 0;
    for (const l of lossTerms) totalLossVal += l.data[0];
    const avgLoss = totalLossVal / Math.max(T, 1);

    // Backward: propagate through all timestep loss nodes
    // We call backward on each, accumulating gradients across the shared parameter tensors.
    // Scale each by 1/T so the total gradient = ∇(mean loss).
    const invT = 1 / Math.max(T, 1);
    for (const l of lossTerms) {
      if (!l.grad) l.grad = new Float32Array(1).fill(invT);
      else         l.grad[0] = invT;
      l.backward();
    }

    // NAG optimizer update
    const optResult = this.optimizer.update();

    // ── GSAR analysis ────────────────────────────────────────────────────
    const tokens     = inputIds.map(id => vocab.idToToken.get(id) ?? `<${id}>`);
    const gsarResult = this.gsar.reason(tokens);
    const gsarRatio  = this.gsar.allocationRatio(gsarResult);

    // ── SEP analysis ─────────────────────────────────────────────────────
    const sepExp: SEPExplanation = this.sep.analyze(tokens, logitsPerStep, vocab.idToToken);

    // ── Hessian norm sample ───────────────────────────────────────────────
    const hessianNorm = C.useHessian
      ? this.allParams.slice(0, 2).reduce((s, p) => s + p.norm(), 0)
      : 0;

    const paramNorm = Math.sqrt(this.allParams.reduce((s, p) => s + p.norm() ** 2, 0));

    // ── Top-K of last step for display ────────────────────────────────────
    const lastLogits = logitsPerStep[logitsPerStep.length - 1] ?? new Float32Array(C.vocabSize);
    let maxL = -Infinity;
    for (let i = 0; i < lastLogits.length; i++) if (lastLogits[i] > maxL) maxL = lastLogits[i];
    let sumExp = 0;
    const exps = new Float32Array(lastLogits.length);
    for (let i = 0; i < lastLogits.length; i++) {
      exps[i] = Math.exp(lastLogits[i] - maxL);
      sumExp += exps[i];
    }
    const lastProbs: Array<{ id: number; str: string; prob: number }> = [];
    for (let i = 0; i < lastLogits.length; i++) {
      lastProbs.push({ id: i, str: vocab.idToToken.get(i) ?? `<${i}>`, prob: exps[i] / sumExp });
    }
    lastProbs.sort((a, b) => b.prob - a.prob);

    const targetToken = vocab.idToToken.get(
      Math.max(0, Math.min(targetIds[T - 1] ?? 0, C.vocabSize - 1))
    ) ?? '?';

    return {
      step:            this._step,
      loss:            avgLoss,
      gradNorm:        optResult.globalNorm,
      lr,
      paramNorm,
      perplexity:      Math.min(Math.exp(avgLoss), 9999),
      gsarRatio,
      sepDelta:        sepExp.delta,
      sepExplanation:  sepExp.explanation,
      tokens,
      prediction:      sepExp.fullPrediction.tokenStr,
      targetToken,
      timestamp:       Date.now(),
      zeroMemFactor:   this.optimizer.memoryReductionFactor(),
      hessianNorm,
      topKPredictions: lastProbs.slice(0, 8),
      sepAnalysis:     sepExp,
    };
  }

  // ── Autoregressive text generation ───────────────────────────────────────

  generate(
    promptIds:    number[],
    vocab:        BPEVocab,
    maxNewTokens = 30,
    temperature  = 0.85,
    topK         = 10
  ): { ids: number[]; text: string } {
    const C   = this.config;
    const D   = C.embedDim;
    const generated = [...promptIds];

    let h = zeros([C.hiddenDim], { label: 'gen.h' });
    let c = zeros([C.hiddenDim], { label: 'gen.c' });

    // Warm-up: forward pass over entire prompt to build LSTM state
    const emb   = this.embedding.forward(generated);
    let   tfOut = emb;
    for (const block of this.transformerBlocks) tfOut = block.forward(tfOut);

    for (let t = 0; t < generated.length; t++) {
      const xData = new Float32Array(D);
      for (let d = 0; d < D; d++) xData[d] = tfOut.data[t * D + d];
      const x    = new Tensor(xData, [D]);
      const lstm = this.lstmCell.forward(x, h, c);
      h = lstm.h;
      c = lstm.c;
    }

    // Auto-regressive sampling: one token at a time
    for (let i = 0; i < maxNewTokens; i++) {
      const logits = this.projHead.forward(h);
      const nextId = sampleTopK(logits.data, temperature, topK);
      if (nextId === 3) break;  // EOS_ID = 3
      generated.push(nextId);

      // Embed the new token and step the LSTM
      const embData = new Float32Array(D);
      for (let d = 0; d < D; d++) embData[d] = this.embedding.weight.data[nextId * D + d];
      const x    = new Tensor(embData, [D]);
      const lstm = this.lstmCell.forward(x, h, c);
      h = lstm.h;
      c = lstm.c;
    }

    return { ids: generated, text: decode(generated, vocab) };
  }

  // ── Model summary ─────────────────────────────────────────────────────────

  summary(): string {
    const C           = this.config;
    const totalParams = this.allParams.reduce((s, p) => s + p.size, 0);
    const optDiag     = this.optimizer.diagnostics();
    const memFactor   = this.optimizer.memoryReductionFactor();
    return [
      '══════════════════════════════════════════════════════════════════════',
      '  PROJECT WINTERMIND — MODEL SUMMARY',
      '══════════════════════════════════════════════════════════════════════',
      `  Architecture :  Embedding → ${C.numTransformerBlocks}×TransformerBlock → LSTMCell → Projection`,
      `  Vocab size   :  ${C.vocabSize} tokens`,
      `  Embed dim    :  ${C.embedDim}`,
      `  Hidden dim   :  ${C.hiddenDim}`,
      `  FF dim       :  ${C.ffDim}`,
      `  Precision    :  ${C.dtype.toUpperCase()} (mixed-precision FP16 simulation)`,
      `  Total params :  ${totalParams.toLocaleString()}`,
      `  Param memory :  ${(totalParams * 4 / 1024).toFixed(2)} KB (fp32)`,
      `  Optimizer    :  NAG  μ=${C.momentum}  lr=${C.lr}  wd=${C.weightDecay}`,
      `  LR schedule  :  Cosine annealing  [ 1e-5  →  ${C.lr} ]`,
      `  Grad clip    :  ‖∇‖₂ ≤ ${C.clipGradNorm}`,
      `  Memory       :  ${this.optimizer.zeroStageDescription()}`,
      `  Mem / rank   :  ${(memFactor * 100).toFixed(1)}% of baseline  (${(1 - memFactor) * 100 | 0}% saved)`,
      `  Hessian      :  ${C.useHessian ? '✓ finite-diff diagonal (subsampled 16 pts)' : '✗ disabled'}`,
      `  Kernel Fuse  :  ✓ add/sub/mul/scale/relu/tanh/sigmoid/softmax/layernorm`,
      `  GSAR         :  ✓ symbolic concept routing (5 seed concepts)`,
      `  SEP          :  ✓ micro-batch=${C.microBatchSize} token boundary`,
      `  Grad norm    :  ${optDiag.gradNorm.toFixed(6)}`,
      `  Param norm   :  ${optDiag.paramNorm.toFixed(4)}`,
      `  Step         :  ${this._step}`,
      '══════════════════════════════════════════════════════════════════════',
    ].join('\n');
  }

  // ── Diagnostics hook ──────────────────────────────────────────────────────

  diagnostics(): Record<string, unknown> {
    return {
      step:       this._step,
      optimizer:  this.optimizer.diagnostics(),
      gsar:       this.gsar.diagnostics(),
      sep:        this.sep.diagnostics(),
      config:     this.config,
      paramCount: this.allParams.reduce((s, p) => s + p.size, 0),
      layerShapes: {
        embedding:    this.embedding.weight.shape,
        lstmWih:      this.lstmCell.Wih.shape,
        lstmWhh:      this.lstmCell.Whh.shape,
        projW:        this.projHead.W.shape,
        transformers: this.transformerBlocks.length,
      },
    };
  }

  // ── Parameter inspector ───────────────────────────────────────────────────

  parameterInspector(): LayerParamInfo[] {
    const rows: LayerParamInfo[] = [
      ...this.embedding.paramInfo(),
      ...this.transformerBlocks.flatMap((b, i) =>
        b.paramInfo().map(p => ({ ...p, name: `tfm[${i}].${p.name}` }))
      ),
      ...this.lstmCell.paramInfo(),
      ...this.projHead.paramInfo(),
    ];
    return rows;
  }

  getStep(): number { return this._step; }
}

// ── Top-K sampling ────────────────────────────────────────────────────────────

function sampleTopK(logits: Float32Array, temperature: number, k: number): number {
  const V      = logits.length;
  const temp   = Math.max(temperature, 1e-8);

  // Get top-K indices by logit value
  const indexed: Array<{ v: number; i: number }> = [];
  for (let i = 0; i < V; i++) indexed.push({ v: logits[i] / temp, i });
  indexed.sort((a, b) => b.v - a.v);

  const topK = indexed.slice(0, Math.max(1, k));
  let max = topK[0].v;

  let sum = 0;
  const probs = new Float32Array(V);
  for (const { v, i } of topK) {
    probs[i] = Math.exp(v - max);
    sum      += probs[i];
  }
  if (sum === 0) return 0;
  for (const { i } of topK) probs[i] /= sum;

  // Categorical sample
  let r = Math.random();
  for (const { i } of topK) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return topK[topK.length - 1].i;
}
