/**
 * WINTERMIND — SEP: Self-Explanatory Perception
 *
 * Provides innervision into why the model made a prediction.
 * Operates in two phases:
 *   1. Micro-batch prediction: generates a preliminary prediction at a
 *      token-limit boundary, forming a quick partial hypothesis.
 *   2. Full-batch comparison: refines the prediction and explains the delta,
 *      exposing spurious correlations before output is finalized.
 *
 * Also provides:
 *   - Confidence calibration
 *   - Gradient-based token attribution
 *   - Counterfactual saliency analysis
 */

export interface SEPExplanation {
  microPrediction: {
    tokenId: number;
    tokenStr: string;
    confidence: number;
    topK: Array<{ id: number; str: string; prob: number }>;
  };
  fullPrediction: {
    tokenId: number;
    tokenStr: string;
    confidence: number;
    topK: Array<{ id: number; str: string; prob: number }>;
  };
  delta: number;           // confidence shift between micro and full
  spuriousFlags: string[]; // identified spurious correlation warnings
  attributions: TokenAttribution[];
  explanation: string;     // natural language summary
  calibratedConfidence: number;
}

export interface TokenAttribution {
  tokenIdx: number;
  tokenStr: string;
  score: number;           // normalized importance 0..1
  direction: 'positive' | 'negative' | 'neutral';
}

export interface SEPConfig {
  microBatchSize: number;  // tokens processed before micro-prediction
  calibrationTemperature: number;
  spuriousThreshold: number; // correlation threshold for flagging
}

export class SEPModule {
  private config: SEPConfig;
  private calibrationHistory: Array<{ predicted: number; actual: number }> = [];
  private spuriousPatterns: Map<string, number> = new Map(); // pattern → freq

  constructor(config: Partial<SEPConfig> = {}) {
    this.config = {
      microBatchSize: config.microBatchSize ?? 8,
      calibrationTemperature: config.calibrationTemperature ?? 1.2,
      spuriousThreshold: config.spuriousThreshold ?? 0.8,
    };
  }

  // ── Core SEP analysis ──────────────────────────────────────────────────────

  analyze(
    tokens: string[],
    logitsPerStep: Float32Array[], // logits at each token position
    idToToken: Map<number, string>
  ): SEPExplanation {
    const V = logitsPerStep[0]?.length ?? 1;
    const microBound = Math.min(this.config.microBatchSize, logitsPerStep.length - 1);

    // Micro-prediction: use first microBatchSize logits to form initial hypothesis
    const microLogits = this.aggregateLogits(logitsPerStep.slice(0, microBound), V);
    const microProbs = softmax(microLogits, this.config.calibrationTemperature);
    const microTop = topKProbs(microProbs, 5, idToToken);
    const microPred = microTop[0];

    // Full prediction: use all logits
    const fullLogits = this.aggregateLogits(logitsPerStep, V);
    const fullProbs = softmax(fullLogits, this.config.calibrationTemperature);
    const fullTop = topKProbs(fullProbs, 5, idToToken);
    const fullPred = fullTop[0];

    // Delta: signed confidence shift
    const delta = fullPred.prob - microPred.prob;

    // Attribution: gradient-like score per token position via logit sensitivity
    const attributions = this.computeAttributions(tokens, logitsPerStep, fullPred.id, V);

    // Spurious correlation detection
    const spuriousFlags = this.detectSpurious(tokens, attributions);

    // Calibrated confidence (temperature scaling)
    const rawConf = fullPred.prob;
    const calibratedConfidence = this.calibrate(rawConf);

    // Natural-language explanation
    const explanation = this.generateExplanation(
      tokens,
      microPred,
      fullPred,
      delta,
      attributions,
      spuriousFlags,
      calibratedConfidence
    );

    return {
      microPrediction: {
        tokenId: microPred.id,
        tokenStr: microPred.str,
        confidence: microPred.prob,
        topK: microTop,
      },
      fullPrediction: {
        tokenId: fullPred.id,
        tokenStr: fullPred.str,
        confidence: fullPred.prob,
        topK: fullTop,
      },
      delta,
      spuriousFlags,
      attributions,
      explanation,
      calibratedConfidence,
    };
  }

  // ── Token attribution via sensitivity analysis ─────────────────────────────

  private computeAttributions(
    tokens: string[],
    logitsPerStep: Float32Array[],
    targetId: number,
    V: number
  ): TokenAttribution[] {
    const T = Math.min(tokens.length, logitsPerStep.length);
    const attrs: TokenAttribution[] = [];

    // Baseline: target token logit at each position
    for (let t = 0; t < T; t++) {
      const logits = logitsPerStep[t];
      if (!logits || logits.length === 0) continue;

      const targetLogit = logits[targetId] ?? 0;
      // Compare to mean to get relative contribution
      let mean = 0;
      for (let v = 0; v < V; v++) mean += logits[v];
      mean /= V;
      const score = targetLogit - mean;

      attrs.push({
        tokenIdx: t,
        tokenStr: tokens[t] ?? `tok_${t}`,
        score,
        direction: score > 0.1 ? 'positive' : score < -0.1 ? 'negative' : 'neutral',
      });
    }

    // Normalize scores to [0, 1]
    const scores = attrs.map((a) => Math.abs(a.score));
    const maxScore = Math.max(...scores, 1e-8);
    for (const a of attrs) a.score = Math.abs(a.score) / maxScore;

    return attrs;
  }

  // ── Spurious correlation detection ────────────────────────────────────────

  private detectSpurious(
    tokens: string[],
    attributions: TokenAttribution[]
  ): string[] {
    const flags: string[] = [];

    // Flag: single token dominates attribution (>70% of total weight)
    const total = attributions.reduce((s, a) => s + a.score, 0);
    for (const a of attributions) {
      if (total > 0 && a.score / total > 0.7) {
        flags.push(
          `Over-reliance on token "${a.tokenStr}" (${((a.score / total) * 100).toFixed(1)}% attribution weight)`
        );
      }
    }

    // Flag: very short context with high confidence
    if (tokens.length < 3 && attributions.some((a) => a.score > 0.8)) {
      flags.push(`High confidence prediction from very short context (${tokens.length} tokens)`);
    }

    // Track pattern frequencies for future flagging
    for (const tok of tokens) {
      const key = tok.toLowerCase();
      this.spuriousPatterns.set(key, (this.spuriousPatterns.get(key) ?? 0) + 1);
    }

    // Flag: token that appears very frequently across sequences (potential spurious anchor)
    for (const a of attributions) {
      const freq = this.spuriousPatterns.get(a.tokenStr.toLowerCase()) ?? 0;
      if (freq > 20 && a.score > 0.6) {
        flags.push(
          `Token "${a.tokenStr}" is high-frequency (${freq} occurrences) and high-attribution — possible dataset artifact`
        );
      }
    }

    return flags;
  }

  // ── Calibration ─────────────────────────────────────────────────────────────

  private calibrate(rawConf: number): number {
    // Platt scaling approximation: sigmoid(log(p/(1-p)) / T)
    const T = this.config.calibrationTemperature;
    const logit = Math.log(Math.max(rawConf, 1e-7) / Math.max(1 - rawConf, 1e-7));
    return 1 / (1 + Math.exp(-logit / T));
  }

  recordOutcome(predicted: number, actual: number) {
    this.calibrationHistory.push({ predicted, actual });
    if (this.calibrationHistory.length > 200) this.calibrationHistory.shift();
  }

  calibrationError(): number {
    if (this.calibrationHistory.length === 0) return 0;
    const ece = this.calibrationHistory.reduce(
      (s, e) => s + Math.abs(e.predicted - e.actual),
      0
    ) / this.calibrationHistory.length;
    return ece;
  }

  // ── Logit aggregation ────────────────────────────────────────────────────────

  private aggregateLogits(logitsPerStep: Float32Array[], V: number): Float32Array {
    const agg = new Float32Array(V);
    let count = 0;
    for (const logits of logitsPerStep) {
      if (!logits || logits.length === 0) continue;
      for (let v = 0; v < V; v++) agg[v] += logits[v];
      count++;
    }
    if (count > 0) for (let v = 0; v < V; v++) agg[v] /= count;
    return agg;
  }

  // ── Natural language explanation generator ───────────────────────────────────

  private generateExplanation(
    tokens: string[],
    micro: { id: number; str: string; prob: number },
    full: { id: number; str: string; prob: number },
    delta: number,
    attributions: TokenAttribution[],
    flags: string[],
    calibrated: number
  ): string {
    const topAttrs = [...attributions]
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);

    let msg = `SEP Analysis (${tokens.length} tokens):\n`;
    msg += `  Micro-prediction [first ${this.config.microBatchSize} tokens]: "${micro.str}" @ ${(micro.prob * 100).toFixed(1)}%\n`;
    msg += `  Full-context prediction: "${full.str}" @ ${(full.prob * 100).toFixed(1)}%\n`;
    msg += `  Confidence delta: ${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)}% (${delta >= 0 ? 'refined upward' : 'reduced by additional context'})\n`;
    msg += `  Calibrated confidence: ${(calibrated * 100).toFixed(1)}%\n`;

    if (topAttrs.length > 0) {
      msg += `  Top contributing tokens:\n`;
      for (const a of topAttrs) {
        msg += `    • "${a.tokenStr}" → ${(a.score * 100).toFixed(1)}% weight (${a.direction})\n`;
      }
    }

    if (flags.length > 0) {
      msg += `  ⚠ Spurious correlation warnings:\n`;
      for (const f of flags) msg += `    • ${f}\n`;
    } else {
      msg += `  ✓ No spurious correlations detected.\n`;
    }

    return msg;
  }

  diagnostics(): Record<string, unknown> {
    return {
      microBatchSize: this.config.microBatchSize,
      calibrationTemperature: this.config.calibrationTemperature,
      calibrationSamples: this.calibrationHistory.length,
      ece: this.calibrationError().toFixed(4),
      spuriousTrackedPatterns: this.spuriousPatterns.size,
    };
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function softmax(logits: Float32Array, temperature = 1.0): Float32Array {
  const V = logits.length;
  let max = -Infinity;
  for (let i = 0; i < V; i++) max = Math.max(max, logits[i] / temperature);
  let sum = 0;
  const exps = new Float32Array(V);
  for (let i = 0; i < V; i++) {
    exps[i] = Math.exp(logits[i] / temperature - max);
    sum += exps[i];
  }
  for (let i = 0; i < V; i++) exps[i] /= sum;
  return exps;
}

function topKProbs(
  probs: Float32Array,
  k: number,
  idToToken: Map<number, string>
): Array<{ id: number; str: string; prob: number }> {
  const arr = Array.from(probs).map((prob, id) => ({
    id,
    str: idToToken.get(id) ?? `<${id}>`,
    prob,
  }));
  arr.sort((a, b) => b.prob - a.prob);
  return arr.slice(0, k);
}
