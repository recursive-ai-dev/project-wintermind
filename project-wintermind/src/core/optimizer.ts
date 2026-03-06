/**
 * WINTERMIND — OPTIMIZER MODULE
 *
 * Implements:
 *  - Nesterov Accelerated Gradient (NAG) with Sutskever et al. 2013 formulation
 *  - Hessian diagonal via finite differences (subsampled, O(16) FD evaluations)
 *  - ZeRO-stage memory sharding simulation (stages 0–3) with exact byte analysis
 *  - Global gradient L2 norm computation and clipping
 *  - Cosine annealing and warmup-linear LR schedulers
 */

import { Tensor } from './tensor';

export interface NAGConfig {
  lr:           number;
  momentum:     number;
  weightDecay:  number;
  clipGradNorm: number;
  zeroStage:    0 | 1 | 2 | 3;
  numRanks:     number;
  useHessian:   boolean;
}

export class NAGOptimizer {
  private params:      Tensor[];
  private velocity:    Float32Array[];    // v_{t-1} per parameter tensor
  private hessianDiag: Float32Array[];    // H_ii diagonal estimate per parameter
  private config:      NAGConfig;
  private _step:       number = 0;

  constructor(params: Tensor[], config: Partial<NAGConfig> = {}) {
    this.params = params;
    this.config = {
      lr:           config.lr           ?? 1e-3,
      momentum:     config.momentum     ?? 0.9,
      weightDecay:  config.weightDecay  ?? 1e-4,
      clipGradNorm: config.clipGradNorm ?? 1.0,
      zeroStage:    config.zeroStage    ?? 1,
      numRanks:     config.numRanks     ?? 4,
      useHessian:   config.useHessian   ?? false,
    };
    this.velocity    = params.map(p => new Float32Array(p.size));
    // Initialize Hessian diagonal to 1 (identity preconditioner — no curvature bias)
    this.hessianDiag = params.map(p => new Float32Array(p.size).fill(1.0));
  }

  // ── Hessian diagonal via finite differences ────────────────────────────────
  //
  // H_ii ≈ (f(x + εe_i) + f(x − εe_i) − 2f(x)) / ε²
  //
  // Subsampled: stride = max(1, size/16) so at most 16 FD evaluations per param.
  // Each sampled diagonal entry is broadcast across its stride window.

  estimateHessianDiag(paramIdx: number, lossFn: () => number): void {
    const p      = this.params[paramIdx];
    const h      = this.hessianDiag[paramIdx];
    const eps    = 1e-4;
    const f0     = lossFn();
    const stride = Math.max(1, Math.floor(p.size / 16));

    for (let i = 0; i < p.size; i += stride) {
      const orig   = p.data[i];
      p.data[i]    = orig + eps;
      const fPlus  = lossFn();
      p.data[i]    = orig - eps;
      const fMinus = lossFn();
      p.data[i]    = orig;  // restore

      // Absolute value guards against numerical noise giving negative curvature
      const hii = Math.abs((fPlus + fMinus - 2 * f0) / (eps * eps)) + 1e-8;
      for (let j = i; j < Math.min(i + stride, p.size); j++) h[j] = hii;
    }
  }

  // ── Global gradient L2 norm ────────────────────────────────────────────────

  globalGradNorm(): number {
    let norm2 = 0;
    for (const p of this.params) {
      if (!p.grad) continue;
      for (let i = 0; i < p.size; i++) norm2 += p.grad[i] * p.grad[i];
    }
    return Math.sqrt(norm2);
  }

  private clipGrads(globalNorm: number): void {
    const clip = this.config.clipGradNorm;
    if (globalNorm <= clip || globalNorm === 0) return;
    const scale = clip / globalNorm;
    for (const p of this.params) {
      if (!p.grad) continue;
      for (let i = 0; i < p.size; i++) p.grad[i] *= scale;
    }
  }

  // ── NAG parameter update ───────────────────────────────────────────────────
  //
  // Sutskever et al. (2013) — "On the importance of initialization and
  // momentum in deep learning", ICML 2013:
  //
  //   g_t   = ∇L(θ_{t-1}) + λ·θ_{t-1}            weight-decayed gradient
  //   v_t   = μ·v_{t-1} + lr·H⁻¹·g_t              velocity update
  //   θ_t   = θ_{t-1} − (1 + μ)·v_t + μ·v_{t-1}  look-ahead correction
  //
  // The look-ahead correction (1+μ)v_t − μv_{t-1} approximates the gradient
  // at the "future" point θ − μv, yielding faster convergence than standard
  // momentum for convex problems and empirically better results in practice.

  update(lossFn?: () => number): { globalNorm: number; step: number } {
    this._step++;
    const mu = this.config.momentum;
    const lr = this.config.lr;
    const wd = this.config.weightDecay;

    // Compute and clip global gradient norm
    const globalNorm = this.globalGradNorm();
    this.clipGrads(globalNorm);

    for (let pidx = 0; pidx < this.params.length; pidx++) {
      const p = this.params[pidx];
      if (!p.grad) continue;

      // Optionally refresh Hessian diagonal for this parameter
      if (this.config.useHessian && lossFn) {
        this.estimateHessianDiag(pidx, lossFn);
      }

      const v = this.velocity[pidx];
      const H = this.hessianDiag[pidx];

      for (let i = 0; i < p.size; i++) {
        const g      = p.grad[i] + wd * p.data[i];           // weight-decayed gradient
        const hScale = this.config.useHessian
          ? 1.0 / Math.max(H[i], 1e-8)                       // H⁻¹ preconditioner
          : 1.0;
        const vOld   = v[i];
        v[i]         = mu * vOld + lr * hScale * g;           // v_t
        p.data[i]   -= (1.0 + mu) * v[i] - mu * vOld;        // NAG look-ahead step
      }
    }

    return { globalNorm, step: this._step };
  }

  // Zero out all gradients (fill, not null — params always have a grad buffer post-init)
  zeroGrads(): void {
    for (const p of this.params) {
      if (p.grad) p.grad.fill(0);
      else        p.grad = new Float32Array(p.size);
    }
  }

  getStep():          number     { return this._step; }
  getLR():            number     { return this.config.lr; }
  setLR(lr: number):  void       { this.config.lr = lr; }
  getConfig():        NAGConfig  { return { ...this.config }; }

  // ── ZeRO memory analysis ──────────────────────────────────────────────────
  //
  // Memory per element (fp32):
  //   4 bytes — parameters
  //   4 bytes — gradients
  //   8 bytes — optimizer states (momentum buffer 4B + Hessian diagonal 4B)
  //   ─────────────────────────
  //   16 bytes total baseline
  //
  // ZeRO-1: optimizer states sharded → 4 (param) + 4 (grad) + 8/R per rank
  // ZeRO-2: + gradients sharded      → 4 (param) + (4 + 8)/R per rank
  // ZeRO-3: + parameters sharded     → (4 + 4 + 8)/R = 16/R per rank

  memoryReductionFactor(): number {
    const R       = Math.max(this.config.numRanks, 1);
    const baseline = 16; // bytes per element
    let perRank: number;
    switch (this.config.zeroStage) {
      case 0:  perRank = 16;             break;  // no sharding
      case 1:  perRank = 4 + 4 + 8 / R; break;  // opt states sharded
      case 2:  perRank = 4 + 12 / R;    break;  // opt states + grads sharded
      case 3:  perRank = 16 / R;         break;  // all sharded
      default: perRank = 16;
    }
    return perRank / baseline;
  }

  zeroStageDescription(): string {
    const R = this.config.numRanks;
    const s = this.config.zeroStage;
    const d: Record<number, string> = {
      0: 'ZeRO-0: No sharding — full replication on every rank',
      1: `ZeRO-1: Optimizer states sharded across ${R} ranks`,
      2: `ZeRO-2: Optimizer states + gradients sharded across ${R} ranks`,
      3: `ZeRO-3: Parameters + gradients + optimizer states fully sharded across ${R} ranks`,
    };
    return d[s];
  }

  diagnostics(): Record<string, number> {
    const totalParams = this.params.reduce((s, p) => s + p.size, 0);
    const paramNorm2  = this.params.reduce((s, p) => s + p.norm() ** 2, 0);
    return {
      step:               this._step,
      lr:                 this.config.lr,
      momentum:           this.config.momentum,
      weightDecay:        this.config.weightDecay,
      clipGradNorm:       this.config.clipGradNorm,
      gradNorm:           this.globalGradNorm(),
      paramNorm:          Math.sqrt(paramNorm2),
      totalParams,
      memReductionFactor: this.memoryReductionFactor(),
      numRanks:           this.config.numRanks,
      zeroStage:          this.config.zeroStage,
    };
  }
}

// ── Learning-rate schedulers ──────────────────────────────────────────────────

/**
 * Cosine Annealing:
 *   lr(t) = lr_min + ½(lr_max − lr_min)(1 + cos(πt / T))
 */
export function cosineAnnealingLR(
  optimizer:  NAGOptimizer,
  step:       number,
  totalSteps: number,
  minLR = 1e-5,
  maxLR = 1e-3
): number {
  const lr = minLR + 0.5 * (maxLR - minLR) *
    (1 + Math.cos((Math.PI * step) / Math.max(totalSteps, 1)));
  optimizer.setLR(lr);
  return lr;
}

/**
 * Linear warmup → linear decay:
 *   if step < warmupSteps:  lr = maxLR · step / warmupSteps
 *   else:                   lr = minLR + (maxLR − minLR) · (1 − progress)
 */
export function warmupLinearSchedule(
  optimizer:   NAGOptimizer,
  step:        number,
  warmupSteps: number,
  maxLR       = 1e-3,
  totalSteps  = 1000,
  minLR       = 1e-5
): number {
  let lr: number;
  if (step < warmupSteps) {
    lr = maxLR * (step / Math.max(warmupSteps, 1));
  } else {
    const progress = (step - warmupSteps) / Math.max(totalSteps - warmupSteps, 1);
    lr = minLR + (maxLR - minLR) * (1 - progress);
  }
  optimizer.setLR(Math.max(lr, minLR));
  return Math.max(lr, minLR);
}
