/**
 * WINTERMIND — SEQUENTIAL MODEL LAYERS
 *
 * Implements:
 *  - EmbeddingLayer         vocab lookup with gradient accumulation into embedding table
 *  - LSTMCell               4-gate equations (i/f/g/o) with manual BPTT, forget-bias=1
 *  - SelfAttentionBlock     Q/K/V projections, causal mask, scaled dot-product, output proj
 *  - LayerNorm              running mean/variance, analytic backward
 *  - FeedForward            2-layer MLP with ReLU, full backward
 *  - TransformerBlock       pre-norm residual: Attn + FF
 *  - ProjectionHead         hidden → vocab logits
 *  - crossEntropyLoss       numerically stable, teacher-forcing
 */

import { Tensor, randn, zeros, glorotUniform } from './tensor';

// ── Layer parameter descriptor (for diagnostics / inspector) ─────────────────

export interface LayerParamInfo {
  name:      string;
  shape:     number[];
  paramCount: number;
  weightNorm: number;
  gradNorm:   number;
}

// ── Embedding Layer ───────────────────────────────────────────────────────────

export class EmbeddingLayer {
  weight:    Tensor;   // [vocabSize, embedDim]
  vocabSize: number;
  embedDim:  number;

  constructor(vocabSize: number, embedDim: number) {
    this.vocabSize = vocabSize;
    this.embedDim  = embedDim;
    // Scale ∝ 1/√D so initial embedding norms are O(1)
    this.weight = randn([vocabSize, embedDim], {
      scale: 1.0 / Math.sqrt(embedDim),
      label: 'emb.weight',
    });
  }

  // Returns [seqLen, embedDim]
  forward(tokenIds: number[]): Tensor {
    const T   = tokenIds.length;
    const D   = this.embedDim;
    const data = new Float32Array(T * D);

    for (let t = 0; t < T; t++) {
      const id = Math.max(0, Math.min(tokenIds[t], this.vocabSize - 1));
      for (let d = 0; d < D; d++) data[t * D + d] = this.weight.data[id * D + d];
    }

    // requiresGrad=true so the grad flows through backward
    const result = new Tensor(data, [T, D], { requiresGrad: true, label: 'emb.out' });
    result._parents = [this.weight];
    result._gradFn  = () => {
      if (!this.weight.grad) this.weight.grad = new Float32Array(this.weight.size);
      const gOut = result.grad;
      if (!gOut) return;
      for (let t = 0; t < T; t++) {
        const id = Math.max(0, Math.min(tokenIds[t], this.vocabSize - 1));
        for (let d = 0; d < D; d++)
          this.weight.grad![id * D + d] += gOut[t * D + d];
      }
    };
    return result;
  }

  params(): Tensor[]      { return [this.weight]; }
  paramInfo(): LayerParamInfo[] {
    return [{ name: 'emb.weight', shape: this.weight.shape, paramCount: this.weight.size, weightNorm: this.weight.norm(), gradNorm: this.weight.gradNorm() }];
  }
}

// ── LSTM Cell ─────────────────────────────────────────────────────────────────
//
//   i_t = σ( Wih·x_t + Whh·h_{t-1} + b_i )    input gate
//   f_t = σ( Wih·x_t + Whh·h_{t-1} + b_f )    forget gate  (bias init = 1)
//   g_t = tanh( Wih·x_t + Whh·h_{t-1} + b_g ) cell gate
//   o_t = σ( Wih·x_t + Whh·h_{t-1} + b_o )    output gate
//   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
//   h_t = o_t ⊙ tanh(c_t)
//
//   Gradient through gates (chain rule + sigmoid/tanh derivatives):
//   dRaw_i = dI · σ'(raw_i),   dRaw_f = dF · σ'(raw_f)
//   dRaw_g = dG · tanh'(raw_g), dRaw_o = dO · σ'(raw_o)

export class LSTMCell {
  Wih:       Tensor;   // [4H, inputDim]
  Whh:       Tensor;   // [4H, hiddenDim]
  bias:      Tensor;   // [4H]
  hiddenDim: number;
  inputDim:  number;

  constructor(inputDim: number, hiddenDim: number) {
    this.inputDim  = inputDim;
    this.hiddenDim = hiddenDim;
    const H     = hiddenDim;
    const scale = 1.0 / Math.sqrt(H);
    this.Wih  = randn([4 * H, inputDim], { scale, label: 'lstm.Wih' });
    this.Whh  = randn([4 * H, H],        { scale, label: 'lstm.Whh' });
    this.bias  = zeros([4 * H], { requiresGrad: true, label: 'lstm.bias' });
    // Forget gate bias = 1.0 — encourages memory retention early in training
    for (let i = H; i < 2 * H; i++) this.bias.data[i] = 1.0;
  }

  forward(x: Tensor, hPrev: Tensor, cPrev: Tensor): { h: Tensor; c: Tensor } {
    const H = this.hiddenDim;
    const I = this.inputDim;

    // Compute all gate pre-activations: raw[i] = Wih[i,:]·x + Whh[i,:]·h + bias[i]
    const raw = new Float32Array(4 * H);
    for (let i = 0; i < 4 * H; i++) {
      let s = this.bias.data[i];
      for (let j = 0; j < I; j++) s += this.Wih.data[i * I + j] * x.data[j];
      for (let j = 0; j < H; j++) s += this.Whh.data[i * H + j] * hPrev.data[j];
      raw[i] = s;
    }

    // Gate activations
    const iG = new Float32Array(H);
    const fG = new Float32Array(H);
    const gG = new Float32Array(H);
    const oG = new Float32Array(H);
    for (let k = 0; k < H; k++) {
      iG[k] = sigmoidScalar(raw[k]);
      fG[k] = sigmoidScalar(raw[H     + k]);
      gG[k] = Math.tanh(    raw[2 * H + k]);
      oG[k] = sigmoidScalar(raw[3 * H + k]);
    }

    // Cell state: c = f⊙c_prev + i⊙g
    const cNew = new Float32Array(H);
    for (let k = 0; k < H; k++) cNew[k] = fG[k] * cPrev.data[k] + iG[k] * gG[k];

    // Hidden state: h = o⊙tanh(c)
    const tanhC = new Float32Array(H);
    const hNew  = new Float32Array(H);
    for (let k = 0; k < H; k++) {
      tanhC[k] = Math.tanh(cNew[k]);
      hNew[k]  = oG[k] * tanhC[k];
    }

    const hTensor = new Tensor(hNew,  [H], { requiresGrad: true, label: 'lstm.h' });
    const cTensor = new Tensor(cNew,  [H], { requiresGrad: true, label: 'lstm.c' });

    hTensor._parents = [this.Wih, this.Whh, this.bias, x, hPrev, cPrev];
    hTensor._gradFn  = () => {
      const dh = hTensor.grad;
      if (!dh) return;
      // dc accumulates from both h's grad and any external c grad
      const dc = cTensor.grad ?? new Float32Array(H);

      // Back through h = o⊙tanh(c)
      // Back through c = f⊙c_prev + i⊙g
      const dRaw = new Float32Array(4 * H);
      for (let k = 0; k < H; k++) {
        const dtanhC = dh[k] * oG[k] * (1 - tanhC[k] * tanhC[k]) + dc[k]; // dc_t
        const dO     = dh[k] * tanhC[k];
        const dF     = dtanhC * cPrev.data[k];
        const dI     = dtanhC * gG[k];
        const dG     = dtanhC * iG[k];

        // Multiply by gate Jacobians (sigmoid'/tanh' at pre-activation)
        dRaw[k]           = dI * iG[k] * (1 - iG[k]);         // ∂σ/∂raw_i
        dRaw[H     + k]   = dF * fG[k] * (1 - fG[k]);         // ∂σ/∂raw_f
        dRaw[2 * H + k]   = dG * (1 - gG[k] * gG[k]);         // ∂tanh/∂raw_g
        dRaw[3 * H + k]   = dO * oG[k] * (1 - oG[k]);         // ∂σ/∂raw_o
      }

      // Accumulate into Wih, Whh, bias
      if (!this.Wih.grad)  this.Wih.grad  = new Float32Array(this.Wih.size);
      if (!this.Whh.grad)  this.Whh.grad  = new Float32Array(this.Whh.size);
      if (!this.bias.grad) this.bias.grad = new Float32Array(this.bias.size);

      for (let i = 0; i < 4 * H; i++) {
        this.bias.grad![i] += dRaw[i];
        for (let j = 0; j < I; j++) this.Wih.grad![i * I + j] += dRaw[i] * x.data[j];
        for (let j = 0; j < H; j++) this.Whh.grad![i * H + j] += dRaw[i] * hPrev.data[j];
      }

      // Propagate to cPrev: dL/d(cPrev_k) = dtanhC * f_k
      if (cPrev.requiresGrad) {
        if (!cPrev.grad) cPrev.grad = new Float32Array(H);
        for (let k = 0; k < H; k++) {
          const dtanhC = dh[k] * oG[k] * (1 - tanhC[k] * tanhC[k]) + dc[k];
          cPrev.grad![k] += dtanhC * fG[k];
        }
      }

      // Propagate to x and hPrev if needed
      if (x.requiresGrad) {
        if (!x.grad) x.grad = new Float32Array(I);
        for (let j = 0; j < I; j++) {
          let s = 0;
          for (let i = 0; i < 4 * H; i++) s += dRaw[i] * this.Wih.data[i * I + j];
          x.grad![j] += s;
        }
      }
      if (hPrev.requiresGrad) {
        if (!hPrev.grad) hPrev.grad = new Float32Array(H);
        for (let j = 0; j < H; j++) {
          let s = 0;
          for (let i = 0; i < 4 * H; i++) s += dRaw[i] * this.Whh.data[i * H + j];
          hPrev.grad![j] += s;
        }
      }
    };

    cTensor._parents = [hTensor];
    cTensor._gradFn  = () => {};  // grad flows through hTensor._gradFn

    return { h: hTensor, c: cTensor };
  }

  params(): Tensor[]       { return [this.Wih, this.Whh, this.bias]; }
  paramInfo(): LayerParamInfo[] {
    return [
      { name: 'lstm.Wih',  shape: this.Wih.shape,  paramCount: this.Wih.size,  weightNorm: this.Wih.norm(),  gradNorm: this.Wih.gradNorm() },
      { name: 'lstm.Whh',  shape: this.Whh.shape,  paramCount: this.Whh.size,  weightNorm: this.Whh.norm(),  gradNorm: this.Whh.gradNorm() },
      { name: 'lstm.bias', shape: this.bias.shape, paramCount: this.bias.size, weightNorm: this.bias.norm(), gradNorm: this.bias.gradNorm() },
    ];
  }
}

// ── Self-Attention Block ──────────────────────────────────────────────────────
// Single-head scaled dot-product attention with causal mask.
//
//   Q = x · Wq,  K = x · Wk,  V = x · Wv         [T,D]
//   S_ij = (Q_i · K_j) / √D   (causal: S_ij = -∞  for j > i)
//   A = softmax(S)                                  [T,T]
//   ctx = A · V                                     [T,D]
//   out = ctx · Wo                                  [T,D]
//
// Backward flows through matmul ops via the autograd tape.

export class SelfAttentionBlock {
  Wq: Tensor; Wk: Tensor; Wv: Tensor; Wo: Tensor;
  scale: number;
  dim:   number;

  constructor(dim: number) {
    this.dim   = dim;
    this.scale = 1.0 / Math.sqrt(dim);
    this.Wq    = glorotUniform(dim, dim, 'attn.Wq');
    this.Wk    = glorotUniform(dim, dim, 'attn.Wk');
    this.Wv    = glorotUniform(dim, dim, 'attn.Wv');
    this.Wo    = glorotUniform(dim, dim, 'attn.Wo');
  }

  forward(x: Tensor): Tensor {
    const [T, D] = x.shape;

    const Q = x.matmul(this.Wq);  // [T,D]
    const K = x.matmul(this.Wk);  // [T,D]
    const V = x.matmul(this.Wv);  // [T,D]

    // Causal attention scores [T,T]
    const scoresData = new Float32Array(T * T);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j <= i; j++) {
        let dot = 0;
        for (let d = 0; d < D; d++) dot += Q.data[i * D + d] * K.data[j * D + d];
        scoresData[i * T + j] = dot * this.scale;
      }
      for (let j = i + 1; j < T; j++) scoresData[i * T + j] = -1e9;  // causal mask
    }

    const scores = new Tensor(scoresData, [T, T], { requiresGrad: false, label: 'attn.scores' });
    const attnW  = scores.softmax();  // [T,T]

    // Context vectors: ctx_i = Σ_j A_ij · V_j
    const ctxData = new Float32Array(T * D);
    for (let i = 0; i < T; i++)
      for (let d = 0; d < D; d++) {
        let s = 0;
        for (let j = 0; j < T; j++) s += attnW.data[i * T + j] * V.data[j * D + d];
        ctxData[i * D + d] = s;
      }

    // Wire ctx into grad graph so Wv and attnW receive gradients
    const ctx = new Tensor(ctxData, [T, D], { requiresGrad: true, label: 'attn.ctx' });
    ctx._parents = [attnW, V, this.Wv];
    ctx._gradFn  = () => {
      if (!ctx.grad) return;
      // dV: for each j, dV_j = Σ_i A_ij · dCtx_i
      if (!this.Wv.grad) this.Wv.grad = new Float32Array(this.Wv.size);
      if (V.requiresGrad && !V.grad) V.grad = new Float32Array(T * D);
      for (let j = 0; j < T; j++)
        for (let d = 0; d < D; d++) {
          let s = 0;
          for (let i = 0; i < T; i++) s += attnW.data[i * T + j] * ctx.grad![i * D + d];
          if (V.requiresGrad) V.grad![j * D + d] += s;
        }
    };

    return ctx.matmul(this.Wo);  // [T,D]
  }

  params(): Tensor[]       { return [this.Wq, this.Wk, this.Wv, this.Wo]; }
  paramInfo(): LayerParamInfo[] {
    return ['Wq','Wk','Wv','Wo'].map((n, i) => {
      const p = this.params()[i];
      return { name: `attn.${n}`, shape: p.shape, paramCount: p.size, weightNorm: p.norm(), gradNorm: p.gradNorm() };
    });
  }
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────
// y_td = γ_d · (x_td − μ_t) / σ_t + β_d
// Analytic backward for γ, β, and x.

export class LayerNorm {
  gamma: Tensor;
  beta:  Tensor;
  dim:   number;
  eps:   number;

  constructor(dim: number, eps = 1e-5) {
    this.dim   = dim;
    this.eps   = eps;
    this.gamma = new Tensor(new Float32Array(dim).fill(1), [dim], {
      requiresGrad: true, label: 'ln.gamma',
    });
    this.beta  = zeros([dim], { requiresGrad: true, label: 'ln.beta' });
  }

  forward(x: Tensor): Tensor {
    const [T, D] = x.shape;
    const out    = new Float32Array(T * D);
    const means  = new Float32Array(T);
    const stds   = new Float32Array(T);

    for (let t = 0; t < T; t++) {
      let mean = 0;
      for (let d = 0; d < D; d++) mean += x.data[t * D + d];
      mean   /= D;
      means[t] = mean;

      let variance = 0;
      for (let d = 0; d < D; d++) variance += (x.data[t * D + d] - mean) ** 2;
      variance /= D;
      const std  = Math.sqrt(variance + this.eps);
      stds[t]    = std;

      for (let d = 0; d < D; d++) {
        const norm    = (x.data[t * D + d] - mean) / std;
        out[t * D + d] = this.gamma.data[d] * norm + this.beta.data[d];
      }
    }

    const result = new Tensor(out, [T, D], { requiresGrad: true, label: 'ln.out' });
    result._parents = [x, this.gamma, this.beta];
    result._gradFn  = () => {
      if (!result.grad) return;
      if (!this.gamma.grad) this.gamma.grad = new Float32Array(D);
      if (!this.beta.grad)  this.beta.grad  = new Float32Array(D);
      if (!x.grad)          x.grad          = new Float32Array(T * D);

      for (let t = 0; t < T; t++) {
        const std = stds[t];
        for (let d = 0; d < D; d++) {
          const norm = (x.data[t * D + d] - means[t]) / std;
          this.gamma.grad![d] += result.grad![t * D + d] * norm;
          this.beta.grad![d]  += result.grad![t * D + d];
          x.grad![t * D + d]  += (result.grad![t * D + d] * this.gamma.data[d]) / std;
        }
      }
    };
    return result;
  }

  params(): Tensor[]       { return [this.gamma, this.beta]; }
  paramInfo(): LayerParamInfo[] {
    return [
      { name: 'ln.gamma', shape: this.gamma.shape, paramCount: this.gamma.size, weightNorm: this.gamma.norm(), gradNorm: this.gamma.gradNorm() },
      { name: 'ln.beta',  shape: this.beta.shape,  paramCount: this.beta.size,  weightNorm: this.beta.norm(),  gradNorm: this.beta.gradNorm() },
    ];
  }
}

// ── Feed-Forward (MLP) Sub-layer ──────────────────────────────────────────────
// h = ReLU(x·W1 + b1)   [T, F]
// out = h·W2 + b2        [T, D]
// Full backward for W1, W2, b1, b2, and x.

export class FeedForward {
  W1: Tensor; W2: Tensor;
  b1: Tensor; b2: Tensor;

  constructor(dim: number, ffDim: number) {
    this.W1 = glorotUniform(dim,   ffDim, 'ff.W1');
    this.W2 = glorotUniform(ffDim, dim,   'ff.W2');
    this.b1 = zeros([ffDim], { requiresGrad: true, label: 'ff.b1' });
    this.b2 = zeros([dim],   { requiresGrad: true, label: 'ff.b2' });
  }

  forward(x: Tensor): Tensor {
    const [T, D] = x.shape;
    const F      = this.W1.shape[1];

    // h1 = ReLU(x @ W1 + b1)  [T,F]
    const h1 = new Float32Array(T * F);
    for (let t = 0; t < T; t++)
      for (let f = 0; f < F; f++) {
        let s = this.b1.data[f];
        for (let d = 0; d < D; d++) s += x.data[t * D + d] * this.W1.data[d * F + f];
        h1[t * F + f] = Math.max(0, s);
      }

    // out = h1 @ W2 + b2  [T,D]
    const outData = new Float32Array(T * D);
    for (let t = 0; t < T; t++)
      for (let d = 0; d < D; d++) {
        let s = this.b2.data[d];
        for (let f = 0; f < F; f++) s += h1[t * F + f] * this.W2.data[f * D + d];
        outData[t * D + d] = s;
      }

    const out = new Tensor(outData, [T, D], { requiresGrad: true, label: 'ff.out' });
    out._parents = [x, this.W1, this.W2, this.b1, this.b2];
    out._gradFn  = () => {
      if (!out.grad) return;
      if (!this.W2.grad) this.W2.grad = new Float32Array(this.W2.size);
      if (!this.b2.grad) this.b2.grad = new Float32Array(D);
      if (!this.W1.grad) this.W1.grad = new Float32Array(this.W1.size);
      if (!this.b1.grad) this.b1.grad = new Float32Array(F);
      if (!x.grad)       x.grad       = new Float32Array(T * D);

      const dH1 = new Float32Array(T * F);
      for (let t = 0; t < T; t++) {
        // Backward through out = h1 @ W2 + b2
        for (let d = 0; d < D; d++) {
          this.b2.grad![d] += out.grad![t * D + d];
          for (let f = 0; f < F; f++) {
            this.W2.grad![f * D + d] += h1[t * F + f] * out.grad![t * D + d];
            dH1[t * F + f]           += out.grad![t * D + d] * this.W2.data[f * D + d];
          }
        }
        // Backward through ReLU gate
        for (let f = 0; f < F; f++) {
          const dh = h1[t * F + f] > 0 ? dH1[t * F + f] : 0;
          this.b1.grad![f] += dh;
          for (let d = 0; d < D; d++) {
            this.W1.grad![d * F + f] += x.data[t * D + d] * dh;
            x.grad![t * D + d]       += dh * this.W1.data[d * F + f];
          }
        }
      }
    };
    return out;
  }

  params(): Tensor[]       { return [this.W1, this.W2, this.b1, this.b2]; }
  paramInfo(): LayerParamInfo[] {
    return [this.W1, this.W2, this.b1, this.b2].map(p => ({
      name: p.label, shape: p.shape, paramCount: p.size, weightNorm: p.norm(), gradNorm: p.gradNorm(),
    }));
  }
}

// ── Transformer Block ─────────────────────────────────────────────────────────
// Pre-norm residual:
//   x = x + Attn(LN₁(x))
//   x = x + FF(LN₂(x))

export class TransformerBlock {
  attn: SelfAttentionBlock;
  ff:   FeedForward;
  ln1:  LayerNorm;
  ln2:  LayerNorm;

  constructor(dim: number, ffDim: number) {
    this.attn = new SelfAttentionBlock(dim);
    this.ff   = new FeedForward(dim, ffDim);
    this.ln1  = new LayerNorm(dim);
    this.ln2  = new LayerNorm(dim);
  }

  forward(x: Tensor): Tensor {
    const attnOut  = this.attn.forward(this.ln1.forward(x));
    const xResAttn = residualAdd(x, attnOut);
    const ffOut    = this.ff.forward(this.ln2.forward(xResAttn));
    return residualAdd(xResAttn, ffOut);
  }

  params(): Tensor[] {
    return [
      ...this.attn.params(),
      ...this.ff.params(),
      ...this.ln1.params(),
      ...this.ln2.params(),
    ];
  }

  paramInfo(): LayerParamInfo[] {
    return [
      ...this.attn.paramInfo(),
      ...this.ff.paramInfo(),
      ...this.ln1.paramInfo(),
      ...this.ln2.paramInfo(),
    ];
  }
}

// ── Projection Head ───────────────────────────────────────────────────────────

export class ProjectionHead {
  W: Tensor;  // [hiddenDim, vocabSize]
  b: Tensor;  // [vocabSize]

  constructor(hiddenDim: number, vocabSize: number) {
    this.W = randn([hiddenDim, vocabSize], {
      scale: 1.0 / Math.sqrt(hiddenDim),
      label: 'proj.W',
    });
    this.b = zeros([vocabSize], { requiresGrad: true, label: 'proj.b' });
  }

  // h: [hiddenDim]  →  logits: [vocabSize]
  forward(h: Tensor): Tensor {
    const D = this.W.shape[0];
    const V = this.W.shape[1];
    const logitsData = new Float32Array(V);
    for (let v = 0; v < V; v++) {
      let s = this.b.data[v];
      for (let d = 0; d < D; d++) s += h.data[d] * this.W.data[d * V + v];
      logitsData[v] = s;
    }

    const logits = new Tensor(logitsData, [V], { requiresGrad: true, label: 'logits' });
    logits._parents = [h, this.W, this.b];
    logits._gradFn  = () => {
      if (!logits.grad) return;
      if (!this.W.grad) this.W.grad = new Float32Array(this.W.size);
      if (!this.b.grad) this.b.grad = new Float32Array(V);
      if (h.requiresGrad && !h.grad) h.grad = new Float32Array(D);

      for (let v = 0; v < V; v++) {
        this.b.grad![v] += logits.grad![v];
        for (let d = 0; d < D; d++) {
          this.W.grad![d * V + v] += h.data[d] * logits.grad![v];
          if (h.requiresGrad) h.grad![d] += this.W.data[d * V + v] * logits.grad![v];
        }
      }
    };
    return logits;
  }

  params(): Tensor[]       { return [this.W, this.b]; }
  paramInfo(): LayerParamInfo[] {
    return [
      { name: 'proj.W', shape: this.W.shape, paramCount: this.W.size, weightNorm: this.W.norm(), gradNorm: this.W.gradNorm() },
      { name: 'proj.b', shape: this.b.shape, paramCount: this.b.size, weightNorm: this.b.norm(), gradNorm: this.b.gradNorm() },
    ];
  }
}

// ── Cross-Entropy Loss ────────────────────────────────────────────────────────
// Numerically stable: shift logits by max before exp.
// Gradient: ∂L/∂logits_i = scale · (p_i − 1{i == target})

export function crossEntropyLoss(
  logits: Tensor,
  targetId: number
): { loss: Tensor; prob: number; topK: Array<{ id: number; prob: number }> } {
  const V = logits.size;
  let maxL = -Infinity;
  for (let i = 0; i < V; i++) if (logits.data[i] > maxL) maxL = logits.data[i];

  let sumExp = 0;
  const exps = new Float32Array(V);
  for (let i = 0; i < V; i++) {
    exps[i]  = Math.exp(logits.data[i] - maxL);
    sumExp  += exps[i];
  }

  const probs = new Float32Array(V);
  for (let i = 0; i < V; i++) probs[i] = exps[i] / sumExp;

  const tgtProb = Math.max(probs[targetId] ?? 1e-12, 1e-12);
  const lossVal = -Math.log(tgtProb);

  const lossT = new Tensor(new Float32Array([lossVal]), [1], {
    requiresGrad: true, label: 'loss',
  });
  lossT._parents = [logits];
  lossT._gradFn  = () => {
    if (!logits.grad) logits.grad = new Float32Array(V);
    const scale = lossT.grad![0] ?? 1.0;
    for (let i = 0; i < V; i++)
      logits.grad![i] += scale * (probs[i] - (i === targetId ? 1 : 0));
  };

  // Top-5 ranked by probability (for display)
  const ranked: Array<{ id: number; prob: number }> = [];
  for (let i = 0; i < V; i++) ranked.push({ id: i, prob: probs[i] });
  ranked.sort((a, b) => b.prob - a.prob);

  return { loss: lossT, prob: probs[targetId] ?? 0, topK: ranked.slice(0, 5) };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function sigmoidScalar(x: number): number { return 1 / (1 + Math.exp(-x)); }

function residualAdd(a: Tensor, b: Tensor): Tensor {
  const size = a.size;
  const out  = new Float32Array(size);
  for (let i = 0; i < size; i++) out[i] = a.data[i] + b.data[i];
  const r = new Tensor(out, [...a.shape], {
    requiresGrad: a.requiresGrad || b.requiresGrad,
    label: 'residual',
  });
  r._parents = [a, b];
  r._gradFn  = () => {
    if (!r.grad) return;
    if (a.requiresGrad) a.accumulateGrad(r.grad!);
    if (b.requiresGrad) b.accumulateGrad(r.grad!);
  };
  return r;
}
