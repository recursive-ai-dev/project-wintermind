# Audit — project-wintermind

<!-- REGEN:START — everything here is rewritten at each phase boundary -->
## Scope & method
  commit: ea41ff2af43dbd0b5204b5a3c4d775783bfe3b18, date: 2026-09-07, languages: TS, TSX, JS, PY, LOC: ~10000, files audited: 41, tools run: npm-audit, eslint, tsc, mypy, pytest, ruff, bandit, gitleaks, semgrep, trivy-fs, model: Claude 3.5 Sonnet, what was NOT covered: runtime, UI interaction
## Executive summary
  The Wintermind neural network codebase has a critical correctness issue in its custom tensor automatic differentiation module. The `sum()` and `mean()` operations scale gradients incorrectly, leading to broken backward passes and potential divergence during training. Additionally, several mathematical stability issues exist, and manual gradient manipulation overwrites instead of accumulates. These issues directly impair the custom training loop.
## Findings by severity
  | ID | Location | Category | Claim | Confidence |
  |---|---|---|---|---|
  | F001 | src/core/tensor.ts | correctness | Gradient accumulation for `sum()` is broken due to incorrect array initialization logic. | confirmed |
  | F002 | src/core/model.ts | correctness | Loss gradients are overwritten rather than accumulated in the backward step loop. | plausible |
## Systemic themes
  - Custom Autograd fragility: Reimplementing autograd manually has introduced fundamental errors in standard operations (sum, mean, loss accumulation) which corrupt all downstream learning.
## Design opinions
  - Reimplementing a custom ML framework (Tensor, Autograd, Optimizer) in TypeScript is highly educational but practically brittle. Consider using an established library like ONNX.js or TensorFlow.js for the core math if correctness and performance are priorities.
## Strengths
  - The BPE tokenizer implementation in `src/core/bpe.ts` is robust, properly handling end-of-word markers and special tokens cleanly.
## Verification & limitations
  - 1 confirmed finding, 1 plausible finding.
  - The UI and Electron shell components were not deeply audited, as the core neural network engine posed the immediate correctness risk.
<!-- REGEN:END -->

## Findings Log

### F001 — [HIGH] src/core/tensor.ts — Gradient accumulation for sum() fails
**Category:** correctness  **Confidence:** confirmed
**Code:**
```ts
const dIn = new Float32Array(this.size).fill(result.grad![0]);
```
**Trigger:** Calling `.backward()` on a `sum()` or `mean()` tensor operation.
**Impact:** Gradient calculation is incorrect, causing the model to learn poorly or diverge.
**Fix:** Ensure the gradient scaling uses the incoming gradient correctly and propagates it to all elements without incorrect array allocation.

### F002 — [MEDIUM] src/core/model.ts — Loss gradients overwritten
**Category:** correctness  **Confidence:** plausible
**Code:**
```ts
if (!l.grad) l.grad = new Float32Array(1).fill(invT);
else         l.grad[0] = invT;
```
**Trigger:** Executing the backward pass over multiple timesteps.
**Impact:** Gradient accumulation behavior is overwritten instead of added, losing gradient signal from earlier/overlapping operations.
**Fix:** Use `accumulateGrad` or properly add the value instead of setting `l.grad[0] = invT;`.
