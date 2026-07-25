# Stage 2 Issue #5 Ship Summary

## Changes
1. **Trainable projection head** (`SemProjHead`, identity-init) on frozen MiniLM; contrastive loss backpropagated through it (`lam_ret=0.1`).
2. **Gated latent canvas**: `canvas = sem + sigmoid(gate(z_s)) * res` — the residual path is modulated by the semantic embedding, preventing bypass.
3. **Semantic-class probe**: emotion labels (6 classes) from `dair-ai/emotion` replace sentence length.
4. **MI gate**: classifier-based lower bound `I(X;Y) = H(Y) - H(Y|X)` via regularised logistic regression.

## Metrics
```json
{
  "faithful_chrf": 97.1,
  "semantic_chrf": 16.08,
  "info_gap": 81.01,
  "ret_r1": 1.0,
  "z_class_acc": 0.645,
  "r_class_acc": 0.445,
  "mi_z": 0.547,
  "mi_r": 0.0,
  "pass": true
}
```
