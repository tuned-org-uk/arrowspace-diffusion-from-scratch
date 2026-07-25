# Ship Checklist: Stage 2 issue #5 fixes

**Target:** `notebooks/embeddings/02_semantic_residual_split.ipynb`
**Issue:** #5 — fix disentanglement pressure, collapse probe, MI gate
**Ship date:** 260725-1314

## Fixes applied (all four points from issue #5)
- [x] 1. Contrastive loss backpropagated through trainable projection head (`lam_ret=0.1`)
- [x] 2. Collapse probe uses emotion labels (6 semantic classes), not sentence length
- [x] 3. MI gate implemented: classifier-based lower bound I(X;Y) = H(Y) - H(Y|X)
- [x] 4. Gated latent canvas: `sem + g(z_s) * res` prevents residual bypass

## Verification (executed notebook)
- [x] Faithful chrF = 97.10 (gate ≥ 85)
- [x] Semantic chrF = 16.08 (gate ≤ faithful − 10)
- [x] Anti-collapse: r_acc=0.445 ≤ z_acc=0.645
- [x] MI gate: I(r;class)=0.0000 < I(z_s;class)=0.5470
- [x] R@1 = 1.000
- [x] Verdict: PASS
