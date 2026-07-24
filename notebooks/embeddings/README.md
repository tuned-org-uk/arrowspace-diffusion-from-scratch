# ALD-SC: Adaptive Latent Diffusion Semantic Codec

This subdirectory contains staged experiments for the **ALD-SC** architecture: a bidirectional text encoder that produces a retrieval-grade semantic embedding `z_s` and a compact residual code `r`, from which an autoregressive decoder (and optionally a latent diffusion planner) reconstructs the original text.

## Design principles

- `z_s ∈ ℝ^{d_s}` is L2-normalised, retrieval-optimised, and **invariant to surface form**.
- `r ∈ ℝ^{M × d_r}` is a rate-limited residual code that preserves lexical and positional detail needed for faithful reconstruction.
- Diffusion operates over a **latent canvas** `y₀ ∈ ℝ^{K × d_y}` built from `z_s` and `r`; it shapes coarse structure but does **not** replace the autoregressive decoder.
- Spectral conditioning `𝒯_t = g_{η(t)}(L_F)` is added only after reconstruction quality is validated.

## Staged plan

| Notebook | Stage | Goal | Key ablation |
|---|---|---|---|
| `01_encoder_decoder_baseline.ipynb` | Stage 1 | Plain T5 encode → decode reconstruction | BLEU / chrF / BERTScore baseline |
| `02_semantic_residual_split.ipynb` | Stage 2 | Disentangle `z_s` (contrastive) from `r` (quantised residual) | Retrieval accuracy vs reconstruction fidelity Pareto |
| `03_latent_diffusion_canvas.ipynb` | Stage 3 | Add flow-matching diffusion canvas; decode via cross-attention | Does diffusion canvas improve paraphrase-mode decoding? |
| `04_spectral_conditioning.ipynb` | Stage 4 | Inject `𝒯_t = g_t(L_F)` into noise schedule | Retrieval-conditioned generation quality vs isotropic baseline |

## Dependencies

```bash
uv pip install torch transformers sentence-transformers datasets evaluate
uv pip install vector-quantize-pytorch torchdiffeq
```

## Key metrics

- **Faithful mode** (`z_s + r`): chrF ≥ 0.85, BERTScore F1 ≥ 0.93
- **Semantic mode** (`z_s` only): NLI entailment ≥ 80%, retrieval R@1 ≥ 60% on MS-MARCO
- **Anti-collapse**: mutual information I(r; class_label) < I(z_s; class_label) at all epochs
