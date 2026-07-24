# Stage 2 Ship Summary

## What was shipped
The corrected `notebooks/embeddings/02_semantic_residual_split.ipynb` now trains and satisfies the Stage 2 metric gate.

## Key changes from original
1. **Dataset fix:** `sentence-transformers/sentence-compression` no longer exposes `ex['set']`; now uses `google-research-datasets/paws` `labeled_final` paraphrase pairs.
2. **Bottleneck architecture:** replaced the mean-pool-to-`M` bottleneck with a **Perceiver-style bottleneck** in which `M` learned queries cross-attend to the full token sequence, preserving positional/lexical information.
3. **Generation fix:** wrapped `encoder_outputs` in `transformers.modeling_outputs.BaseModelOutput` and provided `bos_token_id` to avoid `ValueError` in `t5_model.generate()`.
4. **Capacity / schedule:** `M=32`, continuous residual, AdamW `lr=1e-4`, 30 epochs on 1000 PAWS pairs.
5. **Dead loss removed:** the contrastive loss now runs under `torch.no_grad()` as a monitor (the semantic encoder is frozen, so it has no trainable path).

## Verification results
```json
{
  "faithful_chrf": 94.58,
  "semantic_chrf": 15.36,
  "info_gap": 79.22,
  "ret_r1": 1.0,
  "z_class_acc": 0.175,
  "r_class_acc": 0.175,
  "pass": true
}
```

## Execution command
```bash
jupyter nbconvert --to notebook --execute 02_semantic_residual_split.ipynb --output 02_semantic_residual_split_executed.ipynb
```

## Artifacts
- `notebooks/embeddings/02_semantic_residual_split.ipynb` (corrected)
- `notebooks/embeddings/02_semantic_residual_split.ipynb.bak` (backup)
- `notebooks/embeddings/02_semantic_residual_split_executed.ipynb`
- `notebooks/embeddings/pareto_capacity_sweep.png`
- `autoresearch/ship-260724-1721/` (this log)
