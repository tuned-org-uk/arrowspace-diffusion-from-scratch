# Ship Checklist: Stage 2 notebook fix

**Target:** `notebooks/embeddings/02_semantic_residual_split.ipynb`
**Type:** notebook / code artifact
**Ship date:** 260724-1721

## Phase 1 — Identify
- [x] Stage 2 pass criteria understood (chrF, semantic gap, anti-collapse, R@1)

## Phase 2 — Inventory
- [x] Files changed inspected
  - `notebooks/embeddings/02_semantic_residual_split.ipynb`
  - backup `notebooks/embeddings/02_semantic_residual_split.ipynb.bak`
  - executed output `notebooks/embeddings/02_semantic_residual_split_executed.ipynb`
  - generated plot `notebooks/embeddings/pareto_capacity_sweep.png`
- [x] Dependencies installed (`sentence-transformers==5.6.1`, `evaluate`, `vector-quantize-pytorch`, `scikit-learn`, `sacrebleu`)
- [x] Dataset selected: `google-research-datasets/paws` `labeled_final`

## Phase 3 — Checklist
- [x] JSON syntax in notebook repaired
- [x] Dataset loading fixed (schema drift in `sentence-compression`)
- [x] `generate()` uses `BaseModelOutput` + `bos_token_id`
- [x] Mean-pool bottleneck replaced with Perceiver query bottleneck
- [x] Residual capacity raised to `M=32`
- [x] Training epochs raised to 30 for convergence
- [x] Frozen contrastive loss moved to monitor-only

## Phase 4 — Prepare (all via executed notebook)
- [x] Training completed (30 epochs, PAWS-1000)
- [x] Reconstruction chrF measured
- [x] Anti-collapse probe measured
- [x] Retrieval R@1 measured
- [x] Blockers: none remaining

## Phase 5 — Dry-run
- [x] Notebook executed non-interactively via `jupyter nbconvert`

## Phase 6 — Ship
- [x] Corrected notebook written in place

## Phase 7 — Verify
- [x] Faithful mode chrF = 94.58 (target ≥ 75)
- [x] Semantic mode chrF = 15.36 (target ≤ faithful − 10)
- [x] Info gap = 79.22
- [x] Retrieval R@1 = 1.000
- [x] Anti-collapse probe: r_acc=0.175 ≤ z_acc=0.175
- [x] Verdict: PASS

## Phase 8 — Log
- [x] `checklist.md`, `summary.md`, `ship-log.tsv` generated
