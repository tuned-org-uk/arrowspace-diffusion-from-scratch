# Ship Checklist — autoresearch/ship-260720-1407

**Ship type:** code-pr (auto-detected; uncommitted changes + untracked files)
**Target:** `tuned-org-uk/arrowspace-diffusion-from-scratch` branch `main`
**Trigger:** user "EXECUTE IMMEDIATELY" with ship skill template
**Time:** 2026-07-20 14:07

## Inventory

### Already pushed to `origin/main` (this session)
| commit | file | status |
|--------|------|--------|
| 4ecef19 | `.entire/`, `.opencode/` | shipped |
| a319716 | `pyproject.toml` (latent extras) | shipped |
| 98eca70 | `src/latent_diffusion.py`, `.gitignore` | shipped |
| 614690e | `notebooks/05_latent_dataset_vae.ipynb` (executed) | shipped |
| e5ca3e6 | `notebooks/06_latent_spectral_geometry.ipynb` (executed) | shipped |
| 4ca0b34 | `src/eval_metrics.py`, `notebooks/08_*.ipynb` (skeleton) | shipped |
| 33b1652 | `src/eval_metrics.py` CLIPScore fix | shipped |

### Unshipped (in working tree)
| file | state | issue |
|------|-------|-------|
| `notebooks/07_latent_unet_train.ipynb` | untracked, 0 executed cells | **BLOCKER**: PID 53990 still running nbconvert; writes file only at completion (~50 min remaining) |
| `notebooks/08_latent_sampling_eval.ipynb` | modified, 0 executed cells | **BLOCKER**: skeleton only; never executed end-to-end; would fail (FID ~10 min CPU, VLM may not load) |
| `notebooks/01_spectral_diffusion_from_scratch.ipynb` | modified, stdout-only diff | **WARNING**: pre-existing from before this session, captured training stdout |

### Stale artifacts (gitignored, won't ship)
| path | state |
|------|-------|
| `data/checkpoints/unet_{spectral,euclidean}_50k.pt` | stale 200-step smoke-test ckpts dated 13:44; real 50k ckpts not yet written |
| `data/cifar10_automobile_64_latents.pt` | cached latents from nb05 (gitignored) |
| `data/latent_geometry_r32_tau0.5.pt` | cached geometry from nb06 (gitignored) |

## Code-PR Checklist

| item | result | evidence |
|------|--------|----------|
| Python syntax | PASS | `py_compile` on all 3 src files |
| Module imports | PASS | `import spectral_diffusion, latent_diffusion, eval_metrics` |
| Secrets in diff | PASS | `git diff HEAD \| rg 'token\|secret\|...'` empty |
| Lint | **NOT CONFIGURED** | no ruff/flake8 in pyproject; AGENTS.md doesn't specify |
| Typecheck | **NOT CONFIGURED** | no mypy/pyright in pyproject |
| Tests | **NOT CONFIGURED** | zero test files in repo; no pytest |
| Notebook 07 executed | **FAIL** | 0 executed cells; nbconvert still running |
| Notebook 08 executed | **FAIL** | 0 executed cells; skeleton only |
| Notebook 05 executed | PASS | (already shipped, 614690e) |
| Notebook 06 executed | PASS | (already shipped, e5ca3e6) |
| End-to-end smoke (05->06->07->08) | **FAIL** | 07/08 incomplete |
| PR description | N/A | shipping to `main` directly (no PR) |
| Reviewers assigned | N/A | solo dev |

## Blockers (must-fix before ship)

1. **B1 — Notebook 07 mid-execution.** PID 53990 alive, 20:41 elapsed, ~50 min remaining (100k steps × 42.7ms = 71 min total). The .ipynb file on disk has 0 executed cells because nbconvert writes only at completion. Shipping it now ships an empty notebook.
2. **B2 — Notebook 08 never executed.** Skeleton with a batched-sampler fix never run. FID computation on 256 images takes ~10 min on CPU (MPS lacks float64 InceptionV3); ImagenWorld VLM slice requires Qwen2.5-VL-7B-Instruct which is not installed locally and would fail with a clear fallback message.
3. **B3 — Stale checkpoints in `data/checkpoints/`.** The 44MB `.pt` files dated 13:44 are 200-step smoke-test artifacts. Real 50k checkpoints will only exist once nb07 completes. These are `.gitignored` so won't ship, but if a downstream notebook is executed before nb07 finishes it will load 200-step weights and produce garbage.

## Warnings (can-ship-with)

1. **W1 — No lint/typecheck/test infrastructure.** pyproject.toml has no `[tool.ruff]`, no `[tool.mypy]`, no pytest. Repo has zero test files. AGENTS.md doesn't specify a test command. Suggest adding at minimum `ruff` config and a `pytest` smoke test for `src/spectral_diffusion.py`.
2. **W2 — Notebook 01 uncommitted stdout diff.** Predates this session (from 2026-07-17 v0.2.0 work). Should be either discarded (`git checkout --`) or stripped with `nbstripout` before shipping.
3. **W3 — Track 1 is incomplete.** Notebooks 05/06 are done; 07/08 are not. Shipping partial Track 1 misrepresents the state of issue #2.

## Recommendation

**DO NOT SHIP NOW.** Wait ~50 min for nb07 to finish, then:
1. Verify nb07 executed cleanly (loss curves, sanity samples).
2. Execute nb08 end-to-end (will take ~15-20 min: sampling + FID + CLIPScore; VLM slice will gracefully fall back).
3. Strip notebook outputs with nbstripout (or commit with outputs — your repo convention has been mixed).
4. Commit nb07 + nb08 + nb01 cleanup as a single Track 1 completion commit.
5. Optionally close issue #2 with a comment linking to the commit.

If you insist on shipping now: the only honest commit is "WIP: nb07/nb08 skeletons (training in progress)" — explicitly labelled WIP, not "Track 1 complete".
