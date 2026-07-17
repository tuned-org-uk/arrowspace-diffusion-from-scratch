# Spectral-geometric diffusion from scratch

A from-scratch implementation of diffusion models with the ArrowSpace spectral-geometric metric, extending the Euclidean-diffusion tutorial at [chenyang.co/diffusion.html](https://chenyang.co/diffusion.html).

The theoretical background is in the blog post [Diffusion as spectral-geometric projection](https://www.tuned.org.uk/posts/021_diffusion_as_spectral_geometric_projection/).

## Key idea

Standard diffusion uses isotropic Gaussian corruption and Euclidean distance. ArrowSpace spectral diffusion replaces the metric with

$$M_{0.5} = \tfrac12(I + \Pi_F)$$

where $\Pi_F$ is the low-frequency projector from the feature-space Laplacian. The forward corruption uses covariance $\sigma^2 M^{-1}$ so the Bayes-optimal denoiser actually respects the hybrid geometry.

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | [01_spectral_diffusion_from_scratch.ipynb](notebooks/01_spectral_diffusion_from_scratch.ipynb) | Baseline Euclidean vs spectral-geometric diffusion on the 2-D Swiss roll. Includes metric-matched noise justification. |
| 02 | [02_spectral_diffusion_theory.ipynb](notebooks/02_spectral_diffusion_theory.ipynb) | Smoothed distance contours, relative error model, and sampling trajectories with varying $\gamma/\mu$ on a $D=128$ multi-cluster dataset. |
| 03 | [03_spiral_manifold.ipynb](notebooks/03_spiral_manifold.ipynb) | Spiral (Swiss roll) manifold in $D=128$ — the canonical 1-D intrinsic / high ambient case. Spiral fidelity metric across sampler configurations. |

## Setup

```bash
uv pip install --python .venv/bin/python torch numpy matplotlib scipy jupyterlab ipykernel nbformat
.venv/bin/python -m ipykernel install --user --name arrowspace-diffusion --display-name "Python (arrowspace-diffusion)"
```

## Run

```bash
.venv/bin/jupyter lab
```

## Structure

```
src/
  spectral_diffusion.py   # Core library: schedules, datasets, models, training loops, samplers
notebooks/
  01_spectral_diffusion_from_scratch.ipynb
  02_spectral_diffusion_theory.ipynb
  03_spiral_manifold.ipynb
```

## References

- [Chenyang Yuan — Diffusion models from scratch](https://chenyang.co/diffusion.html) (ICML 2024)
- [Blog post 021 — Diffusion as spectral-geometric projection](https://www.tuned.org.uk/posts/021_diffusion_as_spectral_geometric_projection/)
- [ArrowSpace — Spectral Search for Embeddings](https://doi.org/10.21105/joss.09002)
- [Sohl-Dickstein et al. 2015 — Deep Unsupervised Learning](https://arxiv.org/abs/1503.03578)
