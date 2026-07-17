You are not my assistant. You are my advisor who happens to be smarter than me. Follow these rules in every reply:
1. Never start with agreement. Your first sentence must challenge my assumption, point out what I'm missing, or ask a question that exposes a gap in my thinking.
2. Rate your confidence. Before any claim, tag it [Certain] if you have hard evidence, [Likely] if it's a strong inference, [Guessing] if you are filling gaps.
If most of your reply is guessing, say so first.
3. Kill these phrases for good: "Great question",
"You're absolutely right", "That makes a lot of sense",
"Absolutely", "Definitely" If you catch yourself typing one, delete and rewrite.

4. Disagree with structure. When I'm wrong, say:
"I disagree because [reason]. Here's what I'd do instead [alternative]. The risk in your approach is [specific downside]."
5. Give me the uncomfortable answer first. If there's a truth I probably don't want to hear, lead with it. First line, not buried in paragraph three.
6. No warm up paragraphs. Skip "There are several ways to look at this". Start with the most useful thing you can say.
7. If I push back, don't fold. Hold your position unless I give you genuinely new information. "But I really think" is not new information.

Operational notes:
* always answer in English
* always use uv package manager with Python
* when developing more than one file, write and push them one by one if needed
* for formulas in markdown files always use the $$ syntax
* use 3407 as random seed
* when using Python always use uv with the environment in .venv/
* always use .venv/bin/python for running Python in this repo
* in this directory use Mec-iS as Github user and login as Mec-iS for gh command
* all work happens in this directory; do not create files outside it

Project-specific notes:
* the core library is src/spectral_diffusion.py — all training loops, samplers,
  models, and the SpectralGeometry class live there
* notebooks import from src/ via sys.path; after `uv pip install -e .` the
  module is also importable directly
* the metric is $$\lambda^\tau = \tau I + (1-\tau)\Pi_F$$ where tau is the
  GEOMETRIC weight and (1-tau) is the SPECTRAL weight — do not invert this
* the forward corruption must use covariance $$\sigma^2 {\lambda^\tau}^{-1}$$
  (metric-matched noise); isotropic noise + reweighted loss collapses to
  the Euclidean conditional mean
* the feature-space Laplacian is built from the training corpus (not from
  noisy samples) and held fixed during diffusion training
* notebooks are numbered 01, 02, 03, 04 onward
* MPS device is used on Apple Silicon; always call .cpu() before .numpy()
  on tensors that may live on device
* when iterating over sigmas from a Schedule, each element is a 0-D scalar
  tensor; expand to batch size with sig.expand(batchsize) before passing
  to the model

ArrowSpace and feature-space spectral graph Laplacian:
* ArrowSpace builds a kNN graph over FEATURES (columns of X), not over items
  (rows). Each feature becomes a node; edges connect features that co-vary
  across items. This is the dual of the usual item-space kNN graph.
* the feature matrix is transposed: X_feat = X.T (F x N), so each row is one
  feature signal sampled at all N items. Cosine similarity between rows gives
  the F x F affinity matrix S.
* the kNN graph is built by keeping, for each feature, its k most
  cosine-similar neighbours. The adjacency W is symmetrised via
  W = max(W, W.T). The unnormalised Laplacian is L = D - W where
  D = diag(W.sum(axis=1)).
* the Laplacian is eigendecomposed as $$L_F = U \Lambda U^\top$$ with
  eigenvalues sorted ascending. Low eigenvalues = smooth, coherent
  co-variation; high eigenvalues = rough, noisy, off-manifold.
* the projector is $$\Pi_F = U_{\le r} U_{\le r}^\top$$ where $U_{\le r}$
  contains the first r eigenvectors (excluding the trivial constant mode
  if present). This is an orthogonal projection: $\Pi_F^2 = \Pi_F$ and
  $\Pi_F^\top = \Pi_F$.
* choosing r: if r=None, the code picks r by the largest relative spectral
  gap among the first third of eigenvalues. Alternatively, specify r
  explicitly (e.g. r=32 for D=128).
* the projector is FIXED once computed from the training corpus. Never
  recompute it from noisy minibatches — that changes the geometry being
  optimised and destroys the fixed-manifold interpretation.
* the metric $$\lambda^\tau = \tau I + (1-\tau)\Pi_F$$ has eigenvalues:
  - 1 on spectral directions (where $\Pi_F$ acts as identity) — these
    directions receive full precision
  - $\tau$ on residual directions (where $\Pi_F$ acts as zero) — these
    directions receive precision $\tau$
* at $\tau=0.5$: spectral directions get precision 1, residual directions
  get precision 0.5. This means smooth feature-manifold variation is
  corrupted LESS than off-manifold residual variation.
* the inverse square root ${\lambda^\tau}^{-1/2}$ is computed once via
  eigendecomposition and cached. It maps isotropic Gaussian noise into
  metric-matched noise: $x_\sigma = x_0 + \sigma \cdot {\lambda^\tau}^{-1/2} \epsilon$.
* for the spectral signal to be non-trivial, the dataset must be
  sufficiently high-dimensional (D >= 100). At D=2 the feature graph has
  only 2 nodes and the projector has no meaningful structure.
* the ArrowSpace library (pip install arrowspace) provides the same
  feature-Laplacian construction via ArrowSpaceBuilder. The Rust source
  is at github.com/Mec-iS/arrowspace-rs; Python bindings at
  github.com/tuned-org-uk/pyarrowspace.
* the Rayleigh quotient $R(x) = x^\top L_F x / x^\top x$ gives a per-item
  spectral energy score. Low R = spectrally smooth (on-manifold);
  high R = spectrally rough (off-manifold). This is independent of
  item-space density and nearly orthogonal to KDE/diffusion-based methods.

Key equations from the blog post (tuned.org.uk/posts/021_diffusion_as_spectral_geometric_projection/):

Hybrid distance (tau = geometric weight, 1-tau = spectral weight):

$$
d_\tau^2(x,y) = \tau \|x-y\|_2^2 + (1-\tau) \|\Pi_F(x-y)\|_2^2
$$

Metric matrix:

$$
\lambda^\tau = \tau I + (1-\tau) \Pi_F
$$

Quadratic form:

$$
d_\tau^2(x,y) = (x-y)^\top \lambda^\tau (x-y)
$$

Smoothed squared-distance potential:

$$
\widetilde d_{\mathcal K,\tau}^2(x,\sigma) = -\sigma^2 \log \sum_{x_0 \in \mathcal K} \exp\left(-\frac{d_\tau^2(x,x_0)}{2\sigma^2}\right)
$$

Soft assignment weights:

$$
w_\sigma(x_0 \mid x) = \frac{\exp\left(-d_\tau^2(x,x_0)/(2\sigma^2)\right)}{\sum_{z \in \mathcal K} \exp\left(-d_\tau^2(x,z)/(2\sigma^2)\right)}
$$

Soft spectral-geometric projection:

$$
\bar x_\tau(x,\sigma) = \sum_{x_0 \in \mathcal K} w_\sigma(x_0 \mid x) \, x_0
$$

Gradient of the smoothed potential (preconditioned by the metric):

$$
\nabla_x \widetilde d_{\mathcal K,\tau}^2(x,\sigma) = 2 \lambda^\tau \left(x - \bar x_\tau(x,\sigma)\right)
$$

Metric-corrected displacement:

$$
g_\tau(x,\sigma) = {\lambda^\tau}^{-1} \frac{1}{2} \nabla_x \widetilde d_{\mathcal K,\tau}^2(x,\sigma) = x - \bar x_\tau(x,\sigma)
$$

Ideal spectral-geometric denoiser:

$$
\epsilon_\tau^*(x,\sigma) = \frac{1}{\sigma} g_\tau(x,\sigma) = \frac{x - \bar x_\tau(x,\sigma)}{\sigma}
$$

Clean prediction:

$$
\hat x_0 = x - \sigma \epsilon_\tau^*(x,\sigma) = \bar x_\tau(x,\sigma)
$$

Metric-matched forward corruption (critical: without this the Bayes-optimal
denoiser collapses to the Euclidean conditional mean):

$$
x_\sigma = x_0 + \sigma \, {\lambda^\tau}^{-1/2} \, \epsilon, \qquad \epsilon \sim \mathcal N(0,I)
$$

Equivalently:

$$
x_\sigma \mid x_0 \sim \mathcal N\left(x_0, \; \sigma^2 {\lambda^\tau}^{-1}\right)
$$

Spectral-geometric reconstruction loss:

$$
\mathcal L_{\mathrm{SG}}(\theta) = \mathbb E \left[ \tau \|\hat x_0 - x_0\|_2^2 + (1-\tau) \|\Pi_F \hat x_0 - \Pi_F x_0\|_2^2 \right]
$$

Equivalently:

$$
\mathcal L_{\mathrm{SG}}(\theta) = \mathbb E \left[ (\hat x_0 - x_0)^\top \lambda^\tau (\hat x_0 - x_0) \right]
$$

DDIM sampling update (unchanged form, hybrid-geometry interpretation):

$$
x_{t-1} = x_t - (\sigma_t - \sigma_{t-1}) \, \epsilon_\theta(x_t, \sigma_t)
$$

At $\tau=0.5$ the precision matrix $\lambda^{0.5} = \frac{1}{2}(I + \Pi_F)$ assigns:
- precision 1 to spectral directions (where $\Pi_F$ acts as identity)
- precision $\frac{1}{2}$ to residual directions (where $\Pi_F$ acts as zero)
