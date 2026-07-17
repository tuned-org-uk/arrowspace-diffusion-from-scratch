"""Spectral-geometric diffusion from scratch.

This module mirrors the Euclidean-diffusion tutorial at
https://chenyang.co/diffusion.html, then extends it with an ArrowSpace-style
feature-manifold metric lambda^tau = tau*I + (1-tau)*Pi.

Key design choice (from blog post 021): the forward corruption must use the
metric's inverse covariance, x_sigma = x_0 + sigma * M^{-1/2} eps, so that the
Bayes-optimal denoiser actually respects the spectral-geometric geometry.
"""
from __future__ import annotations

import math
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

class Schedule:
    """Discrete list of noise levels sigma."""

    def __init__(self, sigmas: torch.Tensor):
        self.sigmas = sigmas

    def __getitem__(self, i) -> torch.Tensor:
        return self.sigmas[i]

    def __len__(self) -> int:
        return len(self.sigmas)

    def sample_batch(self, x0: torch.Tensor) -> torch.Tensor:
        idx = torch.randint(len(self), (x0.shape[0],))
        return self[idx].to(x0)

    def sample_sigmas(self, steps: int) -> torch.Tensor:
        """Subsampling used by deterministic samplers (DDIM-style)."""
        indices = (len(self) * (1 - np.arange(steps) / steps)).round().astype(np.int64) - 1
        indices = np.clip(indices, 0, len(self) - 1)
        return self[torch.from_numpy(indices).long()]


class ScheduleLogLinear(Schedule):
    def __init__(self, N: int, sigma_min: float = 0.02, sigma_max: float = 10.0):
        super().__init__(torch.logspace(math.log10(sigma_min), math.log10(sigma_max), N))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class Swissroll(Dataset):
    """2-D spiral dataset from Sohl-Dickstein et al. 2015."""

    def __init__(self, start: float, end: float, n: int, noise: float = 0.4):
        theta = torch.linspace(start, end, n)
        x = theta * torch.cos(theta)
        y = theta * torch.sin(theta)
        self.data = torch.stack([x, y], dim=1)
        if noise > 0:
            self.data += noise * torch.randn_like(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def get_sigma_embeds(sigma: torch.Tensor) -> torch.Tensor:
    """Two-dimensional log-frequency embedding used in the tutorial."""
    sigma = sigma.unsqueeze(1)
    return torch.cat([torch.sin(torch.log(sigma) / 2),
                      torch.cos(torch.log(sigma) / 2)], dim=1)


class TimeInputMLP(nn.Module):
    """MLP that concatenates x with an embedding of sigma."""

    def __init__(self, dim: int, hidden_dims: Tuple[int, ...] = (16, 128, 128, 128, 128, 16)):
        super().__init__()
        dims = (dim + 2,) + hidden_dims + (dim,)
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != dim:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.input_dims = (dim,)

    def rand_input(self, batchsize: int, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.randn((batchsize,) + self.input_dims, device=device)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_embeds = get_sigma_embeds(sigma)
        nn_input = torch.cat([x, sigma_embeds], dim=1)
        return self.net(nn_input)


# ---------------------------------------------------------------------------
# Spectral geometry (ArrowSpace feature-manifold metric)
# ---------------------------------------------------------------------------

class SpectralGeometry:
    """Implements lambda^tau = tau*I + (1-tau)*Pi and metric-matched noise."""

    def __init__(self, projector: torch.Tensor, tau: float = 0.5):
        """Args:
            projector: orthogonal projection matrix Pi (F x F).
            tau: blend weight between Euclidean and spectral geometry.
        """
        self.Pi = projector
        self.tau = tau
        self.F = projector.shape[0]
        self.M = tau * torch.eye(self.F) + (1.0 - tau) * projector
        self.M_inv = torch.linalg.inv(self.M)
        self.M_inv_sqrt = self._matrix_inv_sqrt(self.M)

    @staticmethod
    def _matrix_inv_sqrt(M: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(M)
        # Guard against tiny negative eigenvalues from numerical error.
        eigvals = eigvals.clamp(min=1e-8)
        return eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Pi.T

    def squared_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        geometric = ((x - y) ** 2).sum(dim=-1)
        spectral = ((self.project(x) - self.project(y)) ** 2).sum(dim=-1)
        return self.tau * geometric + (1.0 - self.tau) * spectral

    def corrupt(self, x0: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Metric-matched corruption: x_sigma = x_0 + sigma * M^{-1/2} eps."""
        eps = torch.randn_like(x0)
        # sigma has shape (B,); broadcast along features.
        x_sigma = x0 + sigma[:, None] * (eps @ self.M_inv_sqrt.T)
        return x_sigma, eps

    def reconstruction_loss(
        self, x_hat: torch.Tensor, x_true: torch.Tensor
    ) -> torch.Tensor:
        geometric = ((x_hat - x_true) ** 2).mean()
        spectral = ((self.project(x_hat) - self.project(x_true)) ** 2).mean()
        return self.tau * geometric + (1.0 - self.tau) * spectral

    def to(self, device: torch.device) -> "SpectralGeometry":
        self.Pi = self.Pi.to(device)
        self.M = self.M.to(device)
        self.M_inv = self.M_inv.to(device)
        self.M_inv_sqrt = self.M_inv_sqrt.to(device)
        return self


# ---------------------------------------------------------------------------
# Feature-space Laplacian construction
# ---------------------------------------------------------------------------

def build_feature_laplacian(
    X: torch.Tensor, k: int = 8, symmetric: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (unnormalised) feature-space Laplacian from data matrix X (N x F).

    Features are columns of X.  The kNN graph is built on cosine similarity
    between feature signals (columns), exactly as in ArrowSpace's feature
    Laplacian path.
    """
    X_np = X.detach().cpu().numpy()
    N, F = X_np.shape
    X_feat = X_np.T  # (F, N): each row is one feature signal
    # Normalise for cosine similarity.
    X_feat_norm = X_feat / (np.linalg.norm(X_feat, axis=1, keepdims=True) + 1e-12)
    S = X_feat_norm @ X_feat_norm.T  # (F, F)
    W = np.zeros_like(S)
    for i in range(F):
        nbrs = np.argsort(-S[i])[1 : k + 1]
        W[i, nbrs] = S[i, nbrs]
    if symmetric:
        W = np.maximum(W, W.T)
    D = np.diag(W.sum(axis=1))
    L = D - W
    return torch.from_numpy(L).float(), torch.from_numpy(W).float()


def build_projector_from_laplacian(
    L: torch.Tensor, r: Optional[int] = None, drop_constant: bool = True
) -> torch.Tensor:
    """Return low-frequency orthogonal projector Pi = U_r U_r^T.

    Args:
        L: F x F symmetric Laplacian.
        r: number of eigenvectors to keep.  If None, uses the spectral gap.
        drop_constant: exclude the trivial constant eigenvector if present.
    """
    eigvals, eigvecs = torch.linalg.eigh(L)
    # Sort ascending (eigh already does, but be explicit).
    idx = torch.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    start = 1 if drop_constant else 0

    if r is None:
        # Pick r by largest relative spectral gap among first 1/3 of modes.
        vals = eigvals[start:].numpy()
        gaps = vals[1:] - vals[:-1]
        if len(gaps) > 0:
            r_local = int(np.argmax(gaps[: max(1, len(gaps) // 3)]) + 1)
        else:
            r_local = 1
    else:
        r_local = r

    r_local = min(max(r_local, 1), eigvecs.shape[1] - start)
    U_r = eigvecs[:, start : start + r_local]
    Pi = U_r @ U_r.T
    return Pi


def build_spectral_geometry(
    X: torch.Tensor, k: int = 8, tau: float = 0.5, r: Optional[int] = None
) -> SpectralGeometry:
    """Convenience: build feature Laplacian and return SpectralGeometry."""
    L, _ = build_feature_laplacian(X, k=k)
    Pi = build_projector_from_laplacian(L, r=r)
    return SpectralGeometry(Pi, tau=tau)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def training_loop_euclidean(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    schedule: Schedule,
    epochs: int = 10000,
    device: torch.device = torch.device("cpu"),
) -> Iterator[dict]:
    """Baseline Euclidean training loop from the tutorial."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for x0 in loader:
            x0 = x0.to(device)
            optimizer.zero_grad()
            sigma = schedule.sample_batch(x0)
            eps = torch.randn_like(x0)
            x_sigma = x0 + sigma[:, None] * eps
            eps_hat = model(x_sigma, sigma)
            loss = nn.MSELoss()(eps_hat, eps)
            loss.backward()
            optimizer.step()
            yield {"epoch": epoch, "loss": loss.item()}


def training_loop_spectral(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    schedule: Schedule,
    geometry: SpectralGeometry,
    epochs: int = 10000,
    device: torch.device = torch.device("cpu"),
    loss_weight: float = 0.5,
) -> Iterator[dict]:
    """ArrowSpace metric-matched training loop."""
    model = model.to(device)
    geometry = geometry.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        for x0 in loader:
            x0 = x0.to(device)
            optimizer.zero_grad()
            sigma = schedule.sample_batch(x0)
            x_sigma, eps = geometry.corrupt(x0, sigma)
            eps_hat = model(x_sigma, sigma)
            x_hat = x_sigma - sigma[:, None] * eps_hat

            loss_noise = nn.MSELoss()(eps_hat, eps)
            loss_projection = geometry.reconstruction_loss(x_hat, x0)
            loss = loss_weight * loss_noise + (1.0 - loss_weight) * loss_projection

            loss.backward()
            optimizer.step()
            yield {
                "epoch": epoch,
                "loss": loss.item(),
                "loss_noise": loss_noise.item(),
                "loss_projection": loss_projection.item(),
            }


# ---------------------------------------------------------------------------
# Ideal denoisers (for small datasets)
# ---------------------------------------------------------------------------

class IdealDenoiser:
    """Closed-form Euclidean denoiser over a finite dataset."""

    def __init__(self, dataset: Dataset, device: torch.device = torch.device("cpu")):
        self.data = torch.stack(list(dataset)).to(device)
        self.data_flat = self.data.flatten(start_dim=1)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        d = self.data_flat
        xb, db = x.shape[0], d.shape[0]
        sq_diffs = (
            (x ** 2).sum(dim=1, keepdim=True).expand(xb, db)
            + (d ** 2).sum(dim=1, keepdim=True).T.expand(xb, db)
            - 2 * x @ d.T
        )
        weights = torch.nn.functional.softmax(-sq_diffs / (2 * sigma[:, None] ** 2), dim=1)
        return (x - torch.einsum("ij,j...->i...", weights, self.data)) / sigma[:, None]


class IdealSpectralDenoiser:
    """Closed-form spectral-geometric denoiser over a finite dataset."""

    def __init__(
        self,
        dataset: Dataset,
        geometry: SpectralGeometry,
        device: torch.device = torch.device("cpu"),
    ):
        self.data = torch.stack(list(dataset)).to(device)
        self.geometry = geometry.to(device)

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Use the hybrid squared distance d_tau^2(x, x0).
        # weights[i, j] proportional to exp(-d_tau^2(x_i, x0_j) / (2 sigma_i^2))
        dist2 = self.geometry.squared_distance(
            x.unsqueeze(1), self.data.unsqueeze(0)
        )  # (B, N)
        weights = torch.nn.functional.softmax(-dist2 / (2 * sigma[:, None] ** 2), dim=1)
        x_bar = torch.einsum("ij,j...->i...", weights, self.data)
        # The model predicts M^{1/2} eps, not raw eps; convert displacement.
        disp = x - x_bar  # (B, D)
        disp_m = disp @ self.geometry.M_inv_sqrt.T  # (B, D)
        return disp_m / sigma[:, None]  # (B, D)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    it = iter(iterable)
    try:
        prev = next(it)
    except StopIteration:
        return
    for curr in it:
        yield prev, curr
        prev = curr


@torch.no_grad()
def sample_euclidean(
    model: nn.Module,
    sigmas: torch.Tensor,
    batchsize: int = 2000,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """DDIM-style deterministic sampler for Euclidean model."""
    model = model.to(device)
    x = model.rand_input(batchsize, device=device) * sigmas[0]
    for sig, sig_prev in pairwise(sigmas):
        eps = model(x, sig.to(x))
        x = x - (sig - sig_prev) * eps
    return x


@torch.no_grad()
def sample_spectral(
    model: nn.Module,
    sigmas: torch.Tensor,
    geometry: SpectralGeometry,
    batchsize: int = 2000,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """DDIM-style deterministic sampler for metric-matched model.

    The model predicts M^{1/2} eps.  To step in raw coordinates we multiply by
    M^{-1/2} before applying the update.
    """
    model = model.to(device)
    geometry = geometry.to(device)
    # Initial noise must match the metric covariance.
    z = torch.randn(batchsize, geometry.F, device=device)
    x = z @ geometry.M_inv_sqrt.T * sigmas[0]
    for sig, sig_prev in pairwise(sigmas):
        eps_M = model(x, sig.to(x).expand(x.shape[0]))  # model predicts M^{1/2} eps
        eps = eps_M @ geometry.M_inv_sqrt.T
        x = x - (sig - sig_prev) * eps
    return x


# ---------------------------------------------------------------------------
# Momentum / gradient-estimation sampler (from tutorial)
# ---------------------------------------------------------------------------

@torch.no_grad()
def samples_with_momentum(
    model: nn.Module,
    sigmas: torch.Tensor,
    gam: float = 1.0,
    mu: float = 0.0,
    batchsize: int = 1,
    device: torch.device = torch.device("cpu"),
) -> Iterator[torch.Tensor]:
    """Generalised sampler (DDIM: gam=1, mu=0; DDPM: gam=1, mu=0.5)."""
    model = model.to(device)
    xt = model.rand_input(batchsize, device=device) * sigmas[0]
    eps_prev = None
    for i, (sig, sig_prev) in enumerate(pairwise(sigmas)):
        eps = model(xt, sig.to(xt).expand(batchsize))
        if i == 0 or eps_prev is None:
            eps_av = eps
        else:
            eps_av = gam * eps + (1 - gam) * eps_prev
        if mu == 0.0:
            sig_p = sig_prev
            eta = 0.0
        else:
            sig_p = (sig_prev / (sig ** mu)) ** (1.0 / (1.0 - mu))
            eta = math.sqrt(max(sig_prev ** 2 - sig_p ** 2, 0.0))
        xt = xt - (sig - sig_p) * eps_av + eta * model.rand_input(batchsize, device=device).to(xt)
        eps_prev = eps
        yield xt
