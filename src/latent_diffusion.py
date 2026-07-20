"""Latent spectral-geometric diffusion (Track 1, issue #2).

Wraps a frozen diffusers AutoencoderKL for latent space, builds an ArrowSpace
feature-Laplacian over latent dimensions, trains a small U-Net with
metric-matched corruption, and samples in latent space then decodes to pixels.

Convention (cleaner than the tutorial's blend):
- Forward: z_sigma = z0 + sigma * M^{-1/2} eps, eps ~ N(0, I).
- Model predicts the *actual noise added* (whitened): eps_pred = M^{-1/2} eps.
- Single loss = spectral-geometric reconstruction loss:
  L_SG = ((z0_hat - z0)^T M (z0_hat - z0)) / F,  z0_hat = z_sigma - sigma * eps_pred.
- DDIM: z_{t-1} = z_t - (sigma_t - sigma_{t-1}) * eps_pred.
- Initial sample: z_T = M^{-1/2} randn * sigma_max.

This removes the loss_noise / loss_projection inconsistency in the
Swiss-roll training_loop_spectral (where loss_noise trains raw eps but
loss_projection trains whitened eps).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from spectral_diffusion import (
    Schedule,
    SpectralGeometry,
    build_spectral_geometry,
    pairwise,
)


# ---------------------------------------------------------------------------
# Image dataset -> latents
# ---------------------------------------------------------------------------

class _HFClassSubset(Dataset):
    """A torch Dataset wrapping a single-class slice of a HF `datasets` image
    dataset. Images are resized to img_size x img_size and normalised to
    [-1, 1] in CHW float layout."""

    def __init__(self, hf_split, indices, img_size: int = 64):
        self.hf_split = hf_split
        self.indices = indices
        self.img_size = img_size
        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        row = self.hf_split[int(self.indices[i])]
        return self.transform(row["img"])


def make_cifar10_class_subset(
    root: str,
    class_name: str,
    img_size: int = 64,
    max_n: Optional[int] = None,
    cache_dir: str = "./data/hf_cache",
) -> Tuple[Dataset, int]:
    """Single-class CIFAR-10 subset resized to img_size x img_size, normalised
    to [-1, 1] in CHW layout. Downloaded from HF Hub (`uoft-cs/cifar10`) — much
    faster than torchvision's mirror and works without a token for this public
    dataset."""
    from datasets import load_dataset
    hf = load_dataset("uoft-cs/cifar10", split="train", cache_dir=cache_dir)
    names = hf.features["label"].names
    class_idx = names.index(class_name) if isinstance(class_name, str) else int(class_name)
    indices = [i for i, lbl in enumerate(hf["label"]) if lbl == class_idx]
    if max_n is not None:
        indices = indices[:max_n]
    return _HFClassSubset(hf, indices, img_size=img_size), len(indices)


@torch.no_grad()
def encode_dataset_to_latents(
    vae: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Encode all images in loader through frozen VAE. Returns (N, C, H, W) latents
    already scaled by vae.config.scaling_factor."""
    z_list = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device)
        posterior = vae.encode(batch).latent_dist
        z = posterior.sample() * vae.config.scaling_factor
        z_list.append(z.cpu())
    return torch.cat(z_list, dim=0)


def latent_tensor_dataset(z: torch.Tensor) -> TensorDataset:
    return TensorDataset(z)


# ---------------------------------------------------------------------------
# Spectral geometry over latent features
# ---------------------------------------------------------------------------

def build_latent_spectral_geometry(
    z: torch.Tensor,
    k: int = 8,
    tau: float = 0.5,
    r: Optional[int] = None,
) -> SpectralGeometry:
    """Build feature-Laplacian over the F = C*H*W latent dimensions.

    z: (N, C, H, W) latents. Each *latent coordinate* (one of C*H*W) becomes a
    node in the feature graph; cosine similarity is computed across the N items.
    """
    N, C, H, W = z.shape
    X = z.flatten(start_dim=1)  # (N, F)
    return build_spectral_geometry(X, k=k, tau=tau, r=r)


# ---------------------------------------------------------------------------
# U-Net wrapper (diffusers UNet2DModel, random init)
# ---------------------------------------------------------------------------

class LatentUNet(nn.Module):
    """Wraps a diffusers UNet2DModel to expose the tutorial's (x, sigma) -> eps
    interface. Sigma (a float per sample) is passed as the diffusion timestep;
    diffusers' internal timestep embedding handles it."""

    def __init__(
        self,
        latent_channels: int = 4,
        latent_size: int = 8,
        block_out_channels: Tuple[int, ...] = (64, 128, 256),
        layers_per_block: int = 1,
    ):
        super().__init__()
        from diffusers import UNet2DModel

        n_levels = len(block_out_channels)
        down_block_types = ("DownBlock2D",) + ("AttnDownBlock2D",) * (n_levels - 1)
        up_block_types = ("AttnUpBlock2D",) * (n_levels - 1) + ("UpBlock2D",)
        self.net = UNet2DModel(
            sample_size=latent_size,
            in_channels=latent_channels,
            out_channels=latent_channels,
            layers_per_block=layers_per_block,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
        )
        self.latent_channels = latent_channels
        self.latent_size = latent_size

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.net(x, timestep=sigma).sample

    def rand_input(self, batchsize: int, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.randn(
            (batchsize, self.latent_channels, self.latent_size, self.latent_size),
            device=device,
        )


# ---------------------------------------------------------------------------
# Training loops (latent space)
# ---------------------------------------------------------------------------

def training_loop_latent_euclidean(
    loader: DataLoader,
    model: nn.Module,
    schedule: Schedule,
    latent_shape: Tuple[int, int, int],
    epochs: int = 10000,
    device: torch.device = torch.device("cpu"),
    lr: float = 2e-4,
    seed: int = 3407,
) -> Iterator[dict]:
    """Baseline isotropic-noise latent diffusion (Euclidean)."""
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    C, H, W = latent_shape
    for epoch in range(epochs):
        for z0 in loader:
            z0 = z0[0].to(device) if isinstance(z0, (list, tuple)) else z0.to(device)
            opt.zero_grad()
            sigma = schedule.sample_batch(z0)  # (B,)
            eps = torch.randn_like(z0)
            z_sigma = z0 + sigma[:, None, None, None] * eps
            eps_hat = model(z_sigma, sigma)
            loss = F.mse_loss(eps_hat, eps)
            loss.backward()
            opt.step()
            yield {"epoch": epoch, "loss": loss.item()}


def training_loop_latent_spectral(
    loader: DataLoader,
    model: nn.Module,
    schedule: Schedule,
    geometry: SpectralGeometry,
    latent_shape: Tuple[int, int, int],
    epochs: int = 10000,
    device: torch.device = torch.device("cpu"),
    lr: float = 2e-4,
    seed: int = 3407,
) -> Iterator[dict]:
    """Metric-matched latent spectral-geometric diffusion.

    Single consistent loss: spectral-geometric reconstruction loss
    L_SG = ((z0_hat - z0)^T M (z0_hat - z0)) / F.
    The model predicts the actual noise added (whitened): eps_pred = M^{-1/2} eps.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    geometry = geometry.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    C, H, W = latent_shape
    F = C * H * W
    M = geometry.M  # (F, F), symmetric
    for epoch in range(epochs):
        for z0 in loader:
            z0 = z0[0].to(device) if isinstance(z0, (list, tuple)) else z0.to(device)
            B = z0.shape[0]
            z0_flat = z0.flatten(start_dim=1)  # (B, F)
            opt.zero_grad()
            sigma = schedule.sample_batch(z0_flat)  # (B,)
            z_sigma_flat, _ = geometry.corrupt(z0_flat, sigma)
            z_sigma = z_sigma_flat.reshape(B, C, H, W)
            eps_pred_flat = model(z_sigma, sigma).flatten(start_dim=1)  # (B, F)
            z0_hat = z_sigma_flat - sigma[:, None] * eps_pred_flat
            diff = z0_hat - z0_flat  # (B, F)
            # L_SG = (diff^T M diff) / F, averaged over batch
            diff_M = diff @ M.T  # M symmetric so M.T == M
            loss = (diff * diff_M).sum(dim=1).mean() / F
            loss.backward()
            opt.step()
            yield {"epoch": epoch, "loss": loss.item()}


# ---------------------------------------------------------------------------
# Sampling (latent space + VAE decode)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_latent_euclidean(
    model: nn.Module,
    sigmas: torch.Tensor,
    latent_shape: Tuple[int, int, int],
    batchsize: int = 64,
    device: torch.device = torch.device("cpu"),
    vae: Optional[nn.Module] = None,
) -> torch.Tensor:
    """DDIM sampler for the Euclidean latent model. Returns latents (or decoded
    images if vae is provided)."""
    model = model.to(device)
    C, H, W = latent_shape
    x = model.rand_input(batchsize, device=device) * sigmas[0]
    for sig, sig_prev in pairwise(sigmas):
        sig_b = sig.to(x).expand(x.shape[0])
        eps = model(x, sig_b)
        x = x - (sig - sig_prev) * eps
    if vae is not None:
        vae = vae.to(device).eval()
        return vae.decode(x / vae.config.scaling_factor).sample
    return x


@torch.no_grad()
def sample_latent_spectral(
    model: nn.Module,
    sigmas: torch.Tensor,
    geometry: SpectralGeometry,
    latent_shape: Tuple[int, int, int],
    batchsize: int = 64,
    device: torch.device = torch.device("cpu"),
    vae: Optional[nn.Module] = None,
) -> torch.Tensor:
    """DDIM sampler for the metric-matched latent spectral model. The model
    predicts the whitened noise (actual noise added), so the DDIM update uses
    eps_pred directly. Initial sample is M^{-1/2} z * sigma_max.

    Returns latents (B, C, H, W) or decoded images (B, 3, H_img, W_img) if vae
    is provided."""
    model = model.to(device)
    geometry = geometry.to(device)
    C, H, W = latent_shape
    F = C * H * W
    # Initial noise matches the metric covariance.
    z = torch.randn(batchsize, F, device=device)
    x_flat = z @ geometry.M_inv_sqrt.T * sigmas[0]
    for sig, sig_prev in pairwise(sigmas):
        sig_b = sig.to(x_flat).expand(x_flat.shape[0])
        x_4d = x_flat.reshape(batchsize, C, H, W)
        eps_pred_flat = model(x_4d, sig_b).flatten(start_dim=1)
        x_flat = x_flat - (sig - sig_prev) * eps_pred_flat
    z_final = x_flat.reshape(batchsize, C, H, W)
    if vae is not None:
        vae = vae.to(device).eval()
        return vae.decode(z_final / vae.config.scaling_factor).sample
    return z_final


# ---------------------------------------------------------------------------
# Latent cache (so we don't re-encode through the VAE every run)
# ---------------------------------------------------------------------------

def save_latent_cache(z: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": z, "shape": tuple(z.shape)}, path)


def load_latent_cache(path: str | Path) -> torch.Tensor:
    obj = torch.load(Path(path), map_location="cpu")
    return obj["latents"]


# ---------------------------------------------------------------------------
# Convenience: build everything for a latent run
# ---------------------------------------------------------------------------

def build_latent_pipeline(
    vae: nn.Module,
    image_dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
    cache_path: Optional[str | Path] = None,
) -> torch.Tensor:
    """Encode an image dataset to latents, with optional on-disk cache."""
    if cache_path is not None and Path(cache_path).exists():
        return load_latent_cache(cache_path)
    loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    z = encode_dataset_to_latents(vae, loader, device)
    if cache_path is not None:
        save_latent_cache(z, cache_path)
    return z
