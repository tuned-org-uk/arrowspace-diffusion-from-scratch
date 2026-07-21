"""Train both Track 1 U-Nets and save checkpoints + loss histories.

Run: .venv/bin/python train_track1.py
Outputs:
  data/checkpoints/unet_spectral_25k.pt
  data/checkpoints/unet_euclidean_25k.pt
  data/checkpoints/losses_spectral_25k.json
  data/checkpoints/losses_euclidean_25k.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, 'src')
from latent_diffusion import (
    load_latent_cache, LatentUNet, latent_tensor_dataset,
    training_loop_latent_spectral, training_loop_latent_euclidean,
)
from spectral_diffusion import ScheduleLogLinear, SpectralGeometry

torch.manual_seed(3407)
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'device: {DEVICE}', flush=True)

DATA = Path('data')
CKPT = DATA / 'checkpoints'
CKPT.mkdir(parents=True, exist_ok=True)

N_STEPS = 25_000
BATCH = 64

# Load
z = load_latent_cache(DATA / 'cifar10_automobile_64_latents.pt')
N, C, H, W = z.shape
F = C * H * W
geom_data = torch.load(DATA / 'latent_geometry_r32_tau0.5.pt', weights_only=False)
geometry = SpectralGeometry(geom_data['Pi'], tau=geom_data['tau']).to(DEVICE)
schedule = ScheduleLogLinear(N=200, sigma_min=0.01, sigma_max=10.0)
loader = DataLoader(latent_tensor_dataset(z), batch_size=BATCH, shuffle=True, drop_last=True)
print(f'latents {tuple(z.shape)}, F={F}, r={geom_data["r"]}, steps={N_STEPS}', flush=True)

def train_one(name, fn, **kw):
    torch.manual_seed(3407)
    model = LatentUNet(latent_channels=4, latent_size=8, block_out_channels=(64,128,256)).to(DEVICE)
    it = fn(loader, model, schedule, latent_shape=(C,H,W),
            epochs=N_STEPS//len(loader)+1, device=DEVICE, lr=2e-4, seed=3407, **kw)
    losses = []
    t0 = time.time()
    for i, ns in enumerate(it):
        losses.append(float(ns['loss']))
        if i % 2500 == 0 and i > 0:
            print(f'  [{name}] step {i:6d}  loss(500-avg) {np.mean(losses[-500:]):.4f}  '
                  f'elapsed {time.time()-t0:.0f}s', flush=True)
        if i >= N_STEPS:
            break
    dt = time.time() - t0
    print(f'  [{name}] DONE: {dt:.0f}s = {dt/60:.1f}min', flush=True)
    torch.save(model.state_dict(), CKPT / f'unet_{name}_25k.pt')
    (CKPT / f'losses_{name}_25k.json').write_text(json.dumps(losses))
    print(f'  [{name}] saved checkpoint + losses', flush=True)
    return model

print('=== spectral ===', flush=True)
train_one('spectral', training_loop_latent_spectral, geometry=geometry)
print('=== euclidean ===', flush=True)
train_one('euclidean', training_loop_latent_euclidean)
print('ALL DONE', flush=True)
