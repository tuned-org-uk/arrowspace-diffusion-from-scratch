"""Evaluation metrics for Track 1 (issue #2).

- FID via torchmetrics (CPU — MPS lacks float64 for InceptionV3).
- CLIPScore via torchmetrics (uses openai/clip-vit-base-patch32).
- ImagenWorld TIG VLM-as-judge slice (arXiv:2603.27862): rates generated
  images on the paper's four criteria (Prompt Relevance, Aesthetic Quality,
  Content Coherence, Artifact) using a local open VLM. Falls back with a
  clear message if no VLM is available locally.

Honest framing: our model is unconditional, so the "prompt" for TIG is the
class label (e.g. "a photo of an automobile"). This is a degenerate case of
the paper's TIG task (which uses natural-language instructions). We use the
ImagenWorld rubric as a structured sanity eval, NOT as a leaderboard claim.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(
    real_images: torch.Tensor,  # (N, 3, H, W) in [-1, 1]
    gen_images: torch.Tensor,   # (M, 3, H, W) in [-1, 1]
    batch_size: int = 64,
    feature: int = 2048,
) -> float:
    """Frechet Inception Distance. Runs on CPU (MPS lacks float64 for InceptionV3)."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=feature, normalize=True).to('cpu')
    real_cpu = real_images.detach().cpu()
    gen_cpu = gen_images.detach().cpu()

    real_loader = DataLoader(TensorDataset(real_cpu), batch_size=batch_size, shuffle=False)
    gen_loader = DataLoader(TensorDataset(gen_cpu), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (b,) in real_loader:
            fid.update(b, real=True)
        for (b,) in gen_loader:
            fid.update(b, real=False)
    return fid.compute().item()


# ---------------------------------------------------------------------------
# CLIPScore
# ---------------------------------------------------------------------------

def compute_clip_score(
    images: torch.Tensor,       # (N, 3, H, W) in [-1, 1]
    prompts: Sequence[str],     # one prompt per image (or a single prompt broadcast)
    batch_size: int = 64,
    model_name: str = "openai/clip-vit-base-patch32",
) -> float:
    """Mean CLIPScore (text-image alignment, 100 * cosine similarity).

    Uses CLIPModel directly because torchmetrics' CLIPScore wrapper is broken
    with transformers >= 5 (BaseModelOutputWithPooling has no .norm).
    Runs on CPU for portability.
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to('cpu').eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    if isinstance(prompts, str):
        prompts = [prompts] * len(images)
    assert len(prompts) == len(images), f"{len(prompts)} prompts vs {len(images)} images"

    images_cpu = images.detach().cpu()
    # CLIP expects [0, 1] uint8 or float; we have [-1, 1] float -> rescale
    images_01 = (images_cpu * 0.5 + 0.5).clamp(0, 1)

    all_sims = []
    with torch.no_grad():
        for i in range(0, len(images_01), batch_size):
            b_imgs = images_01[i:i+batch_size]
            b_prompts = list(prompts[i:i+batch_size])
            inputs = processor(text=b_prompts, images=b_imgs, return_tensors="pt",
                               padding=True, do_resize=True, do_center_crop=False)
            img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_feat = model.get_text_features(input_ids=inputs["input_ids"],
                                               attention_mask=inputs["attention_mask"])
            # transformers >= 5 returns BaseModelOutputWithPooling; use pooler_output
            if hasattr(img_feat, 'pooler_output'):
                img_feat = img_feat.pooler_output
            if hasattr(txt_feat, 'pooler_output'):
                txt_feat = txt_feat.pooler_output
            img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(p=2, dim=-1, keepdim=True)
            sims = (img_feat * txt_feat).sum(dim=-1)  # (B,)
            all_sims.append(sims)
    sims = torch.cat(all_sims)
    # CLIPScore convention: 100 * mean cosine similarity
    return float(100.0 * sims.mean().item())


# ---------------------------------------------------------------------------
# ImagenWorld TIG VLM-as-judge slice
# ---------------------------------------------------------------------------

_IMAGENWORLD_RUBRIC = """You are evaluating a generated image for the ImagenWorld benchmark.
Rate the image on four criteria, each on a 1-5 Likert scale (1=poor, 5=excellent):

1. Prompt Relevance: Does the image faithfully reflect the instruction "{prompt}"?
2. Aesthetic Quality: Overall visual appeal and design.
3. Content Coherence: Logical and semantic consistency.
4. Artifact: Are there visual flaws, distortions, gibberish text, warped edges, extra limbs?

Respond with ONLY four integers on one line, separated by spaces, in this order:
prompt_relevance aesthetic_quality content_coherence artifact

Example output: 4 3 5 2"""


class ImagenWorldTIGSlice:
    """VLM-as-judge for the ImagenWorld TIG criteria.

    Uses a local open VLM (default: Qwen2.5-VL-7B-Instruct). If the VLM cannot
    be loaded, `evaluate` raises RuntimeError with a clear fallback message.

    The model is unconditional, so the "prompt" is the class label (e.g.
    "a photo of an automobile"). This is a degenerate TIG; we use the rubric
    as a structured sanity eval, not a leaderboard claim.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "mps",
    ):
        self.model_name = model_name
        self.device = device
        self._loaded = False
        self._model = None
        self._processor = None

    def _load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                f"transformers not available: {e}. Install with: "
                "uv pip install transformers"
            ) from e
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            ).to(self.device).eval()
        except Exception as e:
            raise RuntimeError(
                f"Could not load VLM {self.model_name}: {e}\n"
                f"Fallback: use compute_clip_score() instead, or install a local VLM.\n"
                f"  uv pip install transformers\n"
                f"  huggingface-cli login  # if model is gated"
            ) from e
        self._loaded = True

    @torch.no_grad()
    def evaluate(
        self,
        images: torch.Tensor,       # (N, 3, H, W) in [-1, 1]
        prompt: str,                # single prompt for all images (class label)
    ) -> dict:
        """Return per-criterion mean scores and an Overall, on the [0, 1] scale
        used by the paper (5-point Likert rescaled by /5)."""
        self._load()
        from PIL import Image as PILImage

        images_cpu = images.detach().cpu()
        images_uint8 = ((images_cpu * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)

        all_scores = []  # list of (pr, aq, cc, ar) tuples
        for i in range(len(images_uint8)):
            pil_img = PILImage.fromarray(images_uint8[i].permute(1, 2, 0).numpy())
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": _IMAGENWORLD_RUBRIC.format(prompt=prompt)},
                ]}
            ]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[pil_img], return_tensors="pt").to(self.device)
            out = self._model.generate(**inputs, max_new_tokens=20, do_sample=False)
            response = self._processor.batch_decode(
                out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )[0].strip()
            try:
                parts = [int(x) for x in response.split()[:4]]
                if len(parts) == 4 and all(1 <= p <= 5 for p in parts):
                    all_scores.append(parts)
                else:
                    all_scores.append([None, None, None, None])
            except ValueError:
                all_scores.append([None, None, None, None])

        arr = np.array([[s or 3 for s in row] for row in all_scores], dtype=float)
        means = arr.mean(axis=0) / 5.0  # rescale to [0, 1]
        return {
            "n_evaluated": len(all_scores),
            "n_valid": sum(1 for row in all_scores if row[0] is not None),
            "prompt_relevance": float(means[0]),
            "aesthetic_quality": float(means[1]),
            "content_coherence": float(means[2]),
            "artifact": float(means[3]),
            "overall": float(means.mean()),
        }
