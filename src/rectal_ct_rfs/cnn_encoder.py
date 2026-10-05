"""Frozen self-supervised 3D CNN encoder used to generate 128 features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(4, 8, 3, stride=2, padding=1), nn.GroupNorm(4, 8), nn.GELU(),
            nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.proj = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(64, 128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x))


def load_encoder(checkpoint: str | Path, device: str = "cpu") -> Encoder:
    model = Encoder().to(device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["encoder"])
    return model


def prepare_four_channel_input(ct: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """Create CT, bowel-wall, tumor, and nodal channels at 64³ voxels.

    The input arrays must already be cropped and resampled according to the
    study preprocessing specification. Mask labels are 1=bowel wall,
    2=primary tumor, and 3=regional lymph nodes.
    """
    if ct.shape != mask.shape or ct.ndim != 3:
        raise ValueError("ct and mask must be same-shape 3D arrays")
    channels = np.stack([ct, mask == 1, mask == 2, mask == 3], axis=0).astype(np.float32)
    tensor = torch.from_numpy(channels)
    tensor = F.interpolate(tensor[None], size=(64, 64, 64), mode="trilinear", align_corners=False)[0]
    tensor[1:] = (tensor[1:] > 0.25).float()
    return tensor


@torch.no_grad()
def encode(model: Encoder, ct: np.ndarray, mask: np.ndarray, device: str = "cpu") -> np.ndarray:
    tensor = prepare_four_channel_input(ct, mask)[None].to(device)
    return model(tensor).float().cpu().numpy()[0]
