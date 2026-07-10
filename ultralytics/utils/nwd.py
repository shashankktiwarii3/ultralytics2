# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Normalized Wasserstein Distance (Xu et al. 2022, AI-TOD) + tiny-object size gate."""

from __future__ import annotations

import torch


def bbox_nwd(box1: torch.Tensor, box2: torch.Tensor, C: float = 12.8, eps: float = 1e-7) -> torch.Tensor:
    """Normalized Wasserstein Distance between xyxy boxes modeled as 2D Gaussians N(c, diag(w^2/4, h^2/4)).

    W2^2 = ||c1-c2||^2 + (||w1-w2||^2 + ||h1-h2||^2)/4 ; NWD = exp(-sqrt(W2)/C).

    Args:
        box1 (torch.Tensor): Boxes in xyxy format, pixel space.
        box2 (torch.Tensor): Boxes in xyxy format, pixel space, broadcastable with box1.
        C (float): Dataset mean absolute box size at train resolution (AI-TOD convention: 12.8).
        eps (float): Numerical stability epsilon.

    Returns:
        (torch.Tensor): NWD similarity in [0, 1], same leading shape as box1/box2.
    """
    cx1, cy1 = (box1[..., 0] + box1[..., 2]) / 2, (box1[..., 1] + box1[..., 3]) / 2
    w1, h1 = box1[..., 2] - box1[..., 0], box1[..., 3] - box1[..., 1]
    cx2, cy2 = (box2[..., 0] + box2[..., 2]) / 2, (box2[..., 1] + box2[..., 3]) / 2
    w2, h2 = box2[..., 2] - box2[..., 0], box2[..., 3] - box2[..., 1]
    w2d = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    return torch.exp(-torch.sqrt(w2d.clamp(min=eps)) / C)


def tiny_gamma(gt_xyxy: torch.Tensor, lo: float = 8.0, hi: float = 32.0) -> torch.Tensor:
    """Blend weight: 1.0 for objects with sqrt(w*h) <= lo px, 0.0 for >= hi px, linear ramp between.

    Units are input-image pixels at train imgsz — keep lo tied to s_min (P3 stride = 8).
    """
    w = gt_xyxy[..., 2] - gt_xyxy[..., 0]
    h = gt_xyxy[..., 3] - gt_xyxy[..., 1]
    size = torch.sqrt((w * h).clamp(min=0))
    return ((hi - size) / (hi - lo)).clamp(0.0, 1.0)
