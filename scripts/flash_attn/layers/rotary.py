from __future__ import annotations

import torch


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Apply rotary embeddings with the flash-attn-compatible call signature."""
    del kwargs
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.ndim == 3 and x.ndim == 4:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    if interleaved:
        even = x[..., ::2]
        odd = x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        rotated = rotated.flatten(-2)
    else:
        half = x.shape[-1] // 2
        first, second = x[..., :half], x[..., half:]
        rotated = torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)
    if inplace:
        x.copy_(rotated)
        return x
    return rotated
