"""Small SDPA compatibility shim for checkpoints that import flash-attn.

The GR00T RoboCasa checkpoint uses the SigLIP vision path, but its custom
RADIO module imports flash-attn unconditionally.  This shim makes that import
optional on CUDA machines where the compiled flash-attn wheel is unavailable.
"""
from __future__ import annotations

import torch


def _attention(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    """PyTorch SDPA implementation for the flash-attn API subset used here."""
    # Transformers may pass grouped-query attention (more query than
    # key/value heads), while SDPA in this Torch version expects matching
    # head counts unless enable_gqa is explicitly available.
    if q.shape[-2] != k.shape[-2]:
        if q.shape[-2] % k.shape[-2]:
            raise ValueError(f"incompatible attention heads: q={q.shape}, k={k.shape}")
        repeats = q.shape[-2] // k.shape[-2]
        k = k.repeat_interleave(repeats, dim=-2)
        v = v.repeat_interleave(repeats, dim=-2)
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    ).transpose(1, 2)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    return _attention(q, k, v, dropout_p, softmax_scale, causal)


def flash_attn_varlen_func(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, softmax_scale=None, causal=False, **kwargs
):
    outputs = []
    for q_start, q_end, k_start, k_end in zip(
        cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist(),
        cu_seqlens_k[:-1].tolist(), cu_seqlens_k[1:].tolist(), strict=True
    ):
        outputs.append(_attention(
            q[q_start:q_end].unsqueeze(0),
            k[k_start:k_end].unsqueeze(0),
            v[k_start:k_end].unsqueeze(0),
            dropout_p, softmax_scale, causal,
        )[0])
    if not outputs:
        return q.new_empty((0, q.shape[-2], q.shape[-1]))
    return torch.cat(outputs, dim=0)
