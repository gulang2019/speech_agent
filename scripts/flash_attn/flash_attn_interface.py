from __future__ import annotations

import torch


def flash_attn_unpadded_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """SDPA fallback matching flash-attn's packed-QKV return shape."""
    outputs = []
    scale = softmax_scale
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
        packed = qkv[start:end]
        q, k, v = packed.unbind(dim=1)
        output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            dropout_p=dropout_p,
            is_causal=causal,
            scale=scale,
        )[0].transpose(0, 1)
        outputs.append(output)
    if not outputs:
        return qkv.new_empty((0, qkv.shape[-2], qkv.shape[-1]))
    return torch.cat(outputs, dim=0)


flash_attn_varlen_qkvpacked_func = flash_attn_unpadded_qkvpacked_func
