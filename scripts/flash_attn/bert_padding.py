from __future__ import annotations

import torch


def index_first_axis(x: torch.Tensor, indices: torch.Tensor):
    return x[indices]


def unpad_input(x: torch.Tensor, mask: torch.Tensor):
    """Return the subset selected by a ``[batch, seq]`` boolean mask."""
    batch, seq = mask.shape
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    lengths = mask.sum(dim=1, dtype=torch.int32)
    cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device=x.device)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
    return x.reshape(batch * seq, *x.shape[2:])[indices], indices, cu_seqlens, max_seqlen


def pad_input(x: torch.Tensor, indices: torch.Tensor, batch: int, seq: int):
    output = torch.zeros((batch * seq, *x.shape[1:]), dtype=x.dtype, device=x.device)
    output[indices] = x
    return output.reshape(batch, seq, *x.shape[1:])
