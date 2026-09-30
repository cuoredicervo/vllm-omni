# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small fused tensor expressions used by Helios on Ascend."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bf16_residual_gate(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Apply a residual gate without explicitly promoting BF16 operands.

    Keep the expression intact. Materializing the product in BF16 before the
    addition introduces an extra rounding and is not equivalent.
    """
    if residual.dtype != torch.bfloat16 or branch.dtype != torch.bfloat16:
        raise TypeError("BF16 residual gating requires BF16 residual and branch tensors")
    return (residual + branch * gate).type_as(residual)


def layer_norm_two_population_affine(
    x: torch.Tensor,
    scale_history: torch.Tensor | None,
    shift_history: torch.Tensor | None,
    scale_current: torch.Tensor,
    shift_current: torch.Tensor,
    history_length: int,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    """Fuse LayerNorm with Helios's population-specific affine parameters."""

    def fused(piece: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            piece,
            normalized_shape,
            scale.reshape(normalized_shape),
            shift.reshape(normalized_shape),
            eps,
        )

    if history_length <= 0:
        return fused(x, scale_current, shift_current)
    assert scale_history is not None and shift_history is not None
    return torch.cat(
        (
            fused(x[:, :history_length], scale_history, shift_history),
            fused(x[:, history_length:], scale_current, shift_current),
        ),
        dim=1,
    )
