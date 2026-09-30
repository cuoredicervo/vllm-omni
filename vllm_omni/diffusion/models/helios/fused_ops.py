# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small fused tensor expressions used by Helios on Ascend."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
