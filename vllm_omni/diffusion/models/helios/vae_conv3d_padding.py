# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Avoid materializing symmetric spatial padding for Wan VAE Conv3D."""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d


def conv3d_with_internal_spatial_padding(
    self: WanCausalConv3d,
    x: torch.Tensor,
    cache_x: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep causal temporal padding explicit and move H/W padding into Conv3D."""
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]

    temporal_left, temporal_right = padding[4], padding[5]
    if temporal_left != 0 or temporal_right != 0:
        x = F.pad(x, (0, 0, 0, 0, temporal_left, temporal_right))

    return F.conv3d(
        x,
        self.weight,
        self.bias,
        self.stride,
        padding=(0, self._padding[2], self._padding[0]),
        dilation=self.dilation,
        groups=self.groups,
    )


def enable_internal_spatial_padding(vae: torch.nn.Module) -> int:
    """Patch the Wan causal convolutions in one VAE instance."""
    modules = [module for module in vae.modules() if isinstance(module, WanCausalConv3d)]
    if not modules:
        raise RuntimeError("No WanCausalConv3d modules were found in the Helios VAE.")
    if any(hasattr(module, "_helios_original_forward") for module in modules):
        raise RuntimeError("Wan VAE Conv3D spatial padding was enabled more than once.")

    for module in modules:
        module._helios_original_forward = module.forward
        module.forward = MethodType(conv3d_with_internal_spatial_padding, module)
    return len(modules)
