# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

_PATH = Path(__file__).parents[4] / "vllm_omni" / "diffusion" / "models" / "helios" / "vae_conv3d_padding.py"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load():
    spec = importlib.util.spec_from_file_location("test_helios_vae_conv3d_padding", _PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cache_frames", [None, 2])
def test_internal_spatial_padding_matches_original(cache_frames: int | None) -> None:
    torch.manual_seed(0)
    original = WanCausalConv3d(3, 5, kernel_size=3, stride=1, padding=1)
    patched = WanCausalConv3d(3, 5, kernel_size=3, stride=1, padding=1)
    patched.load_state_dict(original.state_dict())
    x = torch.randn(2, 3, 4, 7, 9)
    cache = None if cache_frames is None else torch.randn(2, 3, cache_frames, 7, 9)

    expected = original(x, cache_x=cache)
    assert _load().enable_internal_spatial_padding(patched) == 1
    actual = patched(x, cache_x=cache)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_enable_rejects_duplicate_patch() -> None:
    conv = WanCausalConv3d(2, 2, kernel_size=3, padding=1)
    module = _load()
    module.enable_internal_spatial_padding(conv)
    with pytest.raises(RuntimeError, match="more than once"):
        module.enable_internal_spatial_padding(conv)
