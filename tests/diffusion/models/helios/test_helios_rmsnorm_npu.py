# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.helios import helios_transformer, perf_gates
from vllm_omni.diffusion.models.helios.helios_transformer import DistributedRMSNorm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _reference(norm: DistributedRMSNorm, x: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    rms = torch.sqrt((x_float**2).mean(dim=-1, keepdim=True) + norm.eps)
    return ((x_float / rms) * norm.weight.float()).to(x.dtype)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_fused_tp1_rmsnorm_is_explicit_and_numerically_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the opt-in kernel at a representative Helios attention shape."""
    torch.manual_seed(1101)
    monkeypatch.setattr(perf_gates, "FUSED_RMS_NORM", True)
    monkeypatch.setattr(helios_transformer, "get_tensor_model_parallel_world_size", lambda: 1)

    norm = DistributedRMSNorm(5120, eps=1e-5).to(device="npu", dtype=torch.float32)
    x = torch.randn(1, 735, 5120, dtype=torch.bfloat16, device="npu")
    expected = _reference(norm, x)
    actual = norm(x)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_fused_rmsnorm_gate_absent_preserves_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production default must retain Helios's explicit FP32 expression."""
    torch.manual_seed(1101)
    monkeypatch.setattr(perf_gates, "FUSED_RMS_NORM", False)
    monkeypatch.setattr(helios_transformer, "get_tensor_model_parallel_world_size", lambda: 1)

    norm = DistributedRMSNorm(128, eps=1e-5).to(device="npu", dtype=torch.float32)
    x = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="npu")
    torch.testing.assert_close(norm(x), _reference(norm, x), atol=0, rtol=0)
