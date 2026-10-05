# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.helios import helios_transformer, perf_gates
from vllm_omni.diffusion.models.helios.helios_transformer import DistributedRMSNorm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _reference(norm: DistributedRMSNorm, x: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    rms = torch.sqrt((x_float**2).mean(dim=-1, keepdim=True) + norm.eps)
    return ((x_float / rms) * norm.weight.float()).to(x.dtype)


def test_fused_gate_keeps_cpu_on_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(perf_gates, "FUSED_RMS_NORM", True)
    monkeypatch.setattr(helios_transformer, "get_tensor_model_parallel_world_size", lambda: 1)
    norm = DistributedRMSNorm(128, eps=1e-5)
    x = torch.randn(1, 32, 128, dtype=torch.bfloat16)
    torch.testing.assert_close(norm(x), _reference(norm, x), atol=0, rtol=0)


def test_fused_gate_keeps_tp2_on_distributed_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(perf_gates, "FUSED_RMS_NORM", True)
    monkeypatch.setattr(helios_transformer, "get_tensor_model_parallel_world_size", lambda: 2)
    all_reduce_calls = 0

    def all_reduce_identical_rank(tensor: torch.Tensor) -> None:
        nonlocal all_reduce_calls
        all_reduce_calls += 1
        tensor.mul_(2)

    monkeypatch.setattr(helios_transformer, "tensor_model_parallel_all_reduce", all_reduce_identical_rank)
    norm = DistributedRMSNorm(128, eps=1e-5)
    x = torch.randn(1, 32, 128, dtype=torch.bfloat16)
    torch.testing.assert_close(norm(x), _reference(norm, x), atol=0, rtol=0)
    assert all_reduce_calls == 1
