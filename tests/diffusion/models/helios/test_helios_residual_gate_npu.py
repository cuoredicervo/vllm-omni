# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.helios.fused_ops import bf16_residual_gate

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_bf16_residual_gate_npu_matches_promoted_expression() -> None:
    """Exercise the gate at Helios's TP1 hidden-state shape."""
    torch.manual_seed(1501)
    residual = torch.randn(1, 2940, 5120, dtype=torch.bfloat16, device="npu")
    branch = torch.randn_like(residual)
    gate = torch.randn(1, 2940, 5120, dtype=torch.float32, device="npu")

    expected = (residual.float() + branch.float() * gate).type_as(residual)
    actual = bf16_residual_gate(residual, branch, gate)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
