# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.helios.fused_ops import layer_norm_two_population_affine

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("history_length", [0, 2205])
@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_layer_norm_affine_npu_matches_decomposed(history_length: int) -> None:
    """Exercise the fused expression at Helios's TP1 hidden-state shape."""
    torch.manual_seed(1101)
    x = torch.randn(1, 2940, 5120, dtype=torch.bfloat16, device="npu")
    scale_history = torch.randn(1, 1, 5120, dtype=torch.float32, device="npu")
    shift_history = torch.randn(1, 1, 5120, dtype=torch.float32, device="npu")
    scale_current = torch.randn(1, 1, 5120, dtype=torch.float32, device="npu")
    shift_current = torch.randn(1, 1, 5120, dtype=torch.float32, device="npu")

    normalized = F.layer_norm(x.float(), (5120,), None, None, 1e-6)
    if history_length:
        expected = torch.cat(
            (
                normalized[:, :history_length] * scale_history + shift_history,
                normalized[:, history_length:] * scale_current + shift_current,
            ),
            dim=1,
        )
    else:
        expected = normalized * scale_current + shift_current

    actual = layer_norm_two_population_affine(
        x,
        scale_history,
        shift_history,
        scale_current,
        shift_current,
        history_length,
        (5120,),
        1e-6,
    ).type_as(x)

    torch.testing.assert_close(actual, expected.type_as(x), atol=0, rtol=0)
