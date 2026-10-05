# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[4] / "vllm_omni" / "diffusion" / "models" / "helios" / "perf_gates.py"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load(monkeypatch: pytest.MonkeyPatch, **environment: str):
    monkeypatch.delenv("HELIOS_BF16_ROPE_FREQUENCIES", raising=False)
    monkeypatch.delenv("HELIOS_FUSED_RMS_NORM", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location("test_helios_perf_gates", _PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_approximate_optimizations_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load(monkeypatch)
    assert module.BF16_ROPE_FREQUENCIES is False
    assert module.FUSED_RMS_NORM is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " TRUE "])
def test_bf16_rope_frequency_gate_accepts_explicit_true(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    module = _load(monkeypatch, HELIOS_BF16_ROPE_FREQUENCIES=value)
    assert module.BF16_ROPE_FREQUENCIES is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " FALSE "])
def test_bf16_rope_frequency_gate_accepts_explicit_false(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    module = _load(monkeypatch, HELIOS_BF16_ROPE_FREQUENCIES=value)
    assert module.BF16_ROPE_FREQUENCIES is False


def test_invalid_approximation_gate_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        _load(monkeypatch, HELIOS_BF16_ROPE_FREQUENCIES="maybe")


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " TRUE "])
def test_fused_rms_norm_gate_accepts_explicit_true(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    module = _load(monkeypatch, HELIOS_FUSED_RMS_NORM=value)
    assert module.FUSED_RMS_NORM is True
