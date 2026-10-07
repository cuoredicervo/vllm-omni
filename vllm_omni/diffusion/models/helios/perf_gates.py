# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit, default-off gate for approximate fused Helios RMSNorm."""

from __future__ import annotations

import os


def _read_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}.")


# The Ascend native RMSNorm changes accumulation/rounding relative to Helios's
# explicit FP32 reference expression. It is currently valid only for TP=1;
# distributed TP must retain the reference reduction.
FUSED_RMS_NORM = _read_flag("HELIOS_FUSED_RMS_NORM")
