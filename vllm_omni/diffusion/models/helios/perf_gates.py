# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit, default-off gates for approximate Helios optimizations."""

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


# This casts the much smaller RoPE cosine/sine tables to BF16 before the fused
# Ascend kernel. It improves performance but changes model output, so it must
# never become an implicit platform default.
BF16_ROPE_FREQUENCIES = _read_flag("HELIOS_BF16_ROPE_FREQUENCIES")
