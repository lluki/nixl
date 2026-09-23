# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in JSONL timing events for NIXLShard experiments."""

from __future__ import annotations

import json
import os
from typing import Any


def emit_trace(event: dict[str, Any]) -> None:
    path = os.environ.get("NIXLSHARD_TRACE_PATH")
    if not path:
        return
    try:
        payload = (json.dumps(event, separators=(",", ":")) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except OSError:
        # Diagnostic output must not change storage results.
        pass
