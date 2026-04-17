# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash


@config
class MemoryProfilerConfig:
    """Configuration for the live GPU memory breakdown profiler."""

    enabled: bool = False
    """Enable the GPU memory profiler."""

    sampling_interval_ms: int = Field(default=1000, ge=100)
    """Polling interval for frontend snapshot collection."""
    """for default, we sample every second"""

    history_size: int = Field(default=3600, ge=300)
    """Number of recent snapshots to keep in the in-memory ring buffer."""
    """for default, we record last history_size * (sampling_interval_ms / 1000) seconds"""

    web_enabled: bool = True
    """Expose the built-in webpage and HTTP/WebSocket APIs."""

    web_path: str = "/debug/memory-profiler"
    """Base path for the profiler webpage and APIs on the API server."""

    jsonl_path: str | None = None
    """Optional JSONL file path for snapshot archival."""

    min_change_bytes: int = Field(default=0, ge=0)
    """Only push a new point into history when change exceeds this threshold."""

    max_points_per_gpu: int = Field(default=3000, ge=32)
    """Compatibility alias for future frontend tuning. Currently matches history_size."""

    def compute_hash(self) -> str:
        factors: list[Any] = [
            self.enabled,
            self.sampling_interval_ms,
            self.history_size,
            self.web_enabled,
            self.web_path,
            self.min_change_bytes,
            self.max_points_per_gpu,
        ]
        return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()

    @model_validator(mode="after")
    def _normalize_paths(self) -> Self:
        if self.jsonl_path:
            self.jsonl_path = os.path.abspath(os.path.expanduser(self.jsonl_path))
        return self
