# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any, Protocol


class _MemoryProfilerTracker(Protocol):
    def record_cuda_graph_capture(
        self,
        delta_bytes: int,
        *,
        batch_descriptor: Any | None = None,
        runtime_mode: str | None = None,
    ) -> None: ...


_PROCESS_TRACKER: _MemoryProfilerTracker | None = None


def set_process_memory_tracker(
    tracker: _MemoryProfilerTracker | None,
) -> None:
    global _PROCESS_TRACKER
    _PROCESS_TRACKER = tracker


def get_process_memory_tracker() -> _MemoryProfilerTracker | None:
    return _PROCESS_TRACKER


def record_cuda_graph_capture(
    delta_bytes: int,
    *,
    batch_descriptor: Any | None = None,
    runtime_mode: str | None = None,
) -> None:
    tracker = _PROCESS_TRACKER
    if tracker is None:
        return
    tracker.record_cuda_graph_capture(
        delta_bytes,
        batch_descriptor=batch_descriptor,
        runtime_mode=runtime_mode,
    )
