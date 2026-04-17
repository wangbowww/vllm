# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class GPUProfilerBreakdown:
    model_weights_bytes: int = 0
    kv_cache_bytes: int = 0
    activations_bytes: int = 0
    cuda_graph_bytes: int = 0
    fragmentation_bytes: int = 0
    unaccounted_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class GPUProfilerMeta:
    worker_rank: int
    local_rank: int
    gpu_uuid: str
    physical_gpu_id: int
    sampling_method: str
    model_weights_source: str
    kv_cache_usage_fraction: float = 0.0
    kv_cache_reserved_bytes: int = 0
    kv_cache_used_bytes: int = 0
    kv_cache_num_blocks: int = 0
    cuda_graph_capture_bytes: int = 0
    cuda_graph_capture_events: int = 0
    torch_allocated_bytes: int = 0
    torch_reserved_bytes: int = 0
    activation_candidate_bytes: int = 0
    non_torch_used_bytes: int = 0
    scheduler_iteration: int = 0
    scheduler_running_requests: int = 0
    scheduler_waiting_requests: int = 0
    worker_event_seq: int = 0
    last_event: str = "poll"
    is_estimated: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GPUProfilerSample:
    gpu_id: int
    name: str
    total_memory_bytes: int
    used_memory_bytes: int
    free_memory_bytes: int
    breakdown: GPUProfilerBreakdown
    meta: GPUProfilerMeta

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "total_memory_bytes": self.total_memory_bytes,
            "used_memory_bytes": self.used_memory_bytes,
            "free_memory_bytes": self.free_memory_bytes,
            "breakdown": self.breakdown.to_dict(),
            "meta": self.meta.to_dict(),
        }


@dataclass(slots=True)
class GPUProfilerSnapshot:
    timestamp: float
    sequence: int
    gpus: list[GPUProfilerSample]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "gpus": [gpu.to_dict() for gpu in self.gpus],
        }
