# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vllm.config.memory_profiler import MemoryProfilerConfig
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.memory_profiling.types import (
    GPUProfilerBreakdown,
    GPUProfilerMeta,
    GPUProfilerSample,
    GPUProfilerSnapshot,
)

logger = init_logger(__name__)


def _compute_activations_and_unaccounted(
    *,
    used_memory_bytes: int,
    torch_allocated_bytes: int,
    model_weights_bytes: int,
    kv_cache_reserved_bytes: int,
    cuda_graph_bytes: int,
    fragmentation_bytes: int,
) -> tuple[int, int, int]:
    activation_candidate = max(
        torch_allocated_bytes - model_weights_bytes - kv_cache_reserved_bytes,
        0,
    )
    activation_budget = max(
        used_memory_bytes
        - model_weights_bytes
        - kv_cache_reserved_bytes
        - cuda_graph_bytes
        - fragmentation_bytes,
        0,
    )
    activations_bytes = min(activation_candidate, activation_budget)
    unaccounted_bytes = max(
        used_memory_bytes
        - model_weights_bytes
        - kv_cache_reserved_bytes
        - activations_bytes
        - cuda_graph_bytes
        - fragmentation_bytes,
        0,
    )
    return activations_bytes, unaccounted_bytes, activation_candidate


def build_gpu_profiler_sample(
    worker_state: dict[str, Any],
    scheduler_state: dict[str, Any],
) -> GPUProfilerSample:
    kv_cache_reserved_bytes = int(worker_state.get("kv_cache_reserved_bytes", 0))
    kv_cache_usage_fraction = float(scheduler_state.get("kv_cache_usage_fraction", 0.0))
    kv_cache_used_bytes = int(round(kv_cache_reserved_bytes * kv_cache_usage_fraction))
    model_weights_bytes = int(worker_state.get("model_weights_bytes", 0))
    used_memory_bytes = int(worker_state.get("used_memory_bytes", 0))
    torch_allocated_bytes = int(worker_state.get("torch_allocated_bytes", 0))
    torch_reserved_bytes = int(worker_state.get("torch_reserved_bytes", 0))
    cuda_graph_bytes = int(worker_state.get("cuda_graph_bytes", 0))
    fragmentation_bytes = max(torch_reserved_bytes - torch_allocated_bytes, 0)
    activations_bytes, unaccounted_bytes, activation_candidate = (
        _compute_activations_and_unaccounted(
            used_memory_bytes=used_memory_bytes,
            torch_allocated_bytes=torch_allocated_bytes,
            model_weights_bytes=model_weights_bytes,
            kv_cache_reserved_bytes=kv_cache_reserved_bytes,
            cuda_graph_bytes=cuda_graph_bytes,
            fragmentation_bytes=fragmentation_bytes,
        )
    )

    breakdown = GPUProfilerBreakdown(
        model_weights_bytes=model_weights_bytes,
        kv_cache_bytes=kv_cache_reserved_bytes,
        activations_bytes=activations_bytes,
        cuda_graph_bytes=cuda_graph_bytes,
        fragmentation_bytes=fragmentation_bytes,
        unaccounted_bytes=unaccounted_bytes,
    )
    meta = GPUProfilerMeta(
        worker_rank=int(worker_state["worker_rank"]),
        local_rank=int(worker_state["local_rank"]),
        gpu_uuid=str(worker_state.get("gpu_uuid", "")),
        physical_gpu_id=int(worker_state.get("physical_gpu_id", worker_state["gpu_id"])),
        sampling_method=str(worker_state.get("sampling_method", "hybrid_nvml_vllm_torch")),
        model_weights_source=str(
            worker_state.get("model_weights_source", "unknown")
        ),
        kv_cache_usage_fraction=kv_cache_usage_fraction,
        kv_cache_reserved_bytes=kv_cache_reserved_bytes,
        kv_cache_used_bytes=kv_cache_used_bytes,
        kv_cache_num_blocks=int(worker_state.get("kv_cache_num_blocks", 0)),
        cuda_graph_capture_bytes=cuda_graph_bytes,
        cuda_graph_capture_events=int(
            worker_state.get("cuda_graph_capture_events", 0)
        ),
        torch_allocated_bytes=torch_allocated_bytes,
        torch_reserved_bytes=torch_reserved_bytes,
        activation_candidate_bytes=activation_candidate,
        non_torch_used_bytes=int(worker_state.get("non_torch_used_bytes", 0)),
        scheduler_iteration=int(scheduler_state.get("iteration", 0)),
        scheduler_running_requests=int(
            scheduler_state.get("running_requests", worker_state.get("last_running_requests", 0))
        ),
        scheduler_waiting_requests=int(
            scheduler_state.get("waiting_requests", worker_state.get("last_waiting_requests", 0))
        ),
        worker_event_seq=int(worker_state.get("worker_event_seq", 0)),
        last_event=str(worker_state.get("last_event", "poll")),
        is_estimated={
            "activations": True,
            "cuda_graph": False,
            "fragmentation": False,
            "kv_cache": False,
            "model_weights": False,
            "unaccounted": True,
        },
        notes={
            "kv_cache": "breakdown.kv_cache_bytes reports reserved capacity; meta also exposes used bytes",
            "activations": "estimated from torch allocated minus persistent weights and KV cache",
            "fragmentation": "defined as torch_reserved_bytes - torch_allocated_bytes",
            "unaccounted": "remaining NVML used bytes after modeled components",
        },
    )
    return GPUProfilerSample(
        gpu_id=int(worker_state["gpu_id"]),
        name=str(worker_state["name"]),
        total_memory_bytes=int(worker_state["total_memory_bytes"]),
        used_memory_bytes=used_memory_bytes,
        free_memory_bytes=int(worker_state["free_memory_bytes"]),
        breakdown=breakdown,
        meta=meta,
    )


class MemoryProfilerService:
    """Frontend-side snapshot aggregator and broadcaster."""

    def __init__(
        self,
        engine_client: EngineClient,
        config: MemoryProfilerConfig,
    ) -> None:
        self.engine_client = engine_client
        self.config = config
        self.history: deque[GPUProfilerSnapshot] = deque(maxlen=config.history_size)
        self._sequence = 0
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._subscribers: set[asyncio.Queue[GPUProfilerSnapshot]] = set()
        self._jsonl_fp = None
        self._last_snapshot: GPUProfilerSnapshot | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        if self.config.jsonl_path:
            path = Path(self.config.jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_fp = path.open("a", encoding="utf-8")
        self._task = asyncio.create_task(self._run(), name="vllm-memory-profiler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._jsonl_fp is not None:
            self._jsonl_fp.close()
            self._jsonl_fp = None

    async def _run(self) -> None:
        await self.collect_once(force=True)
        interval = self.config.sampling_interval_ms / 1000.0
        while True:
            await asyncio.sleep(interval)
            try:
                await self.collect_once()
            except Exception:
                logger.exception("Memory profiler polling failed")

    async def _fetch_scheduler_state(self) -> dict[str, Any]:
        return await self.engine_client.get_memory_profiler_engine_state()

    async def _fetch_worker_states(self) -> list[dict[str, Any]]:
        states = await self.engine_client.collective_rpc("get_memory_profiler_state")
        return [state for state in states if state and state.get("enabled", False)]

    async def collect_once(
        self,
        *,
        force: bool = False,
    ) -> GPUProfilerSnapshot | None:
        if not self.enabled:
            return None
        async with self._lock:
            scheduler_state, worker_states = await asyncio.gather(
                self._fetch_scheduler_state(),
                self._fetch_worker_states(),
            )
            if not worker_states:
                return self._last_snapshot

            samples = [
                build_gpu_profiler_sample(worker_state, scheduler_state)
                for worker_state in sorted(
                    worker_states, key=lambda item: (item["gpu_id"], item["worker_rank"])
                )
            ]
            timestamp = max(
                float(scheduler_state.get("timestamp", 0.0)),
                max(float(worker.get("timestamp", 0.0)) for worker in worker_states),
            )
            if timestamp <= 0:
                timestamp = time.time()
            next_snapshot = GPUProfilerSnapshot(
                timestamp=timestamp,
                sequence=self._sequence,
                gpus=samples,
            )

            if not force and not self._should_store(next_snapshot):
                return self._last_snapshot

            self._sequence += 1
            next_snapshot.sequence = self._sequence
            self.history.append(next_snapshot)
            self._last_snapshot = next_snapshot
            self._write_jsonl(next_snapshot)
            await self._broadcast(next_snapshot)
            return next_snapshot

    def _should_store(self, snapshot: GPUProfilerSnapshot) -> bool:
        if self._last_snapshot is None:
            return True
        threshold = self.config.min_change_bytes
        if threshold <= 0:
            return True
        previous_by_gpu = {gpu.gpu_id: gpu for gpu in self._last_snapshot.gpus}
        for gpu in snapshot.gpus:
            prev = previous_by_gpu.get(gpu.gpu_id)
            if prev is None:
                return True
            fields = (
                gpu.used_memory_bytes,
                gpu.breakdown.model_weights_bytes,
                gpu.breakdown.kv_cache_bytes,
                gpu.breakdown.activations_bytes,
                gpu.breakdown.cuda_graph_bytes,
                gpu.breakdown.fragmentation_bytes,
                gpu.breakdown.unaccounted_bytes,
                gpu.meta.kv_cache_used_bytes,
            )
            prev_fields = (
                prev.used_memory_bytes,
                prev.breakdown.model_weights_bytes,
                prev.breakdown.kv_cache_bytes,
                prev.breakdown.activations_bytes,
                prev.breakdown.cuda_graph_bytes,
                prev.breakdown.fragmentation_bytes,
                prev.breakdown.unaccounted_bytes,
                prev.meta.kv_cache_used_bytes,
            )
            if any(abs(cur - old) >= threshold for cur, old in zip(fields, prev_fields)):
                return True
        return False

    def _write_jsonl(self, snapshot: GPUProfilerSnapshot) -> None:
        if self._jsonl_fp is None:
            return
        self._jsonl_fp.write(json.dumps(snapshot.to_dict(), ensure_ascii=False) + "\n")
        self._jsonl_fp.flush()

    async def _broadcast(self, snapshot: GPUProfilerSnapshot) -> None:
        stale: list[asyncio.Queue[GPUProfilerSnapshot]] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(snapshot)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            self._subscribers.discard(queue)

    def get_latest_snapshot(self) -> dict[str, Any] | None:
        return self._last_snapshot.to_dict() if self._last_snapshot else None

    def get_history(self) -> list[dict[str, Any]]:
        return [snapshot.to_dict() for snapshot in self.history]

    def subscribe(self) -> asyncio.Queue[GPUProfilerSnapshot]:
        queue: asyncio.Queue[GPUProfilerSnapshot] = asyncio.Queue(maxsize=4)
        self._subscribers.add(queue)
        if self._last_snapshot is not None:
            queue.put_nowait(self._last_snapshot)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[GPUProfilerSnapshot]) -> None:
        self._subscribers.discard(queue)
