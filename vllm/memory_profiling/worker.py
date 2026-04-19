# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from typing import Any

import torch

from vllm.config.memory_profiler import MemoryProfilerConfig
from vllm.logger import init_logger
from vllm.memory_profiling.runtime import set_process_memory_tracker
from vllm.platforms import current_platform
from vllm.utils.import_utils import import_pynvml

logger = init_logger(__name__)


def _to_int(value: Any) -> int:
    return int(value) if value is not None else 0


class WorkerMemoryProfiler:
    """Process-local memory collector for a single CUDA worker."""

    def __init__(
        self,
        config: MemoryProfilerConfig,
        *,
        rank: int,
        local_rank: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.device = torch.device(device)
        self.enabled = config.enabled and self.device.type == "cuda"

        self.gpu_name = torch.cuda.get_device_name(self.device)
        self.model_weights_bytes = 0
        self.model_weights_source = "uninitialized"
        self.kv_cache_reserved_bytes = 0
        self.kv_cache_num_blocks = 0
        self.cuda_graph_capture_bytes = 0
        self.cuda_graph_capture_events = 0
        self.worker_event_seq = 0
        self.last_event = "init"
        self.last_event_ts = time.time()
        self.last_forward_pass = False
        self.last_num_scheduled_tokens = 0
        self.last_running_requests = 0
        self.last_waiting_requests = 0

        self._pynvml = None
        self._nvml_handle = None
        self.physical_gpu_id = current_platform.device_id_to_physical_device_id(
            self.device.index or 0
        )
        self.gpu_uuid = ""
        self._sampling_method = "hybrid_nvml_vllm_torch"

        if self.enabled:
            self._init_nvml()
            set_process_memory_tracker(self)

    def shutdown(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                logger.debug("Failed to shutdown NVML cleanly", exc_info=True)
        set_process_memory_tracker(None)

    def _init_nvml(self) -> None:
        pynvml = import_pynvml()
        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical_gpu_id)
        raw_uuid = pynvml.nvmlDeviceGetUUID(self._nvml_handle)
        self.gpu_uuid = (
            raw_uuid.decode("utf-8") if isinstance(raw_uuid, bytes) else str(raw_uuid)
        )
        raw_name = pynvml.nvmlDeviceGetName(self._nvml_handle)
        self.gpu_name = (
            raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        )

    def _mark_event(self, event: str) -> None:
        self.worker_event_seq += 1
        self.last_event = event
        self.last_event_ts = time.time()

    def record_model_weights(self, model: torch.nn.Module, fallback_bytes: int) -> None:
        scanned_bytes = 0
        seen_ptrs: set[tuple[int, int]] = set()
        for tensor in list(model.parameters()) + list(model.buffers()):
            if tensor.device == self.device:
                tensor_bytes = tensor.numel() * tensor.element_size()
                tensor_key = (tensor.data_ptr(), tensor_bytes)
                if tensor_key in seen_ptrs:
                    continue
                seen_ptrs.add(tensor_key)
                scanned_bytes += tensor_bytes
        if scanned_bytes > 0:
            self.model_weights_bytes = scanned_bytes
            self.model_weights_source = "model_params_and_buffers_scan"
        else:
            self.model_weights_bytes = int(fallback_bytes)
            self.model_weights_source = "device_memory_profiler_delta"
        self._mark_event("model_loaded")

    def record_kv_cache_config(self, kv_cache_config: Any) -> None:
        self.kv_cache_reserved_bytes = sum(
            int(tensor.size) for tensor in kv_cache_config.kv_cache_tensors
        )
        self.kv_cache_num_blocks = int(kv_cache_config.num_blocks)
        self._mark_event("kv_cache_initialized")

    def record_scheduler_activity(self, scheduler_output: Any) -> None:
        self.last_num_scheduled_tokens = int(
            getattr(scheduler_output, "total_num_scheduled_tokens", 0)
        )
        self.last_forward_pass = self.last_num_scheduled_tokens > 0
        self._mark_event("execute_model")

    def record_scheduler_counts(
        self,
        *,
        running_requests: int,
        waiting_requests: int,
    ) -> None:
        self.last_running_requests = int(running_requests)
        self.last_waiting_requests = int(waiting_requests)

    def record_cuda_graph_capture(
        self,
        delta_bytes: int,
        *,
        batch_descriptor: Any | None = None,
        runtime_mode: str | None = None,
    ) -> None:
        delta_bytes = max(int(delta_bytes), 0)
        if delta_bytes <= 0:
            return
        self.cuda_graph_capture_bytes += delta_bytes
        self.cuda_graph_capture_events += 1
        self._mark_event("cuda_graph_capture")

    def finalize_cuda_graph_capture(self, total_bytes: int) -> None:
        total_bytes = max(int(total_bytes), 0)
        if total_bytes > self.cuda_graph_capture_bytes:
            self.cuda_graph_capture_bytes = total_bytes
        if total_bytes > 0:
            self._mark_event("cuda_graph_capture_finished")

    def collect_state(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "worker_rank": self.rank,
                "local_rank": self.local_rank,
            }

        assert self._pynvml is not None and self._nvml_handle is not None
        memory_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
        torch_allocated = _to_int(torch.cuda.memory_allocated(self.device))
        torch_reserved = _to_int(torch.cuda.memory_reserved(self.device))
        torch_active = _to_int(
            torch.cuda.memory_stats(self.device).get("active_bytes.all.current", 0)
        )
        torch_inactive_split = _to_int(
            torch.cuda.memory_stats(self.device).get(
                "inactive_split_bytes.all.current", 0
            )
        )

        return {
            "enabled": True,
            "timestamp": time.time(),
            "worker_rank": self.rank,
            "local_rank": self.local_rank,
            "gpu_id": self.device.index,
            "physical_gpu_id": self.physical_gpu_id,
            "gpu_uuid": self.gpu_uuid,
            "name": self.gpu_name,
            "total_memory_bytes": int(memory_info.total),
            "used_memory_bytes": int(memory_info.used),
            "free_memory_bytes": int(memory_info.free),
            "torch_allocated_bytes": torch_allocated,
            "torch_reserved_bytes": torch_reserved,
            "torch_active_bytes": torch_active,
            "torch_inactive_split_bytes": torch_inactive_split,
            "non_torch_used_bytes": max(int(memory_info.used) - torch_reserved, 0),
            "model_weights_bytes": self.model_weights_bytes,
            "model_weights_source": self.model_weights_source,
            "kv_cache_reserved_bytes": self.kv_cache_reserved_bytes,
            "kv_cache_num_blocks": self.kv_cache_num_blocks,
            "cuda_graph_bytes": self.cuda_graph_capture_bytes,
            "cuda_graph_capture_events": self.cuda_graph_capture_events,
            "worker_event_seq": self.worker_event_seq,
            "last_event": self.last_event,
            "last_event_ts": self.last_event_ts,
            "last_forward_pass": self.last_forward_pass,
            "last_num_scheduled_tokens": self.last_num_scheduled_tokens,
            "last_running_requests": self.last_running_requests,
            "last_waiting_requests": self.last_waiting_requests,
            "sampling_method": self._sampling_method,
        }
