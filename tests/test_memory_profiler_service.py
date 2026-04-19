# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.memory_profiling.service import build_gpu_profiler_sample


def test_build_gpu_profiler_sample_uses_reserved_kv_and_non_negative_unaccounted():
    worker_state = {
        "worker_rank": 0,
        "local_rank": 0,
        "gpu_id": 0,
        "physical_gpu_id": 0,
        "gpu_uuid": "GPU-0",
        "name": "Test GPU",
        "total_memory_bytes": 1000,
        "used_memory_bytes": 800,
        "free_memory_bytes": 200,
        "torch_allocated_bytes": 650,
        "torch_reserved_bytes": 700,
        "non_torch_used_bytes": 100,
        "model_weights_bytes": 300,
        "model_weights_source": "scan",
        "kv_cache_reserved_bytes": 250,
        "kv_cache_num_blocks": 10,
        "cuda_graph_bytes": 50,
        "cuda_graph_capture_events": 2,
        "worker_event_seq": 7,
        "last_event": "execute_model",
        "sampling_method": "hybrid_nvml_vllm_torch",
    }
    scheduler_state = {
        "iteration": 12,
        "kv_cache_usage_fraction": 0.4,
        "running_requests": 3,
        "waiting_requests": 1,
    }

    sample = build_gpu_profiler_sample(worker_state, scheduler_state)

    assert sample.breakdown.model_weights_bytes == 300
    assert sample.breakdown.kv_cache_bytes == 250
    assert sample.meta.kv_cache_used_bytes == 100
    assert sample.breakdown.fragmentation_bytes == 50
    assert sample.breakdown.activations_bytes == 100
    assert sample.breakdown.unaccounted_bytes == 50


def test_build_gpu_profiler_sample_caps_activation_to_keep_balance_non_negative():
    worker_state = {
        "worker_rank": 0,
        "local_rank": 0,
        "gpu_id": 0,
        "physical_gpu_id": 0,
        "gpu_uuid": "GPU-0",
        "name": "Test GPU",
        "total_memory_bytes": 1000,
        "used_memory_bytes": 500,
        "free_memory_bytes": 500,
        "torch_allocated_bytes": 700,
        "torch_reserved_bytes": 750,
        "non_torch_used_bytes": 0,
        "model_weights_bytes": 250,
        "model_weights_source": "scan",
        "kv_cache_reserved_bytes": 200,
        "kv_cache_num_blocks": 10,
        "cuda_graph_bytes": 40,
        "cuda_graph_capture_events": 1,
        "worker_event_seq": 1,
        "last_event": "poll",
        "sampling_method": "hybrid_nvml_vllm_torch",
    }
    scheduler_state = {
        "iteration": 1,
        "kv_cache_usage_fraction": 0.5,
        "running_requests": 0,
        "waiting_requests": 0,
    }

    sample = build_gpu_profiler_sample(worker_state, scheduler_state)

    assert sample.breakdown.activations_bytes == 10
    assert sample.breakdown.unaccounted_bytes == 0
