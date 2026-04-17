# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Memory profiling package.

Keep this package init side-effect free to avoid circular imports during
startup. Import concrete modules directly, e.g.:

- ``vllm.memory_profiling.runtime``
- ``vllm.memory_profiling.service``
- ``vllm.memory_profiling.worker``
"""

__all__: list[str] = []
