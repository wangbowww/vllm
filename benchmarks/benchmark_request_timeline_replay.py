# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproducible ShareGPT Poisson replay benchmark for request timelines.

This script is intentionally smaller than ``vllm bench serve``. It exists to
drive the request-timeline UI with a stable online workload:

* sample a fixed set of ShareGPT requests;
* generate Poisson arrival offsets once;
* save the replay schedule and reuse it on later runs with the same arguments;
* send stable x-request-id values so timeline rows are easy to compare.

Example:
python benchmarks/benchmark_request_timeline_replay.py \
  --base-url http://127.0.0.1:8000 \
  --model auto \
  --tokenizer /data1/models/Qwen3-0.6B \
  --dataset-path /data1/zzh/dataset/raw/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-requests 64 \
  --request-rate 4 \
  --max-output-len 256 \
  --ignore-eos \
  --wait-ready \
  --schedule-path /tmp/qwen3_0p6b_timeline_poisson_64req_4rps_256out_seed0.json \
  --result-out /tmp/qwen3_0p6b_timeline_replay_result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

# Allow ``python benchmarks/benchmark_request_timeline_replay.py`` from a
# source checkout even when vLLM is not installed as an editable package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm.benchmarks.datasets import SampleRequest, ShareGPTDataset
from vllm.benchmarks.lib.endpoint_request_func import (
    AIOHTTP_TIMEOUT,
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
    RequestFuncOutput,
)
from vllm.tokenizers import get_tokenizer


DEFAULT_MODEL_NAME = "auto"
DEFAULT_TOKENIZER_PATH = "/data1/models/Qwen3-0.6B"
DEFAULT_SCHEDULE_PATH = "/tmp/vllm_request_timeline_replay_schedule.json"
SCHEDULE_VERSION = 1

LOCAL_SHAREGPT_CANDIDATES = (
    "/data1/zzh/dataset/raw/ShareGPT_V3_unfiltered_cleaned_split.json",
    "/data1/mt/dataset/ShareGPT52K",
    "/data1/mt/dataset/test/ShareGPT52K",
    "/data1/sn/Hydra/llm_judge/data/sharegpt",
)


@dataclass(frozen=True)
class ScheduledRequest:
    index: int
    request_id: str
    arrival_offset_s: float
    prompt_len: int
    output_len: int
    prompt_preview: str


def _discover_sharegpt_dataset() -> str | None:
    for candidate in LOCAL_SHAREGPT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _api_url(base_url: str, endpoint_type: str) -> str:
    base_url = base_url.rstrip("/")
    if endpoint_type == "openai-chat":
        return f"{base_url}/v1/chat/completions"
    if endpoint_type in ("openai", "vllm"):
        return f"{base_url}/v1/completions"
    raise ValueError(f"Unsupported endpoint type: {endpoint_type}")


async def _resolve_model_name(
    *,
    base_url: str,
    model: str,
    session: aiohttp.ClientSession,
    wait_ready: bool,
    ready_timeout: float,
) -> str:
    if model != "auto":
        return model

    models_url = f"{base_url.rstrip('/')}/v1/models"
    deadline = time.perf_counter() + ready_timeout
    last_error = ""
    while True:
        try:
            async with session.get(models_url) as response:
                if response.status == 200:
                    payload = await response.json()
                    models = payload.get("data") or []
                    if models and models[0].get("id"):
                        return str(models[0]["id"])
                    last_error = f"no served models in response: {payload}"
                else:
                    last_error = f"HTTP {response.status} {response.reason}"
        except Exception as exc:
            last_error = repr(exc)

        if not wait_ready or time.perf_counter() >= deadline:
            raise RuntimeError(
                f"Failed to discover served model from {models_url}: {last_error}"
            )
        await asyncio.sleep(1)


def _finite_request_rate(value: str) -> float:
    if value == "inf":
        return float("inf")
    request_rate = float(value)
    if request_rate <= 0:
        raise argparse.ArgumentTypeError("--request-rate must be positive or 'inf'")
    return request_rate


def _generate_arrival_offsets(
    *,
    num_requests: int,
    request_rate: float,
    seed: int,
    normalize: bool,
) -> list[float]:
    if request_rate == float("inf"):
        return [0.0] * num_requests

    rng = np.random.default_rng(seed)
    intervals = rng.exponential(1.0 / request_rate, size=num_requests)
    offsets = np.cumsum(intervals)

    if normalize and len(offsets) > 0 and offsets[-1] > 0:
        target_total_delay_s = num_requests / request_rate
        offsets = offsets * (target_total_delay_s / offsets[-1])

    return [float(offset) for offset in offsets]


def _prompt_preview(prompt: str | list[str], limit: int = 120) -> str:
    if isinstance(prompt, list):
        text = "\n".join(str(item) for item in prompt)
    else:
        text = prompt
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _load_sharegpt_requests(args: argparse.Namespace) -> list[SampleRequest]:
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    dataset = ShareGPTDataset(
        random_seed=args.seed,
        dataset_path=args.dataset_path,
        disable_shuffle=args.disable_shuffle,
    )
    return dataset.sample(
        tokenizer=tokenizer,
        num_requests=args.num_requests,
        output_len=args.max_output_len,
        request_id_prefix=f"{args.request_id_prefix}-",
        no_oversample=args.no_oversample,
    )


def _schedule_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": SCHEDULE_VERSION,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "disable_shuffle": args.disable_shuffle,
        "num_requests": args.num_requests,
        "request_rate": args.request_rate
        if args.request_rate != float("inf")
        else "inf",
        "max_output_len": args.max_output_len,
        "seed": args.seed,
        "normalize_arrivals": args.normalize_arrivals,
        "request_id_prefix": args.request_id_prefix,
    }


def _build_schedule(
    requests: list[SampleRequest],
    *,
    args: argparse.Namespace,
) -> list[ScheduledRequest]:
    offsets = _generate_arrival_offsets(
        num_requests=len(requests),
        request_rate=args.request_rate,
        seed=args.seed,
        normalize=args.normalize_arrivals,
    )

    schedule: list[ScheduledRequest] = []
    for index, (request, offset) in enumerate(zip(requests, offsets)):
        request_id = f"{args.request_id_prefix}-{args.seed:04d}-{index:05d}"
        request.request_id = request_id
        schedule.append(
            ScheduledRequest(
                index=index,
                request_id=request_id,
                arrival_offset_s=offset,
                prompt_len=request.prompt_len,
                output_len=request.expected_output_len,
                prompt_preview=_prompt_preview(request.prompt),
            )
        )
    return schedule


def _load_schedule(
    path: Path,
    expected_metadata: dict[str, Any],
) -> list[ScheduledRequest]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    if metadata != expected_metadata:
        raise RuntimeError(
            "Existing replay schedule was created with different arguments.\n"
            f"  path: {path}\n"
            f"  expected: {json.dumps(expected_metadata, ensure_ascii=False)}\n"
            f"  found:    {json.dumps(metadata, ensure_ascii=False)}\n"
            "Use --regenerate-schedule or pass a different --schedule-path."
        )
    return [ScheduledRequest(**item) for item in payload["schedule"]]


def _save_schedule(
    path: Path,
    metadata: dict[str, Any],
    schedule: list[ScheduledRequest],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "schedule": [asdict(item) for item in schedule],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_or_create_schedule(
    requests: list[SampleRequest],
    args: argparse.Namespace,
) -> tuple[list[ScheduledRequest], str]:
    metadata = _schedule_metadata(args)
    schedule_path = Path(args.schedule_path) if args.schedule_path else None
    if schedule_path and schedule_path.exists() and not args.regenerate_schedule:
        schedule = _load_schedule(schedule_path, metadata)
        source = "loaded"
    else:
        schedule = _build_schedule(requests, args=args)
        source = "generated"
        if schedule_path:
            _save_schedule(schedule_path, metadata, schedule)

    if len(schedule) != len(requests):
        raise RuntimeError(
            f"Schedule has {len(schedule)} requests but dataset produced "
            f"{len(requests)} requests."
        )

    for request, scheduled in zip(requests, schedule):
        request.request_id = scheduled.request_id
        request.expected_output_len = scheduled.output_len

    return schedule, source


async def _send_one(
    *,
    sample: SampleRequest,
    scheduled: ScheduledRequest,
    api_url: str,
    model: str,
    endpoint_type: str,
    session: aiohttp.ClientSession,
    start_time: float,
    semaphore: asyncio.Semaphore | None,
    pbar: tqdm,
    extra_body: dict[str, Any] | None,
    ignore_eos: bool,
) -> RequestFuncOutput:
    sleep_s = start_time + scheduled.arrival_offset_s - time.perf_counter()
    if sleep_s > 0:
        await asyncio.sleep(sleep_s)

    request_input = RequestFuncInput(
        prompt=sample.prompt,
        api_url=api_url,
        prompt_len=sample.prompt_len,
        output_len=sample.expected_output_len,
        model=model,
        model_name=model,
        extra_body=extra_body,
        multi_modal_content=sample.multi_modal_data,
        ignore_eos=ignore_eos,
        request_id=scheduled.request_id,
    )
    request_func = ASYNC_REQUEST_FUNCS[endpoint_type]

    if semaphore is None:
        return await request_func(request_input, session, pbar)
    async with semaphore:
        return await request_func(request_input, session, pbar)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), percentile))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values))) if values else 0.0


def _tpot(output: RequestFuncOutput) -> float:
    return float(np.mean(np.asarray(output.itl))) if output.itl else 0.0


def _summarize(
    *,
    outputs: list[RequestFuncOutput],
    schedule: list[ScheduledRequest],
    total_elapsed_s: float,
    args: argparse.Namespace,
    schedule_source: str,
) -> dict[str, Any]:
    successes = [output for output in outputs if output.success]
    failures = [output for output in outputs if not output.success]
    latencies = [output.latency for output in successes]
    ttfts = [output.ttft for output in successes]
    tpots = [_tpot(output) for output in successes]
    output_tokens = [output.output_tokens for output in successes]

    return {
        "benchmark": "request_timeline_replay",
        "model": args.model,
        "endpoint_type": args.endpoint_type,
        "base_url": args.base_url,
        "dataset_path": args.dataset_path,
        "schedule_path": args.schedule_path,
        "schedule_source": schedule_source,
        "num_requests": args.num_requests,
        "request_rate": args.request_rate
        if args.request_rate != float("inf")
        else "inf",
        "max_output_len": args.max_output_len,
        "seed": args.seed,
        "normalize_arrivals": args.normalize_arrivals,
        "success_count": len(successes),
        "failure_count": len(failures),
        "total_elapsed_s": total_elapsed_s,
        "p50_latency_s": _percentile(latencies, 50),
        "p90_latency_s": _percentile(latencies, 90),
        "p99_latency_s": _percentile(latencies, 99),
        "p50_ttft_s": _percentile(ttfts, 50),
        "p90_ttft_s": _percentile(ttfts, 90),
        "p99_ttft_s": _percentile(ttfts, 99),
        "mean_tpot_s": _mean(tpots),
        "total_output_tokens": int(sum(token or 0 for token in output_tokens)),
        "schedule": [asdict(item) for item in schedule],
        "requests": [
            {
                "request_id": scheduled.request_id,
                "success": output.success,
                "latency_s": output.latency,
                "ttft_s": output.ttft,
                "tpot_s": _tpot(output),
                "prompt_len": output.prompt_len,
                "output_tokens": output.output_tokens,
                "arrival_offset_s": scheduled.arrival_offset_s,
                "error": output.error,
            }
            for scheduled, output in zip(schedule, outputs)
        ],
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.endpoint_type not in ("openai-chat", "openai", "vllm"):
        raise ValueError("Only openai-chat, openai, and vllm endpoints are supported.")

    input_requests = _load_sharegpt_requests(args)
    schedule, schedule_source = _get_or_create_schedule(input_requests, args)

    api_url = _api_url(args.base_url, args.endpoint_type)
    connector = aiohttp.TCPConnector(limit=args.max_concurrency or 0)
    semaphore = (
        asyncio.Semaphore(args.max_concurrency)
        if args.max_concurrency is not None
        else None
    )
    extra_body = json.loads(args.extra_body) if args.extra_body else None

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=AIOHTTP_TIMEOUT,
    ) as session:
        args.model = await _resolve_model_name(
            base_url=args.base_url,
            model=args.model,
            session=session,
            wait_ready=args.wait_ready,
            ready_timeout=args.ready_timeout,
        )

        print("Request timeline replay benchmark")
        print(f"  endpoint:        {api_url}")
        print(f"  model:           {args.model}")
        print(f"  dataset:         {args.dataset_path}")
        print(f"  requests:        {len(input_requests)}")
        print(f"  request_rate:    {args.request_rate}")
        print(f"  max_output_len:  {args.max_output_len}")
        print(f"  seed:            {args.seed}")
        print(f"  schedule:        {args.schedule_path} ({schedule_source})")
        print(f"  first offset:    {schedule[0].arrival_offset_s:.4f}s")
        print(f"  last offset:     {schedule[-1].arrival_offset_s:.4f}s")
        print(f"  max concurrency: {args.max_concurrency or 'unbounded'}")
        print(f"  timeline UI:     {args.base_url.rstrip('/')}/request-timeline")

        benchmark_start = time.perf_counter()
        pbar = tqdm(total=len(input_requests), desc="replay", unit="req")
        tasks = [
            asyncio.create_task(
                _send_one(
                    sample=sample,
                    scheduled=scheduled,
                    api_url=api_url,
                    model=args.model,
                    endpoint_type=args.endpoint_type,
                    session=session,
                    start_time=benchmark_start,
                    semaphore=semaphore,
                    pbar=pbar,
                    extra_body=extra_body,
                    ignore_eos=args.ignore_eos,
                )
            )
            for sample, scheduled in zip(input_requests, schedule)
        ]
        outputs = await asyncio.gather(*tasks)
        pbar.close()

    total_elapsed_s = time.perf_counter() - benchmark_start
    summary = _summarize(
        outputs=outputs,
        schedule=schedule,
        total_elapsed_s=total_elapsed_s,
        args=args,
        schedule_source=schedule_source,
    )

    if args.result_out:
        Path(args.result_out).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("\nSummary")
    print(f"  success/failure: {summary['success_count']}/{summary['failure_count']}")
    print(f"  elapsed:         {summary['total_elapsed_s']:.3f}s")
    print(f"  p50 latency:     {summary['p50_latency_s']:.3f}s")
    print(f"  p90 latency:     {summary['p90_latency_s']:.3f}s")
    print(f"  p50 TTFT:        {summary['p50_ttft_s']:.3f}s")
    print(f"  p90 TTFT:        {summary['p90_ttft_s']:.3f}s")
    print(f"  mean TPOT:       {summary['mean_tpot_s']:.3f}s")
    if summary["failure_count"]:
        first_failure = next(item for item in summary["requests"] if not item["success"])
        print("  first failure:   " + first_failure["error"][:200])

    return summary


def parse_args() -> argparse.Namespace:
    discovered_dataset = _discover_sharegpt_dataset()
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible ShareGPT Poisson replay benchmark for the vLLM "
            "request timeline profiler."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Served model name. Defaults to 'auto', which uses the first id "
            "returned by /v1/models."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_PATH,
        help="Tokenizer path/name used to measure ShareGPT prompt lengths.",
    )
    parser.add_argument(
        "--endpoint-type",
        choices=["openai-chat", "openai", "vllm"],
        default="openai-chat",
    )
    parser.add_argument(
        "--dataset-path",
        default=discovered_dataset,
        help=(
            "Path to a ShareGPT JSON dataset. By default this script uses the "
            "first local ShareGPT path it can find."
        ),
    )
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument(
        "--request-rate",
        type=_finite_request_rate,
        default=4.0,
        help="Poisson request rate in requests/sec, or 'inf' for all-at-once.",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=256,
        help="Fixed max generation tokens for every request.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ask the server to ignore EOS so requests tend to hit max-output-len.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--request-id-prefix",
        default="timeline-replay",
        help="Stable prefix used for x-request-id values.",
    )
    parser.add_argument(
        "--schedule-path",
        default=DEFAULT_SCHEDULE_PATH,
        help=(
            "JSON file for the replay schedule. If it exists and arguments "
            "match, it is reused; otherwise it is created."
        ),
    )
    parser.add_argument(
        "--regenerate-schedule",
        action="store_true",
        help="Overwrite --schedule-path with a newly generated Poisson schedule.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Optional client-side concurrency cap.",
    )
    parser.add_argument(
        "--disable-shuffle",
        action="store_true",
        help="Keep ShareGPT file order instead of seed-shuffling it.",
    )
    parser.add_argument(
        "--no-oversample",
        action="store_true",
        help="Do not repeat samples if fewer valid ShareGPT rows are available.",
    )
    parser.add_argument(
        "--no-normalize-arrivals",
        dest="normalize_arrivals",
        action="store_false",
        help=(
            "Do not rescale sampled arrivals to exactly "
            "num_requests/request_rate."
        ),
    )
    parser.set_defaults(normalize_arrivals=True)
    parser.add_argument(
        "--extra-body",
        default=None,
        help="JSON object merged into every OpenAI request body.",
    )
    parser.add_argument(
        "--result-out",
        default=None,
        help="Optional JSON file containing metrics and the replay schedule.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--wait-ready",
        action="store_true",
        help="Poll /v1/models until the server is ready before replay starts.",
    )
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    args = parser.parse_args()

    if not args.dataset_path:
        parser.error(
            "--dataset-path is required because no local ShareGPT dataset was found"
        )
    if args.num_requests <= 0:
        parser.error("--num-requests must be positive")
    if args.max_output_len <= 0:
        parser.error("--max-output-len must be positive")
    if args.max_concurrency is not None and args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive when set")
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError as exc:
            parser.error(f"--extra-body must be valid JSON: {exc}")
        if not isinstance(extra_body, dict):
            parser.error("--extra-body must decode to a JSON object")
    if math.isinf(args.request_rate) and args.normalize_arrivals:
        args.normalize_arrivals = False
    return args


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
