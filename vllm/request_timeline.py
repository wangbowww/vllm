# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import threading
import time
from collections import OrderedDict
from typing import Any

_MAX_REQUESTS = 1000
_MAX_TEXT_CHARS = 12000


def now() -> float:
    return time.time()


def _trim_text(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= _MAX_TEXT_CHARS:
        return text
    return text[-_MAX_TEXT_CHARS:]


class RequestTimelineStore:
    """Small in-memory request timeline store.

    vLLM serve usually has a frontend process and one or more engine-core
    processes, so each process owns a local store. The API route merges the
    frontend snapshot with an engine snapshot fetched over existing utility RPC.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._open_events: dict[str, dict[str, str]] = {}
        self._aliases: dict[str, str] = {}
        self._seq = itertools.count()
        self._event_seq = itertools.count()

    def _resolve_id(self, request_id: str) -> str:
        return self._aliases.get(request_id, request_id)

    def _ensure_record(
        self,
        request_id: str,
        *,
        timestamp: float | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        request_id = self._resolve_id(request_id)
        record = self._records.get(request_id)
        ts = now() if timestamp is None else timestamp
        if record is None:
            record = {
                "request_id": request_id,
                "arrival_time": ts,
                "finish_time": None,
                "status": status or "running",
                "prompt": None,
                "output": "",
                "input_tokens": None,
                "events": [],
                "aliases": [],
                "order": next(self._seq),
            }
            self._records[request_id] = record
            self._open_events[request_id] = {}
            while len(self._records) > _MAX_REQUESTS:
                old_id, _ = self._records.popitem(last=False)
                self._open_events.pop(old_id, None)
                self._aliases = {
                    alias: canonical
                    for alias, canonical in self._aliases.items()
                    if canonical != old_id
                }
        elif timestamp is not None:
            record["arrival_time"] = min(record["arrival_time"], timestamp)
        if status is not None:
            record["status"] = status
        return record

    def add_alias(self, canonical_id: str, alias_id: str) -> None:
        if canonical_id == alias_id:
            return
        with self._lock:
            canonical_id = self._resolve_id(canonical_id)
            alias_canonical_id = self._resolve_id(alias_id)
            record = self._ensure_record(canonical_id)
            self._aliases[alias_id] = canonical_id
            aliases = record.setdefault("aliases", [])
            if alias_id not in aliases:
                aliases.append(alias_id)

            if alias_canonical_id == canonical_id:
                return
            alias_record = self._records.pop(alias_canonical_id, None)
            if alias_record is None:
                return
            record["arrival_time"] = min(
                record["arrival_time"], alias_record["arrival_time"]
            )
            record["finish_time"] = max(
                t
                for t in (record.get("finish_time"), alias_record.get("finish_time"))
                if t is not None
            ) if record.get("finish_time") or alias_record.get("finish_time") else None
            if not record.get("prompt") and alias_record.get("prompt"):
                record["prompt"] = alias_record["prompt"]
            if record.get("input_tokens") is None:
                record["input_tokens"] = alias_record.get("input_tokens")
            if alias_record.get("output"):
                record["output"] = alias_record["output"]
            record["events"].extend(alias_record.get("events", []))
            for alias in alias_record.get("aliases", []):
                self._aliases[alias] = canonical_id
                if alias not in aliases:
                    aliases.append(alias)
            self._open_events.setdefault(canonical_id, {}).update(
                self._open_events.pop(alias_canonical_id, {})
            )

    def add_request(
        self,
        request_id: str,
        *,
        timestamp: float | None = None,
        prompt: str | None = None,
        input_tokens: int | None = None,
        status: str | None = None,
    ) -> None:
        with self._lock:
            record = self._ensure_record(
                request_id, timestamp=timestamp, status=status
            )
            if prompt is not None:
                record["prompt"] = _trim_text(prompt)
            if input_tokens is not None:
                record["input_tokens"] = input_tokens

    def update_status(
        self,
        request_id: str,
        status: str,
        *,
        timestamp: float | None = None,
        finish: bool = False,
    ) -> None:
        with self._lock:
            record = self._ensure_record(
                request_id, timestamp=timestamp, status=status
            )
            if finish:
                record["finish_time"] = now() if timestamp is None else timestamp

    def append_output(self, request_id: str, text: str, *, finished: bool = False) -> None:
        if not text and not finished:
            return
        with self._lock:
            record = self._ensure_record(request_id)
            if text:
                record["output"] = _trim_text((record.get("output") or "") + text)
            if finished:
                record["status"] = "finished"
                record["finish_time"] = now()

    def add_event(
        self,
        request_id: str,
        event_name: str,
        start_time: float,
        end_time: float | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            record = self._ensure_record(request_id, timestamp=start_time)
            metadata = dict(metadata or {})
            if end_time is None:
                end_time = start_time
                metadata.setdefault("instant", True)
            elif end_time < start_time:
                end_time = start_time
                metadata.setdefault("instant", True)
            event = {
                "id": f"evt-{next(self._event_seq)}",
                "stage": event_name,
                "start_time": start_time,
                "end_time": end_time,
                "metadata": metadata,
            }
            record["events"].append(event)

    def start_event(
        self,
        request_id: str,
        event_name: str,
        *,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            start_time = now() if timestamp is None else timestamp
            record = self._ensure_record(request_id, timestamp=start_time)
            open_events = self._open_events.setdefault(request_id, {})
            if event_name in open_events:
                return
            event_id = f"evt-{next(self._event_seq)}"
            open_events[event_name] = event_id
            record["events"].append(
                {
                    "id": event_id,
                    "stage": event_name,
                    "start_time": start_time,
                    "end_time": None,
                    "metadata": metadata or {},
                }
            )

    def end_event(
        self,
        request_id: str,
        event_name: str,
        *,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            event_id = self._open_events.get(request_id, {}).pop(event_name, None)
            if event_id is None:
                return
            end_time = now() if timestamp is None else timestamp
            record = self._records.get(request_id)
            if record is None:
                return
            for event in reversed(record["events"]):
                if event["id"] == event_id:
                    event["end_time"] = max(end_time, event["start_time"])
                    if metadata:
                        event["metadata"].update(metadata)
                    return

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = []
            current_time = now()
            for record in self._records.values():
                copied = {
                    **record,
                    "events": [dict(event) for event in record["events"]],
                }
                for event in copied["events"]:
                    if event["end_time"] is None:
                        event["end_time"] = current_time
                records.append(copied)
            return {"requests": records, "snapshot_time": current_time}


request_timeline_store = RequestTimelineStore()
