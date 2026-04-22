# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import html
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from vllm.request_timeline import request_timeline_store

router = APIRouter()


def _merge_snapshots(frontend: dict[str, Any], engine: dict[str, Any] | None) -> dict:
    records: dict[str, dict[str, Any]] = {}
    alias_to_request_id: dict[str, str] = {}

    def add(snapshot: dict[str, Any], source: str) -> None:
        for record in snapshot.get("requests", []):
            request_id = record["request_id"]
            aliases = [request_id, *record.get("aliases", [])]
            request_id = next(
                (
                    alias_to_request_id[alias]
                    for alias in aliases
                    if alias in alias_to_request_id
                ),
                request_id,
            )
            merged = records.setdefault(
                request_id,
                {
                    **record,
                    "request_id": request_id,
                    "events": [],
                    "sources": [],
                },
            )
            merged_aliases = merged.setdefault("aliases", [])
            for alias in aliases:
                alias_to_request_id[alias] = request_id
                if alias != request_id and alias not in merged_aliases:
                    merged_aliases.append(alias)
            merged["sources"].append(source)
            merged["arrival_time"] = min(
                merged.get("arrival_time") or record["arrival_time"],
                record["arrival_time"],
            )
            finish_time = record.get("finish_time")
            if finish_time is not None:
                merged["finish_time"] = max(
                    merged.get("finish_time") or finish_time, finish_time
                )
            if record.get("prompt") and not merged.get("prompt"):
                merged["prompt"] = record["prompt"]
            if merged.get("input_tokens") is None:
                merged["input_tokens"] = record.get("input_tokens")
            if record.get("output"):
                merged["output"] = record["output"]
            if record.get("status") == "finished":
                merged["status"] = "finished"
            elif merged.get("status") != "finished":
                merged["status"] = record.get("status", merged.get("status"))
            for event in record.get("events", []):
                merged["events"].append({**event, "source": source})

    add(frontend, "frontend")
    if engine:
        add(engine, "engine")

    requests = sorted(records.values(), key=lambda item: item.get("arrival_time", 0))
    return {
        "requests": requests,
        "snapshot_time": max(
            frontend.get("snapshot_time", 0),
            (engine or {}).get("snapshot_time", 0),
        ),
    }


async def _get_engine_snapshot(request: Request) -> dict[str, Any] | None:
    engine_client = getattr(request.app.state, "engine_client", None)
    if engine_client is None:
        return None
    getter = getattr(engine_client, "get_request_timeline_engine_snapshot", None)
    if getter is None:
        return None
    try:
        return await getter()
    except Exception:
        return None


@router.get("/request-timeline", include_in_schema=False)
async def request_timeline_index() -> HTMLResponse:
    return HTMLResponse(_render_html())


@router.get("/request-timeline/data", include_in_schema=False)
async def request_timeline_data(request: Request) -> JSONResponse:
    frontend = request_timeline_store.snapshot()
    engine = await _get_engine_snapshot(request)
    return JSONResponse(_merge_snapshots(frontend, engine))


def attach_router(app: FastAPI) -> None:
    app.include_router(router)


def _render_html() -> str:
    title = html.escape("vLLM Request Timeline")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #101214;
      --panel: #181b1f;
      --line: #303741;
      --fg: #eef2f5;
      --muted: #98a4af;
      --accent: #38a169;
      --ingress: #5b8def;
      --queue: #e2b93b;
      --batch: #8b5cf6;
      --batch-queue: #38bdf8;
      --prefill: #16a3a3;
      --decode: #f97316;
      --sample: #d946ef;
      --stream: #22c55e;
      --finish: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font: 13px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: #14171a;
    }}
    h1 {{ margin: 0; font-size: 17px; font-weight: 650; }}
    #status {{ color: var(--muted); }}
    main {{
      height: calc(100vh - 52px);
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      min-width: 900px;
    }}
    aside {{
      border-right: 1px solid var(--line);
      overflow: auto;
      background: #121518;
    }}
    .req-row {{
      width: 100%;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 4px 8px;
      padding: 10px 12px;
      border: 0;
      border-bottom: 1px solid #242a31;
      background: transparent;
      color: var(--fg);
      text-align: left;
      cursor: pointer;
    }}
    .req-row:hover, .req-row.active {{ background: #1d232a; }}
    .req-id {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .req-meta {{ color: var(--muted); font-size: 12px; }}
    .status-pill {{ color: var(--accent); font-size: 12px; }}
    section {{
      min-width: 0;
      overflow: auto;
      padding: 14px 16px 24px;
    }}
    .details {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .field {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: var(--panel);
      min-width: 0;
    }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .text-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }}
    pre {{
      height: 118px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 0;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: #dce5ea;
    }}
    .timeline-wrap {{
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      overflow-x: auto;
      overflow-y: hidden;
      min-height: 430px;
    }}
    svg {{ display: block; min-height: 430px; }}
    .empty {{ color: var(--muted); padding: 24px; }}
    .axis {{ fill: var(--muted); font-size: 11px; }}
    .stage-label {{ fill: #cbd5dd; font-size: 12px; }}
    .event-label {{ fill: #0b0d0f; font-size: 10px; pointer-events: none; }}
    .tooltip {{
      position: fixed;
      z-index: 20;
      display: none;
      max-width: 360px;
      padding: 8px 10px;
      border: 1px solid #3a4652;
      border-radius: 6px;
      background: #0f1317;
      color: #e5edf2;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
      white-space: pre-line;
      pointer-events: none;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div id="status">connecting...</div>
  </header>
  <main>
    <aside id="list"><div class="empty">Waiting for requests...</div></aside>
    <section>
      <div id="details" class="empty">Select a request.</div>
      <div id="timeline" class="timeline-wrap"></div>
    </section>
  </main>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const COLORS = {{
      preprocess: 'var(--ingress)',
      InputProcessor: 'var(--ingress)',
      zmq_send: 'var(--ingress)',
      engine_recv: 'var(--ingress)',
      scheduler_waiting: 'var(--queue)',
      scheduled_batch: 'var(--batch)',
      cudagraph_decision: 'var(--batch)',
      prefill: 'var(--prefill)',
      decode: 'var(--decode)',
      sample: 'var(--sample)',
      OutputProcessor: 'var(--sample)',
      stream_chunk: 'var(--stream)',
      finish: 'var(--finish)',
    }};
    const STAGES = ['preprocess', 'InputProcessor', 'zmq_send', 'engine_recv', 'scheduler_waiting', 'scheduled_batch', 'cudagraph_decision', 'prefill', 'decode', 'sample', 'OutputProcessor', 'stream_chunk', 'finish'];
    const TIME_SCALE_PX_PER_S = 18000;
    const INSTANT_EVENT_WIDTH = 3;
    const MIN_DURATION_WIDTH = 1;
    let data = {{ requests: [] }};
    let selectedId = null;
    let tooltipPinned = false;
    let timelineDirty = false;

    function fmt(ts) {{
      if (!ts) return '-';
      const d = new Date(ts * 1000);
      return d.toLocaleTimeString() + '.' + String(d.getMilliseconds()).padStart(3, '0');
    }}
    function dur(a, b) {{
      if (!a || !b) return '-';
      return ((b - a) * 1000).toFixed(1) + ' ms';
    }}
    function ms(v) {{
      if (v == null || !Number.isFinite(v)) return '-';
      return (v * 1000).toFixed(1) + ' ms';
    }}
    function metrics(req) {{
      const chunks = (req.events || [])
        .filter(e => e.stage === 'stream_chunk')
        .sort((a, b) => a.start_time - b.start_time);
      const tokenChunks = chunks.filter(e => Number(e.metadata?.tokens || 0) > 0);
      const firstToken = (tokenChunks[0] || chunks[0])?.start_time || null;
      const totalTokens = tokenChunks.reduce((sum, e) => sum + Number(e.metadata?.tokens || 0), 0);
      const inputTokens = Math.max(
        0,
        Number(req.input_tokens || 0),
        ...(req.events || []).map(e => Number(e.metadata?.input_tokens || 0))
      );
      const total = req.finish_time ? req.finish_time - req.arrival_time : null;
      const ttft = firstToken ? firstToken - req.arrival_time : null;
      const tpot = firstToken && req.finish_time && totalTokens > 1
        ? (req.finish_time - firstToken) / (totalTokens - 1)
        : null;
      return {{ total, ttft, tpot, totalTokens, inputTokens }};
    }}
    function esc(s) {{
      return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}
    function tooltipText(e) {{
      const meta = e.metadata || {{}};
      const lines = [
        `stage: ${{e.stage}}`,
        `start: ${{fmt(e.start_time)}}`,
        `end: ${{fmt(e.end_time)}}`,
        `duration: ${{meta.instant ? 'instant' : dur(e.start_time, e.end_time)}}`,
      ];
      if (meta.batch_id !== undefined && meta.batch_id !== null) lines.push(`batch: #${{meta.batch_id}}`);
      if (meta.phase) lines.push(`phase: ${{meta.phase}}`);
      if (meta.tokens !== undefined) lines.push(`tokens: ${{meta.tokens}}`);
      if (meta.batch_size !== undefined) lines.push(`batch size: ${{meta.batch_size}}`);
      if (meta.total_batch_tokens !== undefined) lines.push(`batch tokens: ${{meta.total_batch_tokens}}`);
      if (meta.cudagraph_mode !== undefined) lines.push(`cudagraph: ${{meta.cudagraph_mode}}`);
      if (meta.batch_descriptor !== undefined) lines.push(`descriptor: ${{meta.batch_descriptor}}`);
      if (meta.should_ubatch !== undefined) lines.push(`ubatch: ${{meta.should_ubatch}}`);
      if (meta.source) lines.push(`source: ${{meta.source}}`);
      return lines.join('\\n');
    }}
    function attachTooltips(root) {{
      const tip = document.getElementById('tooltip');
      root.querySelectorAll('[data-tip]').forEach(el => {{
        el.addEventListener('mouseenter', () => {{
          tooltipPinned = true;
        }});
        el.addEventListener('mousemove', ev => {{
          tip.textContent = el.dataset.tip || '';
          tip.style.display = 'block';
          tip.style.left = `${{ev.clientX + 14}}px`;
          tip.style.top = `${{ev.clientY + 14}}px`;
        }});
        el.addEventListener('mouseleave', () => {{
          tooltipPinned = false;
          tip.style.display = 'none';
          tip.textContent = '';
          if (timelineDirty) {{
            timelineDirty = false;
            const req = data.requests.find(r => r.request_id === selectedId);
            renderTimeline(req);
          }}
        }});
      }});
    }}
    function renderList() {{
      const list = document.getElementById('list');
      if (!data.requests.length) {{
        list.innerHTML = '<div class="empty">Waiting for requests...</div>';
        return;
      }}
      if (!selectedId || !data.requests.some(r => r.request_id === selectedId)) {{
        selectedId = data.requests[0].request_id;
      }}
      list.innerHTML = data.requests.map(r => `
        <button class="req-row ${{r.request_id === selectedId ? 'active' : ''}}" data-id="${{esc(r.request_id)}}">
          <div class="req-id">${{esc(r.request_id)}}</div>
          <div class="status-pill">${{esc(r.status || 'running')}}</div>
          <div class="req-meta">${{fmt(r.arrival_time)}}</div>
          <div class="req-meta">${{dur(r.arrival_time, r.finish_time || data.snapshot_time)}}</div>
        </button>`).join('');
      list.querySelectorAll('.req-row').forEach(btn => btn.onclick = () => {{
        selectedId = btn.dataset.id;
        timelineDirty = false;
        render(true);
      }});
    }}
    function renderDetails(req) {{
      const details = document.getElementById('details');
      if (!req) {{
        details.className = 'empty';
        details.innerHTML = 'Select a request.';
        return;
      }}
      details.className = '';
      const m = metrics(req);
      details.innerHTML = `
        <div class="details">
          <div class="field"><div class="label">Request ID</div><div class="value">${{esc(req.request_id)}}</div></div>
          <div class="field"><div class="label">Arrival</div><div class="value">${{fmt(req.arrival_time)}}</div></div>
          <div class="field"><div class="label">Finish</div><div class="value">${{fmt(req.finish_time)}}</div></div>
          <div class="field"><div class="label">Status</div><div class="value">${{esc(req.status)}}</div></div>
          <div class="field"><div class="label">Input Tokens</div><div class="value">${{m.inputTokens || '-'}}</div></div>
          <div class="field"><div class="label">Output Tokens</div><div class="value">${{m.totalTokens || '-'}}</div></div>
          <div class="field"><div class="label">Total Latency</div><div class="value">${{ms(m.total)}}</div></div>
          <div class="field"><div class="label">TTFT</div><div class="value">${{ms(m.ttft)}}</div></div>
          <div class="field"><div class="label">TPOT</div><div class="value">${{ms(m.tpot)}}</div></div>
        </div>
        <div class="text-grid">
          <pre>${{esc(req.prompt || '')}}</pre>
          <pre>${{esc(req.output || '')}}</pre>
        </div>`;
    }}
    function renderTimeline(req, force = false) {{
      const root = document.getElementById('timeline');
      const tip = document.getElementById('tooltip');
      if (tooltipPinned && !force) {{
        timelineDirty = true;
        return;
      }}
      tooltipPinned = false;
      tip.style.display = 'none';
      tip.textContent = '';
      if (!req) {{
        root.innerHTML = '';
        return;
      }}
      const events = [...(req.events || [])].sort((a, b) => a.start_time - b.start_time);
      if (!events.length) {{
        root.innerHTML = '<div class="empty">No timeline events yet.</div>';
        return;
      }}
      const minT = Math.min(req.arrival_time, ...events.map(e => e.start_time));
      const maxT = Math.max(req.finish_time || data.snapshot_time || minT, ...events.map(e => e.end_time || e.start_time));
      const span = Math.max(maxT - minT, 0.01);
      const width = Math.max(1100, span * TIME_SCALE_PX_PER_S);
      const left = 150, top = 42, rowH = 34, h = top + STAGES.length * rowH + 45;
      const x = t => left + ((t - minT) / span) * (width - left - 30);
      const rows = new Map(STAGES.map((s, i) => [s, top + i * rowH]));
      let svg = `<svg width="${{width}}" height="${{h}}" viewBox="0 0 ${{width}} ${{h}}">`;
      svg += `<text class="axis" x="${{left}}" y="22">${{fmt(minT)}}</text><text class="axis" x="${{width - 160}}" y="22">${{fmt(maxT)}}</text>`;
      for (const stage of STAGES) {{
        const y = rows.get(stage);
        svg += `<text class="stage-label" x="14" y="${{y + 18}}">${{stage}}</text>`;
        svg += `<line x1="${{left}}" y1="${{y + 12}}" x2="${{width - 24}}" y2="${{y + 12}}" stroke="#2a3038"/>`;
      }}
      for (const e of events) {{
        const stage = STAGES.includes(e.stage) ? e.stage : 'finish';
        const y = rows.get(stage) + 3;
        const sx = x(e.start_time);
        const instant = e.metadata?.instant || !e.end_time || e.end_time <= e.start_time;
        const ex = x(instant ? e.start_time : e.end_time);
        const w = instant
          ? INSTANT_EVENT_WIDTH
          : Math.max(MIN_DURATION_WIDTH, ex - sx);
        svg += `<rect class="event-rect" data-tip="${{esc(tooltipText(e))}}" x="${{sx}}" y="${{y}}" width="${{w}}" height="18" rx="3" fill="${{COLORS[stage] || '#94a3b8'}}"></rect>`;
        if (w > 46) svg += `<text class="event-label" x="${{sx + 5}}" y="${{y + 13}}">${{esc(stage)}}</text>`;
      }}
      svg += '</svg>';
      root.innerHTML = svg;
      attachTooltips(root);
    }}
    function render(forceTimeline = false) {{
      renderList();
      const req = data.requests.find(r => r.request_id === selectedId);
      renderDetails(req);
      renderTimeline(req, forceTimeline);
    }}
    async function refresh() {{
      try {{
        const resp = await fetch('/request-timeline/data', {{ cache: 'no-store' }});
        data = await resp.json();
        document.getElementById('status').textContent = `${{data.requests.length}} requests`;
        render();
      }} catch (err) {{
        document.getElementById('status').textContent = 'disconnected';
      }}
    }}
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>"""
