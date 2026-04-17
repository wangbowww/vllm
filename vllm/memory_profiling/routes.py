# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from argparse import Namespace
from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.datastructures import State
from starlette.websockets import WebSocketDisconnect

from vllm.config.memory_profiler import MemoryProfilerConfig
from vllm.engine.protocol import EngineClient
from vllm.memory_profiling.service import MemoryProfilerService

if TYPE_CHECKING:
    from vllm.tasks import SupportedTask


def _normalize_base_path(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return path.rstrip("/")


def _render_html(base_path: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>vLLM GPU Memory Profiler</title>
  <style>
    :root {{
      --bg: #08111b;
      --panel: #102133;
      --panel-border: rgba(170, 196, 219, 0.22);
      --fg: #e8f1f7;
      --muted: #9fb6c8;
      --grid: rgba(255, 255, 255, 0.08);
      --accent: #59c2ff;
      --weights: #2e7d32;
      --kv: #0288d1;
      --act: #ff8f00;
      --graph: #7b1fa2;
      --frag: #c62828;
      --unacc: #607d8b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(89,194,255,0.12), transparent 28%),
        linear-gradient(180deg, #0a1521 0%, #08111b 100%);
      color: var(--fg);
    }}
    header {{
      padding: 20px 24px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      position: sticky;
      top: 0;
      background: rgba(8, 17, 27, 0.92);
      backdrop-filter: blur(8px);
      z-index: 10;
    }}
    h1 {{ margin: 0; font-size: 24px; }}
    .sub {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
    }}
    .status {{
      margin-top: 10px;
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      padding: 18px 18px 32px;
      display: grid;
      gap: 18px;
    }}
    .gpu-panel {{
      background: linear-gradient(180deg, rgba(20, 38, 58, 0.95), rgba(11, 24, 37, 0.97));
      border: 1px solid var(--panel-border);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 14px 50px rgba(0,0,0,0.24);
      position: relative;
    }}
    .gpu-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      align-items: baseline;
      margin-bottom: 12px;
    }}
    .gpu-title {{
      font-size: 18px;
      font-weight: 600;
    }}
    .gpu-meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .stat {{
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.03);
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .stat-value {{
      font-size: 18px;
      font-weight: 600;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    canvas {{
      width: 100%;
      height: 280px;
      display: block;
      border-radius: 14px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      transform: translate(8px, -8px);
      background: rgba(3, 10, 18, 0.96);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 10px;
      padding: 10px 12px;
      color: var(--fg);
      font-size: 12px;
      line-height: 1.45;
      min-width: 220px;
      display: none;
      z-index: 20;
    }}
    .empty {{
      padding: 36px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed rgba(255,255,255,0.16);
      border-radius: 16px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>vLLM GPU Memory Profiler</h1>
    <div class="sub">Reserved KV cache is shown in the stacked chart. Tooltip also shows KV cache used bytes and estimator metadata.</div>
    <div class="status">
      <div id="conn-status">connecting...</div>
      <div id="sample-status">waiting for samples...</div>
      <div>base path: <code>{base_path}</code></div>
    </div>
  </header>
  <main id="app">
    <div class="empty">Waiting for GPU snapshots...</div>
  </main>
  <script>
    const BASE_PATH = {json.dumps(base_path)};
    const COLORS = {{
      model_weights_bytes: getComputedStyle(document.documentElement).getPropertyValue('--weights').trim(),
      kv_cache_bytes: getComputedStyle(document.documentElement).getPropertyValue('--kv').trim(),
      activations_bytes: getComputedStyle(document.documentElement).getPropertyValue('--act').trim(),
      cuda_graph_bytes: getComputedStyle(document.documentElement).getPropertyValue('--graph').trim(),
      fragmentation_bytes: getComputedStyle(document.documentElement).getPropertyValue('--frag').trim(),
      unaccounted_bytes: getComputedStyle(document.documentElement).getPropertyValue('--unacc').trim(),
    }};
    const LABELS = {{
      model_weights_bytes: 'model_weights',
      kv_cache_bytes: 'kv_cache_reserved',
      activations_bytes: 'activations',
      cuda_graph_bytes: 'cuda_graph',
      fragmentation_bytes: 'fragmentation',
      unaccounted_bytes: 'unaccounted',
    }};
    const STACK_KEYS = Object.keys(COLORS);
    const state = {{
      history: [],
      panels: new Map(),
    }};

    const fmtBytes = (value) => {{
      if (value == null) return '-';
      const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
      let v = Math.max(Number(value), 0);
      let idx = 0;
      while (v >= 1024 && idx < units.length - 1) {{
        v /= 1024;
        idx += 1;
      }}
      return `${{v.toFixed(v >= 100 || idx === 0 ? 0 : 2)}} ${{units[idx]}}`;
    }};

    const fmtTime = (ts) => {{
      const date = new Date(ts * 1000);
      return date.toLocaleTimeString();
    }};

    function updateStatus(text, sampleText) {{
      document.getElementById('conn-status').textContent = text;
      if (sampleText) {{
        document.getElementById('sample-status').textContent = sampleText;
      }}
    }}

    function ensurePanel(gpu) {{
      if (state.panels.has(gpu.gpu_id)) return state.panels.get(gpu.gpu_id);
      const root = document.getElementById('app');
      root.innerHTML = '';
      const panel = document.createElement('section');
      panel.className = 'gpu-panel';
      panel.innerHTML = `
        <div class="gpu-head">
          <div>
            <div class="gpu-title"></div>
            <div class="gpu-meta"></div>
          </div>
          <div class="gpu-meta phase"></div>
        </div>
        <div class="stats"></div>
        <div class="legend"></div>
        <canvas></canvas>
        <div class="tooltip"></div>
      `;
      const legend = panel.querySelector('.legend');
      STACK_KEYS.forEach((key) => {{
        const item = document.createElement('span');
        item.className = 'legend-item';
        item.innerHTML = `<span class="swatch" style="background:${{COLORS[key]}}"></span>${{LABELS[key]}}`;
        legend.appendChild(item);
      }});
      root.appendChild(panel);
      const canvas = panel.querySelector('canvas');
      const ctx = canvas.getContext('2d');
      const panelObj = {{ root: panel, canvas, ctx, tooltip: panel.querySelector('.tooltip') }};
      canvas.addEventListener('mousemove', (event) => showTooltip(panelObj, gpu.gpu_id, event));
      canvas.addEventListener('mouseleave', () => {{
        panelObj.tooltip.style.display = 'none';
      }});
      state.panels.set(gpu.gpu_id, panelObj);
      return panelObj;
    }}

    function latestPerGpu(gpuId) {{
      for (let i = state.history.length - 1; i >= 0; i -= 1) {{
        const match = state.history[i].gpus.find((gpu) => gpu.gpu_id === gpuId);
        if (match) return match;
      }}
      return null;
    }}

    function historyPerGpu(gpuId) {{
      return state.history
        .map((snapshot) => {{
          const gpu = snapshot.gpus.find((item) => item.gpu_id === gpuId);
          if (!gpu) return null;
          return {{ timestamp: snapshot.timestamp, gpu }};
        }})
        .filter(Boolean);
    }}

    function render() {{
      if (!state.history.length) return;
      const latest = state.history[state.history.length - 1];
      updateStatus('connected', `last sample: ${{fmtTime(latest.timestamp)}}  seq=${{latest.sequence}}`);
      latest.gpus.forEach((gpu) => renderGpuPanel(gpu));
    }}

    function renderGpuPanel(gpu) {{
      const panel = ensurePanel(gpu);
      panel.root.querySelector('.gpu-title').textContent = `GPU ${{gpu.gpu_id}} · ${{gpu.name}}`;
      panel.root.querySelector('.gpu-meta').textContent =
        `uuid=${{gpu.meta.gpu_uuid}} · worker_rank=${{gpu.meta.worker_rank}} · sample=${{gpu.meta.sampling_method}}`;
      panel.root.querySelector('.phase').textContent =
        `last_event=${{gpu.meta.last_event}} · iter=${{gpu.meta.scheduler_iteration}}`;
      const stats = panel.root.querySelector('.stats');
      stats.innerHTML = '';
      const statItems = [
        ['Used', fmtBytes(gpu.used_memory_bytes)],
        ['Free', fmtBytes(gpu.free_memory_bytes)],
        ['Total', fmtBytes(gpu.total_memory_bytes)],
        ['KV Used', fmtBytes(gpu.meta.kv_cache_used_bytes)],
        ['KV Reserved', fmtBytes(gpu.meta.kv_cache_reserved_bytes)],
        ['Torch Reserved', fmtBytes(gpu.meta.torch_reserved_bytes)],
      ];
      statItems.forEach(([label, value]) => {{
        const item = document.createElement('div');
        item.className = 'stat';
        item.innerHTML = `<div class="stat-label">${{label}}</div><div class="stat-value">${{value}}</div>`;
        stats.appendChild(item);
      }});
      drawChart(panel, historyPerGpu(gpu.gpu_id));
    }}

    function drawChart(panel, series) {{
      const canvas = panel.canvas;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(Math.floor(rect.width * devicePixelRatio), 320);
      const height = Math.max(Math.floor(rect.height * devicePixelRatio), 240);
      if (canvas.width !== width || canvas.height !== height) {{
        canvas.width = width;
        canvas.height = height;
      }}
      const ctx = panel.ctx;
      ctx.clearRect(0, 0, width, height);
      const pad = {{ left: 52, right: 16, top: 14, bottom: 28 }};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const totals = series.map((point) => STACK_KEYS.reduce((sum, key) => sum + point.gpu.breakdown[key], 0));
      const maxY = Math.max(...totals, 1);
      for (let i = 0; i < 4; i += 1) {{
        const y = pad.top + (plotH * i) / 3;
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width - pad.right, y);
        ctx.stroke();
        const label = fmtBytes(maxY * (1 - i / 3));
        ctx.fillStyle = 'rgba(220,235,245,0.72)';
        ctx.font = `${{12 * devicePixelRatio}}px IBM Plex Sans`;
        ctx.fillText(label, 8 * devicePixelRatio, y + 4 * devicePixelRatio);
      }}
      if (series.length < 2) return;
      const xAt = (index) => pad.left + (plotW * index) / (series.length - 1);
      let cumulative = new Array(series.length).fill(0);
      STACK_KEYS.forEach((key) => {{
        const previous = cumulative.slice();
        cumulative = cumulative.map((value, index) => value + series[index].gpu.breakdown[key]);
        ctx.beginPath();
        ctx.moveTo(xAt(0), pad.top + plotH - (previous[0] / maxY) * plotH);
        for (let i = 0; i < series.length; i += 1) {{
          const x = xAt(i);
          const y = pad.top + plotH - (cumulative[i] / maxY) * plotH;
          ctx.lineTo(x, y);
        }}
        for (let i = series.length - 1; i >= 0; i -= 1) {{
          const x = xAt(i);
          const y = pad.top + plotH - (previous[i] / maxY) * plotH;
          ctx.lineTo(x, y);
        }}
        ctx.closePath();
        ctx.fillStyle = COLORS[key] + 'aa';
        ctx.fill();
      }});
      ctx.strokeStyle = 'rgba(255,255,255,0.16)';
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top + plotH);
      ctx.lineTo(width - pad.right, pad.top + plotH);
      ctx.stroke();
      ctx.fillStyle = 'rgba(220,235,245,0.72)';
      ctx.font = `${{12 * devicePixelRatio}}px IBM Plex Sans`;
      ctx.fillText(fmtTime(series[0].timestamp), pad.left, height - 8 * devicePixelRatio);
      const rightLabel = fmtTime(series[series.length - 1].timestamp);
      const textWidth = ctx.measureText(rightLabel).width;
      ctx.fillText(rightLabel, width - pad.right - textWidth, height - 8 * devicePixelRatio);
    }}

    function showTooltip(panel, gpuId, event) {{
      const series = historyPerGpu(gpuId);
      if (series.length < 1) return;
      const rect = panel.canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const index = Math.min(
        series.length - 1,
        Math.max(0, Math.round((x / rect.width) * (series.length - 1)))
      );
      const point = series[index];
      const gpu = point.gpu;
      panel.tooltip.innerHTML = `
        <div><strong>${{fmtTime(point.timestamp)}}</strong></div>
        ${{
          STACK_KEYS.map((key) => `<div>${{LABELS[key]}}: ${{fmtBytes(gpu.breakdown[key])}}</div>`).join('')
        }}
        <div>kv_cache_used: ${{fmtBytes(gpu.meta.kv_cache_used_bytes)}}</div>
        <div>torch_allocated: ${{fmtBytes(gpu.meta.torch_allocated_bytes)}}</div>
        <div>torch_reserved: ${{fmtBytes(gpu.meta.torch_reserved_bytes)}}</div>
        <div>last_event: ${{gpu.meta.last_event}}</div>
      `;
      panel.tooltip.style.left = `${{event.clientX - rect.left}}px`;
      panel.tooltip.style.top = `${{event.clientY - rect.top}}px`;
      panel.tooltip.style.display = 'block';
    }}

    async function loadHistory() {{
      const resp = await fetch(`${{BASE_PATH}}/api/history`);
      const data = await resp.json();
      state.history = data.history || [];
      render();
    }}

    function connectWs() {{
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${{proto}}://${{location.host}}${{BASE_PATH}}/ws`);
      ws.onopen = () => updateStatus('connected');
      ws.onclose = () => {{
        updateStatus('disconnected, retrying...');
        setTimeout(connectWs, 1000);
      }};
      ws.onmessage = (event) => {{
        const payload = JSON.parse(event.data);
        state.history.push(payload);
        if (state.history.length > 3000) {{
          state.history = state.history.slice(-3000);
        }}
        render();
      }};
    }}

    loadHistory().then(connectWs).catch((err) => {{
      updateStatus(`failed to load history: ${{err}}`);
      setTimeout(connectWs, 1000);
    }});
    window.addEventListener('resize', render);
  </script>
</body>
</html>"""


def register_memory_profiler_routes(app: FastAPI, args: Namespace) -> None:
    config = args.memory_profiler_config
    if isinstance(config, dict):
        config = MemoryProfilerConfig(**config)
    if not config.enabled or not config.web_enabled:
        return
    base_path = _normalize_base_path(config.web_path)
    router = APIRouter()

    @router.get(base_path, response_class=HTMLResponse)
    async def memory_profiler_index() -> HTMLResponse:
        return HTMLResponse(_render_html(base_path))

    @router.get(f"{base_path}/api/snapshot")
    async def memory_profiler_snapshot(request: Request) -> JSONResponse:
        service: MemoryProfilerService | None = getattr(
            request.app.state, "memory_profiler_service", None
        )
        if service is None:
            return JSONResponse({"snapshot": None})
        snapshot = await service.collect_once(force=True)
        return JSONResponse({"snapshot": snapshot.to_dict() if snapshot else None})

    @router.get(f"{base_path}/api/history")
    async def memory_profiler_history(request: Request) -> JSONResponse:
        service: MemoryProfilerService | None = getattr(
            request.app.state, "memory_profiler_service", None
        )
        if service is None:
            return JSONResponse({"history": []})
        return JSONResponse({"history": service.get_history()})

    @router.websocket(f"{base_path}/ws")
    async def memory_profiler_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        service: MemoryProfilerService | None = getattr(
            websocket.app.state, "memory_profiler_service", None
        )
        if service is None:
            await websocket.close()
            return
        queue = service.subscribe()
        try:
            while True:
                snapshot = await queue.get()
                await websocket.send_json(snapshot.to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            service.unsubscribe(queue)

    app.include_router(router)


def init_memory_profiler_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
) -> None:
    del supported_tasks
    config = args.memory_profiler_config
    if isinstance(config, dict):
        config = MemoryProfilerConfig(**config)
    if not config.enabled:
        state.memory_profiler_service = None
        return
    state.memory_profiler_service = MemoryProfilerService(engine_client, config)
