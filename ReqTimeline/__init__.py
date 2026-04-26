from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_ROOT_REQTIMELINE = (
    Path(__file__).resolve().parents[2] / "ReqTimeline" / "request_timeline.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_root_request_timeline", _ROOT_REQTIMELINE
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load ReqTimeline from {_ROOT_REQTIMELINE}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

RequestTimeline = _MODULE.RequestTimeline
requestTimeline = _MODULE.requestTimeline
_Event = _MODULE._Event
