"""Thread-safe in-memory log buffer for streaming run progress to the UI.

Each 'run' is keyed by ``{book_id}__{model_name}`` so the frontend can
poll ``GET /runs/{book_id}/{model_name}/logs`` while the aggregate
endpoint is still processing.

Easy to remove post-demo — nothing else depends on this module.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

_lock = threading.Lock()
_start_times: dict[str, float] = {}
_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)
_status: dict[str, str] = {}          # "running" | "completed" | "failed"
_results: dict[str, dict[str, Any]] = {}


def run_key(book_id: str, model_name: str) -> str:
    return f"{book_id}__{model_name}"


def start_run(key: str) -> None:
    with _lock:
        _start_times[key] = time.time()
        _logs[key] = []
        _status[key] = "running"
        _results.pop(key, None)


def add_log(
    key: str,
    agent: str,
    message: str,
    level: str = "INFO",
    detail: str = "",
) -> None:
    with _lock:
        elapsed = time.time() - _start_times.get(key, time.time())
        _logs[key].append({
            "index": len(_logs[key]),
            "timestamp": time.time(),
            "elapsed_s": round(elapsed, 2),
            "agent": agent,
            "level": level,
            "message": message,
            "detail": detail,
        })


def complete_run(key: str, result: dict[str, Any]) -> None:
    with _lock:
        _status[key] = "completed"
        _results[key] = result


def fail_run(key: str, error: str) -> None:
    with _lock:
        _status[key] = "failed"
        _logs[key].append({
            "index": len(_logs[key]),
            "timestamp": time.time(),
            "elapsed_s": round(time.time() - _start_times.get(key, time.time()), 2),
            "agent": "system",
            "level": "ERROR",
            "message": error,
            "detail": "",
        })


def get_logs(key: str, after: int = 0) -> tuple[list[dict[str, Any]], str]:
    """Return ``(new_log_entries, status)``."""
    with _lock:
        logs = _logs.get(key, [])
        return logs[after:], _status.get(key, "unknown")


def get_result(key: str) -> dict[str, Any] | None:
    with _lock:
        return _results.get(key)
