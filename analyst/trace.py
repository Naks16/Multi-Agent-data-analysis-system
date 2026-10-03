"""Observability: append one JSON line per agent step to runs/<timestamp>/trace.jsonl."""

import json
from datetime import datetime
from pathlib import Path


def new_run_dir(base: str | Path = "runs") -> Path:
    run_dir = Path(base) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_event(run_dir: str | Path, agent: str, **fields) -> dict:
    """Write an event to trace.jsonl and return it (nodes also keep it in state)."""
    event = {"time": datetime.now().isoformat(timespec="seconds"), "agent": agent, **fields}
    with open(Path(run_dir) / "trace.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")
    return event
