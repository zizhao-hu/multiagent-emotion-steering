"""Append-only JSONL logger for rollouts.

One line per turn. Columns: step, agent_id, prompt, response, per-trait mean
projection, R_int, R_ext, R_total. The dashboard reads this file directly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", buffering=1, encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        record = {"t": time.time(), **record}
        self._fp.write(json.dumps(record, default=float) + "\n")

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
