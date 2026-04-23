"""Static HTML trajectory viewer. TODO: implement.

Reads a directory of rollout JSONL files and produces a single static HTML
page with (a) per-trait projection trajectories across training steps,
(b) a transcript viewer indexed by step, (c) reward-component stacked bars.
"""

from __future__ import annotations

from pathlib import Path


def build(runs_dir: str | Path, out_html: str | Path) -> None:
    raise NotImplementedError("Dashboard not yet implemented.")
