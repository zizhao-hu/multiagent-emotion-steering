"""Per-task flip analysis + Spearman summary for emotion steering runs.

Inputs: a directory of per-cell JSONs written by steer_benchmark.py,
each named {trait}_{bench}_alpha{a}.json with fields {benchmark, alpha,
mean_score, n, per_task, samples_preview}.

Outputs:
  - HTML report at <out_html> with three sections per model run:
      1. Spearman(alpha, score) per (trait, bench)
      2. Cross-cell "emotion-sensitive task" leaderboard — tasks that
         flipped (correct<->wrong) under the most (trait, alpha) cells
      3. Per-cell flip detail — which task indices gained/lost vs baseline,
         with task text and any cached sample previews

Usage:
    python scripts/analyze_steering.py \\
        --runs runs/10_steering_benchmarks/llama3_8b_layer15 \\
               runs/10_steering_benchmarks/qwen25_7b_layer14 \\
        --labels llama3-8b qwen2.5-7b \\
        --out analysis/10_steering_benchmarks/report.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

from intrinsic_agents.benchmarks import load_humaneval, load_mmlu_pro

CELL_RE = re.compile(
    r"^(?P<trait>\w+)_(?P<bench>mmlu_pro|humaneval|gpqa)_alpha(?P<alpha>[+-]?\d+\.\d+)\.json$"
)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. Manual impl so no scipy dep."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def rank(vals: list[float]) -> list[float]:
        sorted_vals = sorted((v, i) for i, v in enumerate(vals))
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_vals[j + 1][0] == sorted_vals[i][0]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[sorted_vals[k][1]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((r - mx) ** 2 for r in rx) ** 0.5
    dy = sum((r - my) ** 2 for r in ry) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def load_run(run_dir: Path) -> dict:
    """{(trait, bench): {alpha: cell_dict}} — a single model's grid."""
    grid: dict = defaultdict(dict)
    for f in sorted(run_dir.glob("*_alpha*.json")):
        m = CELL_RE.match(f.name)
        if not m:
            continue
        cell = json.load(open(f))
        grid[(m.group("trait"), m.group("bench"))][float(m.group("alpha"))] = cell
    return grid


def task_text(bench: str, idx: int, task_cache: dict) -> tuple[str, str]:
    """Return (short_label, full_text) for a task index. Lazy-loads dataset."""
    if bench not in task_cache:
        if bench == "mmlu_pro":
            task_cache[bench] = load_mmlu_pro(n=100, seed=0)
        elif bench == "humaneval":
            task_cache[bench] = load_humaneval(n=50, seed=0)
        else:
            task_cache[bench] = []
    tasks = task_cache[bench]
    if idx >= len(tasks):
        return f"{bench}#{idx}", "(out of range)"
    t = tasks[idx]
    if bench == "mmlu_pro":
        # Strip the prompt scaffolding; keep just the question line.
        q = t.prompt.split("Question: ", 1)[-1].split("\n\n", 1)[0]
        label = f"{bench}#{idx} [{t.category}] -> {t.correct_letter}"
        full = f"{q}  (correct: {t.correct_letter})"
    elif bench == "humaneval":
        label = t.qid
        full = t.prompt
    else:
        label = f"{bench}#{idx}"
        full = ""
    return label, full


def analyze_run(label: str, grid: dict, task_cache: dict) -> dict:
    """Compute Spearman + flip sets for one model's grid."""
    out: dict = {"label": label, "spearman": [], "cells": {}, "task_flips": defaultdict(list)}

    for (trait, bench), cells in sorted(grid.items()):
        alphas = sorted(cells.keys())
        scores = [cells[a]["mean_score"] for a in alphas]
        rho = spearman(alphas, scores) if 0.0 in cells else float("nan")
        out["spearman"].append({
            "trait": trait, "bench": bench,
            "n_alphas": len(alphas),
            "rho": rho,
            "scores": dict(zip(alphas, scores)),
        })

        if 0.0 not in cells:
            continue
        baseline = cells[0.0]["per_task"]
        for a, cell in cells.items():
            if a == 0.0:
                continue
            curr = cell["per_task"]
            n = min(len(baseline), len(curr))
            gained = [i for i in range(n) if baseline[i] == 0.0 and curr[i] == 1.0]
            lost = [i for i in range(n) if baseline[i] == 1.0 and curr[i] == 0.0]
            out["cells"][(trait, bench, a)] = {
                "mean_curr": cell["mean_score"],
                "mean_base": cells[0.0]["mean_score"],
                "gained": gained,
                "lost": lost,
                "samples_preview": cell.get("samples_preview", []),
            }
            for i in gained + lost:
                out["task_flips"][(bench, i)].append((trait, a, "gained" if i in gained else "lost"))

    return out


def render_html(runs: list[dict], task_cache: dict) -> str:
    css = """
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 2em auto; color: #222; }
    h1 { border-bottom: 2px solid #333; }
    h2 { margin-top: 2em; color: #333; border-left: 4px solid #4a90e2; padding-left: .5em; }
    h3 { color: #555; }
    table { border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }
    th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
    th { background: #f0f0f0; }
    .pos { color: #2e7d32; font-weight: bold; }
    .neg { color: #c62828; font-weight: bold; }
    .flat { color: #888; }
    details { margin: 0.5em 0; }
    summary { cursor: pointer; padding: 4px 0; font-family: monospace; }
    .gained { background: #e8f5e9; padding: 4px 8px; border-left: 3px solid #2e7d32; margin: 4px 0; }
    .lost { background: #ffebee; padding: 4px 8px; border-left: 3px solid #c62828; margin: 4px 0; }
    pre { background: #fafafa; padding: 8px; border: 1px solid #eee; max-height: 200px; overflow: auto; font-size: 0.8em; }
    .qtext { color: #444; font-size: 0.85em; }
    """

    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Steering analysis</title><style>{css}</style></head><body>"]
    parts.append("<h1>Emotion-Steering Single-Agent Eval — Flip Analysis</h1>")
    parts.append("<p>Per-task flips and Spearman correlations for each (trait, alpha) cell vs. the alpha=0 baseline. "
                 "Single-agent: one steered model answers MMLU-Pro STEM (n=100) and HumanEval (n=50). "
                 "Note: <code>surprise</code> is missing from the grid (job timed out before that trait).</p>")

    for run in runs:
        parts.append(f"<h2>{html.escape(run['label'])}</h2>")

        # Spearman table
        parts.append("<h3>Spearman correlation: alpha &rarr; mean accuracy</h3>")
        parts.append("<table><tr><th>trait</th><th>bench</th><th>n_alphas</th><th>rho</th><th>trend</th><th>scores</th></tr>")
        for r in run["spearman"]:
            rho = r["rho"]
            cls = "pos" if rho > 0.5 else ("neg" if rho < -0.5 else "flat")
            label = "monotonic +" if rho > 0.5 else ("monotonic -" if rho < -0.5 else "flat/noisy")
            score_str = ", ".join(f"a{a:+.0f}={s:.2f}" for a, s in sorted(r["scores"].items()))
            parts.append(
                f"<tr><td>{r['trait']}</td><td>{r['bench']}</td><td>{r['n_alphas']}</td>"
                f"<td class='{cls}'>{rho:+.3f}</td><td>{label}</td><td><code>{score_str}</code></td></tr>"
            )
        parts.append("</table>")

        # Emotion-sensitive task leaderboard (top 15 by total flip count)
        parts.append("<h3>Most emotion-sensitive tasks</h3>")
        parts.append("<p>Tasks that flipped (correct&harr;wrong) under the most cells. "
                     "These are the questions whose outcome most depends on which emotion vector is being added.</p>")
        ranking = sorted(run["task_flips"].items(), key=lambda kv: -len(kv[1]))[:15]
        parts.append("<table><tr><th>task</th><th>n_flips</th><th>which cells</th><th>question</th></tr>")
        for (bench, idx), flips in ranking:
            label, full = task_text(bench, idx, task_cache)
            cell_str = ", ".join(f"{t}/a{a:+.0f}/{d[0]}" for t, a, d in flips[:8])
            if len(flips) > 8:
                cell_str += f" ...({len(flips) - 8} more)"
            qtxt = html.escape(full[:200] + ("..." if len(full) > 200 else ""))
            parts.append(
                f"<tr><td><code>{html.escape(label)}</code></td><td>{len(flips)}</td>"
                f"<td>{cell_str}</td><td class='qtext'>{qtxt}</td></tr>"
            )
        parts.append("</table>")

        # Per-cell flip detail
        parts.append("<h3>Per-cell flip detail (vs alpha=0 baseline)</h3>")
        for (trait, bench, a), info in sorted(run["cells"].items()):
            delta = info["mean_curr"] - info["mean_base"]
            sign = "&#x25B2;" if delta > 0 else ("&#x25BC;" if delta < 0 else "&middot;")
            sumtxt = (f"{trait} / {bench} / a={a:+.1f} &mdash; mean {info['mean_base']:.3f} "
                      f"&rarr; {info['mean_curr']:.3f} ({sign}{delta:+.3f}) &mdash; "
                      f"+{len(info['gained'])} gained / -{len(info['lost'])} lost")
            parts.append(f"<details><summary>{sumtxt}</summary>")
            for i in info["gained"]:
                lab, q = task_text(bench, i, task_cache)
                parts.append(f"<div class='gained'><b>+ {html.escape(lab)}</b><div class='qtext'>{html.escape(q[:300])}</div></div>")
            for i in info["lost"]:
                lab, q = task_text(bench, i, task_cache)
                parts.append(f"<div class='lost'><b>- {html.escape(lab)}</b><div class='qtext'>{html.escape(q[:300])}</div></div>")
            if info["samples_preview"]:
                parts.append("<p><i>First 3 sample completions for this cell:</i></p>")
                for s in info["samples_preview"]:
                    parts.append(f"<pre>{html.escape(s[:600])}</pre>")
            parts.append("</details>")

    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="One or more run dirs")
    p.add_argument("--labels", nargs="+", required=True, help="Display labels, one per run")
    p.add_argument("--out", required=True, help="Output HTML path")
    args = p.parse_args()

    if len(args.runs) != len(args.labels):
        raise ValueError("must pass one --label per --run")

    task_cache: dict = {}
    runs = []
    for run_dir, label in zip(args.runs, args.labels):
        grid = load_run(Path(run_dir))
        runs.append(analyze_run(label, grid, task_cache))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(runs, task_cache), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
