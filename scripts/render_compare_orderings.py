"""Render a side-by-side report comparing alice-first vs bob-first task sweeps.

Loads two finished `demo_task_sweep` runs (one for each first-speaker order)
and emits a single HTML page with paired panels: accuracy bars, bob's drift
heatmap, alice's mean-projection heatmap, averaged trajectory grid, and a
per-problem outcome matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_contagion_sweep import heatmap_svg, EMOTIONS
from demo_task_sweep import average_trajectory

REPO_ROOT = Path(__file__).resolve().parent.parent


def acc_bar_svg(acc: dict[str, float], counts: dict[str, tuple[int, int]], title: str) -> str:
    bar_w = 60
    bar_h = 100
    pad_x = 50
    pad_y = 24
    sep = 6
    conds = list(acc.keys())
    svg_w = pad_x + (bar_w + sep) * len(conds) + 16
    svg_h = pad_y + bar_h + 56
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}" font-family="ui-monospace,Menlo,monospace" font-size="9">'
    ]
    out.append(f'<text x="{pad_x}" y="14" font-size="10" fill="#5c6370">{title}</text>')
    out.append(f'<line x1="{pad_x}" y1="{pad_y + bar_h}" x2="{svg_w - 8}" y2="{pad_y + bar_h}" stroke="#888"/>')
    out.append(f'<text x="{pad_x - 6}" y="{pad_y + 4}" text-anchor="end" fill="#5c6370">1.0</text>')
    out.append(f'<text x="{pad_x - 6}" y="{pad_y + bar_h + 3}" text-anchor="end" fill="#5c6370">0.0</text>')
    if "control" in acc:
        cy = pad_y + bar_h * (1 - acc["control"])
        out.append(f'<line x1="{pad_x}" y1="{cy}" x2="{svg_w - 8}" y2="{cy}" '
                   f'stroke="#c2410c" stroke-dasharray="3 3" stroke-width="0.8" opacity="0.6"/>')
    for i, cond in enumerate(conds):
        x = pad_x + i * (bar_w + sep)
        h = bar_h * acc[cond]
        y = pad_y + bar_h - h
        color = "#0369a1" if cond != "control" else "#5c6370"
        out.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" opacity="0.85"/>')
        out.append(f'<text x="{x + bar_w/2}" y="{y - 3}" text-anchor="middle" font-weight="600" fill="#1a1a1a">{acc[cond]:.2f}</text>')
        out.append(f'<text x="{x + bar_w/2}" y="{pad_y + bar_h + 14}" text-anchor="middle" fill="#1a1a1a">{cond}</text>')
        n_correct, n = counts[cond]
        out.append(f'<text x="{x + bar_w/2}" y="{pad_y + bar_h + 25}" text-anchor="middle" font-size="8" fill="#5c6370">{n_correct}/{n}</text>')
    out.append("</svg>")
    return "\n".join(out)


def trajectory_panel_svg(
    avg_traj: dict, conditions: list[str], emotions: list[str],
    title: str, cell_w: int = 86, cell_h: int = 52,
    label_left_w: int = 60, label_top_h: int = 22,
) -> str:
    width = label_left_w + len(emotions) * cell_w + 8
    height = label_top_h + len(conditions) * cell_h + 30
    y_ranges: dict[str, tuple[float, float]] = {}
    for e in emotions:
        all_v = []
        for cond in conditions:
            for ag in ("alice", "bob"):
                all_v.extend(avg_traj[cond][ag][e])
        if not all_v:
            y_ranges[e] = (-1.0, 1.0)
            continue
        lo, hi = min(all_v), max(all_v)
        pad = 0.1 * max(hi - lo, 0.1)
        y_ranges[e] = (lo - pad, hi + pad)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-monospace,Menlo,monospace" font-size="6">'
    ]
    svg.append(f'<text x="{label_left_w}" y="8" font-size="8" fill="#5c6370">{title}</text>')
    for j, e in enumerate(emotions):
        cx = label_left_w + j * cell_w + cell_w / 2
        lo, hi = y_ranges[e]
        svg.append(f'<text x="{cx}" y="{label_top_h - 8}" text-anchor="middle" font-weight="600" font-size="7" fill="#1a1a1a">s_{e}</text>')
        svg.append(f'<text x="{cx}" y="{label_top_h - 1}" text-anchor="middle" font-size="5" fill="#5c6370">[{lo:+.2f}, {hi:+.2f}]</text>')
    for i, cond in enumerate(conditions):
        cy = label_top_h + i * cell_h + cell_h / 2
        svg.append(f'<text x="{label_left_w - 4}" y="{cy + 2}" text-anchor="end" font-weight="600" font-size="7" fill="#1a1a1a">{cond}</text>')
        for j, e in enumerate(emotions):
            x0 = label_left_w + j * cell_w
            y0 = label_top_h + i * cell_h
            inner_w = cell_w - 4
            inner_h = cell_h - 6
            ax_x = x0 + 2
            ax_y = y0 + 3
            lo, hi = y_ranges[e]
            span = hi - lo
            svg.append(f'<rect x="{ax_x}" y="{ax_y}" width="{inner_w}" height="{inner_h}" fill="#fdfcf8" stroke="#e4e4e0" stroke-width="0.4"/>')
            if lo < 0 < hi:
                zy = ax_y + inner_h * (hi - 0) / span
                svg.append(f'<line x1="{ax_x}" y1="{zy}" x2="{ax_x + inner_w}" y2="{zy}" stroke="#ccc" stroke-dasharray="2 2" stroke-width="0.4"/>')
            for ag, color in (("alice", "#c2410c"), ("bob", "#0369a1")):
                vals = avg_traj[cond][ag][e]
                n = len(vals)
                if n < 1:
                    continue
                pts = []
                for k, v in enumerate(vals):
                    px = ax_x + (inner_w * k / max(n - 1, 1) if n > 1 else inner_w / 2)
                    py = ax_y + inner_h * (hi - v) / span
                    pts.append(f"{px:.1f},{py:.1f}")
                svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1" points="{" ".join(pts)}"/>')
                last_px, last_py = pts[-1].split(",")
                svg.append(f'<circle cx="{last_px}" cy="{last_py}" r="1.1" fill="{color}"/>')
    svg.append("</svg>")
    return "\n".join(svg)


def aggregate(results: dict, problems: list[dict], emotions: list[str]) -> dict:
    conds = list(results.keys())
    n = len(problems)
    acc = {c: sum(p["correct"] for p in results[c]) / max(len(results[c]), 1) for c in conds}
    counts = {c: (sum(p["correct"] for p in results[c]), len(results[c])) for c in conds}
    bob_drift = []
    for c in conds:
        row = []
        for t in emotions:
            ds = []
            for p in results[c]:
                tr = p["trajectories"]["bob"][t]
                if len(tr) >= 2:
                    ds.append(tr[-1] - tr[0])
            row.append(sum(ds) / len(ds) if ds else 0.0)
        bob_drift.append(row)
    alice_mean = []
    for c in conds:
        row = []
        for t in emotions:
            ms = []
            for p in results[c]:
                tr = p["trajectories"]["alice"][t]
                if tr:
                    ms.append(sum(tr) / len(tr))
            row.append(sum(ms) / len(ms) if ms else 0.0)
        alice_mean.append(row)
    avg_traj = {}
    for c in conds:
        avg_traj[c] = {"alice": {}, "bob": {}}
        for t in emotions:
            for ag in ("alice", "bob"):
                series = [p["trajectories"][ag][t] for p in results[c]]
                avg_traj[c][ag][t] = average_trajectory(series)
    return {
        "conds": conds, "acc": acc, "counts": counts,
        "bob_drift": bob_drift, "alice_mean": alice_mean,
        "avg_traj": avg_traj,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alice-first-dir", default=str(REPO_ROOT / "runs" / "demo" / "task_sweep_n20"))
    p.add_argument("--bob-first-dir", default=str(REPO_ROOT / "runs" / "demo" / "task_sweep_n20_bobfirst"))
    p.add_argument("--out", default=str(REPO_ROOT / "runs" / "demo" / "compare_orderings.html"))
    args = p.parse_args()

    A_dir = Path(args.alice_first_dir)
    B_dir = Path(args.bob_first_dir)
    A = json.loads((A_dir / "results.json").read_text())
    B = json.loads((B_dir / "results.json").read_text())
    problems = json.loads((A_dir / "problems.json").read_text())

    emotions = EMOTIONS
    Ag = aggregate(A, problems, emotions)
    Bg = aggregate(B, problems, emotions)

    conds = Ag["conds"]
    n = len(problems)

    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>alice-first vs bob-first — task sweep</title>",
        "<style>",
        "body{font-family:Charter,Georgia,serif;max-width:1280px;margin:2em auto;padding:0 1em;color:#1a1a1a;background:#fafaf7}",
        "h1,h2,h3{letter-spacing:-0.01em}h2{margin-top:1.5em;color:#5c6370;font-size:1em;text-transform:uppercase;letter-spacing:0.08em}",
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:1em;align-items:start}",
        ".pair > div{background:#fff;border:1px solid #e4e4e0;border-radius:6px;padding:0.6em 0.9em}",
        "pre{background:#fff;border:1px solid #e4e4e0;border-radius:6px;padding:0.8em;overflow:auto;font-size:0.78em;white-space:pre-wrap}",
        "table{border-collapse:collapse;font-family:ui-monospace,Menlo,monospace;font-size:0.78em;margin:0.5em 0}",
        "td,th{border:1px solid #e4e4e0;padding:0.25em 0.45em;text-align:center}",
        "th{background:#f0efe8}",
        "td.q,th.q{text-align:left;font-weight:600}",
        "td.gold{color:#5c6370}",
        ".o{background:#dff5e3;color:#15803d;font-weight:bold}",
        ".x{background:#fde4d3;color:#c2410c}",
        ".diff-up{background:#dff5e3;color:#15803d;font-weight:bold}",
        ".diff-down{background:#fde4d3;color:#c2410c;font-weight:bold}",
        ".diff-zero{color:#5c6370}",
        ".caption{color:#5c6370;font-size:0.85em;margin:0.2em 0 1em}",
        "</style></head><body>",
    ]
    html.append("<h1>alice-first vs bob-first — task sweep comparison</h1>")
    html.append(f"<p class='caption'>same {n} GSM8K problems × 7 conditions × 2 orderings · "
                "alice is the steered agent, bob is probe-only · "
                "left column = alice opens the conversation; right column = bob opens.</p>")

    # --- Headline accuracy comparison table ---
    html.append("<h2>Joint accuracy by condition × first-speaker</h2>")
    html.append("<table>")
    html.append("<tr><th class='q'>condition</th><th>alice-first</th><th>bob-first</th><th>Δ (B−A)</th></tr>")
    for c in conds:
        a_n, a_d = Ag["counts"][c]
        b_n, b_d = Bg["counts"][c]
        diff = b_n - a_n
        cls = "diff-up" if diff > 0 else ("diff-down" if diff < 0 else "diff-zero")
        html.append(f"<tr><td class='q'>{c}</td>"
                    f"<td>{a_n}/{a_d}  ({Ag['acc'][c]:.2f})</td>"
                    f"<td>{b_n}/{b_d}  ({Bg['acc'][c]:.2f})</td>"
                    f"<td class='{cls}'>{diff:+d}</td></tr>")
    html.append("</table>")

    # --- Side-by-side accuracy bars ---
    html.append("<h2>Accuracy bars</h2><div class='pair'>")
    html.append(f"<div>{acc_bar_svg(Ag['acc'], Ag['counts'], 'alice opens')}</div>")
    html.append(f"<div>{acc_bar_svg(Bg['acc'], Bg['counts'], 'bob opens')}</div>")
    html.append("</div>")

    # --- Side-by-side heatmaps ---
    html.append("<h2>Bob's averaged trait drift (per condition)</h2><div class='pair'>")
    html.append(f"<div>{heatmap_svg(rows=conds, cols=emotions, values=Ag['bob_drift'], title='alice opens')}</div>")
    html.append(f"<div>{heatmap_svg(rows=conds, cols=emotions, values=Bg['bob_drift'], title='bob opens')}</div>")
    html.append("</div>")

    html.append("<h2>Alice's mean trait projection (sanity)</h2><div class='pair'>")
    html.append(f"<div>{heatmap_svg(rows=conds, cols=emotions, values=Ag['alice_mean'], title='alice opens')}</div>")
    html.append(f"<div>{heatmap_svg(rows=conds, cols=emotions, values=Bg['alice_mean'], title='bob opens')}</div>")
    html.append("</div>")

    # --- Side-by-side trajectory panels ---
    html.append("<h2>Per-condition × per-emotion trajectories (averaged)</h2><div class='pair'>")
    html.append(f"<div>{trajectory_panel_svg(Ag['avg_traj'], conds, emotions, title='alice opens')}</div>")
    html.append(f"<div>{trajectory_panel_svg(Bg['avg_traj'], conds, emotions, title='bob opens')}</div>")
    html.append("</div>")

    # --- Per-problem outcome matrix ---
    html.append("<h2>Per-problem outcomes (each cell shows alice-first / bob-first)</h2>")
    html.append("<table><tr><th class='q'>q</th><th class='gold'>gold</th>")
    for c in conds:
        html.append(f"<th>{c}</th>")
    html.append("</tr>")
    for i in range(n):
        html.append(f"<tr><td class='q'>{i}</td><td class='gold'>{problems[i]['gold']}</td>")
        for c in conds:
            a_ok = A[c][i]["correct"]
            b_ok = B[c][i]["correct"]
            a_mark = "<span class='o'>o</span>" if a_ok else "<span class='x'>x</span>"
            b_mark = "<span class='o'>o</span>" if b_ok else "<span class='x'>x</span>"
            html.append(f"<td>{a_mark} / {b_mark}</td>")
        html.append("</tr>")
    html.append("</table>")
    html.append("<p class='caption'>Each cell shows whether the (condition, problem) was correct under "
                "alice-first / bob-first. Green o = correct, orange x = wrong.</p>")

    html.append("</body></html>")
    Path(args.out).write_text("\n".join(html), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
