"""Side-by-side comparison report with tabs.

Loads two finished `demo_task_sweep` runs (one for each first-speaker order)
and emits a single HTML page with:

  - Overview: condition × first-speaker accuracy table + side-by-side bars
  - Tabs: full per-ordering report (heatmaps, trajectories, per-problem
    transcripts) — switchable via tabs.
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


def acc_bar_svg(acc: dict[str, float], counts: dict[str, tuple[int, int]], title: str,
                bar_w: int = 60, bar_h: int = 100) -> str:
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
    title: str = "", cell_w: int = 110, cell_h: int = 64,
    label_left_w: int = 70, label_top_h: int = 22,
) -> str:
    width = label_left_w + len(emotions) * cell_w + 12
    height = label_top_h + len(conditions) * cell_h + 28
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
        f'viewBox="0 0 {width} {height}" font-family="ui-monospace,Menlo,monospace" font-size="7">'
    ]
    if title:
        svg.append(f'<text x="{label_left_w}" y="10" font-size="9" fill="#5c6370">{title}</text>')
    for j, e in enumerate(emotions):
        cx = label_left_w + j * cell_w + cell_w / 2
        lo, hi = y_ranges[e]
        svg.append(f'<text x="{cx}" y="{label_top_h - 8}" text-anchor="middle" font-weight="600" fill="#1a1a1a">s_{e}</text>')
        svg.append(f'<text x="{cx}" y="{label_top_h - 1}" text-anchor="middle" font-size="6" fill="#5c6370">[{lo:+.2f}, {hi:+.2f}]</text>')
    for i, cond in enumerate(conditions):
        cy = label_top_h + i * cell_h + cell_h / 2
        svg.append(f'<text x="{label_left_w - 6}" y="{cy + 2}" text-anchor="end" font-weight="600" fill="#1a1a1a">{cond}</text>')
        for j, e in enumerate(emotions):
            x0 = label_left_w + j * cell_w
            y0 = label_top_h + i * cell_h
            inner_w = cell_w - 4
            inner_h = cell_h - 6
            ax_x = x0 + 2
            ax_y = y0 + 3
            lo, hi = y_ranges[e]
            span = hi - lo
            svg.append(f'<rect x="{ax_x}" y="{ax_y}" width="{inner_w}" height="{inner_h}" fill="#fdfcf8" stroke="#e4e4e0" stroke-width="0.5"/>')
            if lo < 0 < hi:
                zy = ax_y + inner_h * (hi - 0) / span
                svg.append(f'<line x1="{ax_x}" y1="{zy}" x2="{ax_x + inner_w}" y2="{zy}" stroke="#ccc" stroke-dasharray="2 2" stroke-width="0.5"/>')
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
                svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.2" points="{" ".join(pts)}"/>')
                last_px, last_py = pts[-1].split(",")
                svg.append(f'<circle cx="{last_px}" cy="{last_py}" r="1.3" fill="{color}"/>')
    leg_y = label_top_h + len(conditions) * cell_h + 12
    leg_x = label_left_w
    svg.append(f'<line x1="{leg_x}" y1="{leg_y}" x2="{leg_x + 14}" y2="{leg_y}" stroke="#c2410c" stroke-width="1.4"/>')
    svg.append(f'<text x="{leg_x + 18}" y="{leg_y + 3}" font-size="8" fill="#1a1a1a">alice (steered)</text>')
    svg.append(f'<line x1="{leg_x + 96}" y1="{leg_y}" x2="{leg_x + 110}" y2="{leg_y}" stroke="#0369a1" stroke-width="1.4"/>')
    svg.append(f'<text x="{leg_x + 114}" y="{leg_y + 3}" font-size="8" fill="#1a1a1a">bob (probe-only)</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def aggregate(results: dict, problems: list[dict], emotions: list[str]) -> dict:
    conds = list(results.keys())
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


def render_full_panel(results: dict, problems: list[dict], emotions: list[str], label: str) -> list[str]:
    """All sections for a single ordering — bars, heatmaps, trajectories, transcripts."""
    g = aggregate(results, problems, emotions)
    conds = g["conds"]
    out = []

    out.append(f"<h3>Joint accuracy ({label})</h3>")
    out.append(acc_bar_svg(g["acc"], g["counts"], title=label))

    out.append(f"<h3>Bob's averaged trait drift ({label})</h3>")
    out.append(heatmap_svg(rows=conds, cols=emotions, values=g["bob_drift"],
                           title=f"bob — avg drift across problems · {label}"))

    out.append(f"<h3>Alice's mean trait projection ({label})</h3>")
    out.append(heatmap_svg(rows=conds, cols=emotions, values=g["alice_mean"],
                           title=f"alice — mean projection · {label}"))

    out.append(f"<h3>Per-condition × per-emotion trajectories ({label})</h3>")
    out.append(trajectory_panel_svg(g["avg_traj"], conds, emotions, title=label))

    out.append(f"<h3>Per-problem transcripts ({label})</h3>")
    for cond in conds:
        n_correct = sum(p["correct"] for p in results[cond])
        n = len(results[cond])
        out.append(f"<h4>{cond} — {n_correct}/{n} correct</h4>")
        for i, p in enumerate(results[cond]):
            problem = problems[i]
            mark_cls = "correct" if p["correct"] else "incorrect"
            mark = "✓" if p["correct"] else "✗"
            short_q = problem["question"].replace("\n", " ")[:120]
            label_html = (
                f"<span class='{mark_cls}'>[{mark}]</span> "
                f"<span class='problem-meta'>q#{problem['idx']} · "
                f"pred={p['predicted']} gold={p['gold']} · {p['n_turns']} turns</span>"
            )
            out.append(f"<details><summary>{label_html} · {short_q}…</summary><pre>")
            out.append("Question: " + problem["question"])
            out.append("Gold answer: " + str(problem["gold"]))
            out.append("Predicted: " + str(p["predicted"]) + ("  CORRECT" if p["correct"] else "  WRONG"))
            out.append("")
            out.extend(p["transcript"])
            out.append("</pre></details>")
    return out


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
        "h1,h2,h3,h4{letter-spacing:-0.01em}",
        "h2{margin-top:1.5em;color:#5c6370;font-size:1em;text-transform:uppercase;letter-spacing:0.08em}",
        "h3{margin-top:1.2em;color:#5c6370;font-size:0.95em;text-transform:uppercase;letter-spacing:0.06em}",
        "h4{margin-top:0.9em;font-size:0.95em}",
        "pre{background:#fff;border:1px solid #e4e4e0;border-radius:6px;padding:0.8em;overflow:auto;font-size:0.78em;white-space:pre-wrap}",
        "table{border-collapse:collapse;font-family:ui-monospace,Menlo,monospace;font-size:0.82em;margin:0.5em 0}",
        "td,th{border:1px solid #e4e4e0;padding:0.3em 0.6em;text-align:center}",
        "th{background:#f0efe8}",
        "td.q,th.q{text-align:left;font-weight:600}",
        "td.gold{color:#5c6370}",
        ".o{background:#dff5e3;color:#15803d;font-weight:bold;display:inline-block;width:1em;text-align:center}",
        ".x{background:#fde4d3;color:#c2410c;display:inline-block;width:1em;text-align:center}",
        ".diff-up{background:#dff5e3;color:#15803d;font-weight:bold}",
        ".diff-down{background:#fde4d3;color:#c2410c;font-weight:bold}",
        ".diff-zero{color:#5c6370}",
        ".caption{color:#5c6370;font-size:0.85em;margin:0.2em 0 1em}",
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:1em;align-items:start}",
        ".pair > div{background:#fff;border:1px solid #e4e4e0;border-radius:6px;padding:0.6em 0.9em}",
        "details{background:#fff;border:1px solid #e4e4e0;border-radius:6px;margin:0.3em 0;padding:0.4em 0.8em}",
        "details pre{font-size:0.78em}",
        ".problem-meta{font-family:ui-monospace,Menlo,monospace;font-size:0.78em;color:#5c6370}",
        ".correct{color:#15803d;font-weight:bold}.incorrect{color:#c2410c;font-weight:bold}",
        # Tabs
        ".tabs{display:flex;gap:0;border-bottom:2px solid #1a1a1a;margin-top:1em;margin-bottom:0.8em}",
        ".tabs button{font-family:Charter,Georgia,serif;font-size:1em;padding:0.55em 1.3em;border:none;background:transparent;cursor:pointer;color:#5c6370;border-bottom:2px solid transparent;margin-bottom:-2px;letter-spacing:0.02em}",
        ".tabs button.active{color:#1a1a1a;font-weight:bold;border-bottom:2px solid #c2410c;background:#fff}",
        ".tab-panel{display:none}",
        ".tab-panel.active{display:block}",
        "</style></head><body>",
    ]
    html.append("<h1>alice-first vs bob-first — task sweep comparison</h1>")
    html.append(f"<p class='caption'>same {n} GSM8K problems × 7 conditions × 2 orderings · "
                "alice is the steered agent, bob is probe-only.</p>")

    # --- Headline accuracy table ---
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

    # --- Side-by-side accuracy bars (overview, always visible) ---
    html.append("<h2>Accuracy bars — overview</h2><div class='pair'>")
    html.append(f"<div>{acc_bar_svg(Ag['acc'], Ag['counts'], 'alice opens')}</div>")
    html.append(f"<div>{acc_bar_svg(Bg['acc'], Bg['counts'], 'bob opens')}</div>")
    html.append("</div>")

    # --- Per-problem outcome matrix (overview) ---
    html.append("<h2>Per-problem outcomes (alice-first / bob-first)</h2>")
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
            html.append(f"<td>{a_mark}/{b_mark}</td>")
        html.append("</tr>")
    html.append("</table>")
    html.append("<p class='caption'>Each cell shows whether the (condition, problem) was correct under "
                "alice-first / bob-first. <span class='o'>o</span> = correct, <span class='x'>x</span> = wrong.</p>")

    # --- Tabs for full per-ordering reports ---
    html.append("<h2>Detailed view (per-ordering)</h2>")
    html.append("<p class='caption'>Switch tabs to see the full report — heatmaps, trajectories, "
                "and per-problem transcripts — for each ordering.</p>")
    html.append('<div class="tabs">')
    html.append('<button class="active" onclick="showTab(\'alice\')">alice opens</button>')
    html.append('<button onclick="showTab(\'bob\')">bob opens</button>')
    html.append('</div>')

    html.append('<div id="tab-alice" class="tab-panel active">')
    html.extend(render_full_panel(A, problems, emotions, label="alice-first"))
    html.append('</div>')

    html.append('<div id="tab-bob" class="tab-panel">')
    html.extend(render_full_panel(B, problems, emotions, label="bob-first"))
    html.append('</div>')

    html.append("""<script>
function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}
</script>""")

    html.append("</body></html>")
    Path(args.out).write_text("\n".join(html), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
