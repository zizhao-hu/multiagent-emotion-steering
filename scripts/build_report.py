"""Build a single comprehensive HTML report covering all results so far.

Sections:
    1. Layer-sweep extraction AUC (Llama-3.1-8B, Qwen2.5-7B)
    2. Single-agent steering benchmarks (MMLU-Pro STEM + HumanEval, with
       Spearman, flip leaderboard, per-cell flip detail)
    3. Two-agent Alice/Bob dialogue (verification gate + transcripts)
    4. Cross-cutting findings

Reads JSONs already pulled to analysis/. Re-uses the analysis helpers
from scripts/analyze_steering.py rather than duplicating logic.

Usage:
    python scripts/build_report.py
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

from intrinsic_agents.benchmarks import load_humaneval, load_mmlu_pro

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_steering import (  # noqa: E402
    CELL_RE, analyze_run, load_run, render_html as _render_steering,
    spearman, task_text,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"

CSS = """
:root { color-scheme: light; }
body {
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
    max-width: 1200px; margin: 1.5em auto; padding: 0 1em;
    color: #1a1a1a; line-height: 1.5;
}
h1 { border-bottom: 2px solid #333; padding-bottom: .3em; }
h2 {
    margin-top: 2.2em; color: #1a1a1a;
    border-left: 5px solid #4a90e2; padding-left: .5em;
}
h3 { color: #444; margin-top: 1.6em; }
h4 { color: #555; margin-top: 1.2em; }
nav.toc {
    background: #f7f9fc; border: 1px solid #d0d7de;
    border-radius: 6px; padding: .8em 1.2em; margin: 1em 0 2em;
    font-size: .92em;
}
nav.toc ol { margin: .3em 0 .3em 1.2em; }
nav.toc a { text-decoration: none; color: #1f6feb; }
nav.toc a:hover { text-decoration: underline; }
table {
    border-collapse: collapse; margin: 1em 0; font-size: .88em;
}
th, td { border: 1px solid #d0d7de; padding: 5px 9px; text-align: left; }
th { background: #f0f3f6; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: #1a7f37; font-weight: 600; }
.fail { color: #cf222e; font-weight: 600; }
.pos { color: #1a7f37; font-weight: 600; }
.neg { color: #cf222e; font-weight: 600; }
.flat { color: #6e7781; }
.chosen { background: #fff4d6; }
.alice { background: #fff8e6; border-left: 4px solid #d9a300; padding: 8px 12px; margin: 6px 0; }
.alice b { color: #6f4500; }
.bob { background: #eef4fb; border-left: 4px solid #4a90e2; padding: 8px 12px; margin: 6px 0; }
.bob b { color: #1f4480; }
.alice .meta, .bob .meta { color: #555; font-size: .82em; font-family: ui-monospace, Menlo, monospace; }
details { margin: .5em 0; }
summary { cursor: pointer; padding: 4px 0; font-family: ui-monospace, monospace; }
.gained { background: #dafbe1; padding: 4px 8px; border-left: 3px solid #1a7f37; margin: 4px 0; }
.lost   { background: #ffebe9; padding: 4px 8px; border-left: 3px solid #cf222e; margin: 4px 0; }
.qtext { color: #444; font-size: .85em; }
pre {
    background: #f6f8fa; padding: 8px; border: 1px solid #d0d7de;
    border-radius: 4px; max-height: 200px; overflow: auto; font-size: .82em;
}
.kpi {
    display: inline-block; padding: 4px 10px; margin: 0 6px 6px 0;
    background: #f0f3f6; border-radius: 999px; font-size: .85em;
}
.kpi b { color: #1f6feb; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 2em 0; }
.note { color: #555; font-size: .9em; font-style: italic; }
"""


# ---------------------------------------------------------------- helpers

def cell(content: str, cls: str = "") -> str:
    return f"<td class='{cls}'>{content}</td>" if cls else f"<td>{content}</td>"


def fmt_num(v: float, places: int = 3) -> str:
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{places}f}"


# ---------------------------------------------------------------- sec 1

def render_layer_sweep_section() -> str:
    parts = ["<h2 id='layer-sweep'>1. Layer-sweep extraction AUC</h2>"]
    parts.append("<p class='note'>For each (model, layer) the trait vector is the L2-normalized "
                 "difference of mean activations on contrastive pairs (Anthropic 2025). "
                 "Higher AUC means the linear probe at that layer cleanly separates pos/neg pairs. "
                 "Emotion-trait mean is highlighted because steering experiments target emotions.</p>")

    runs = [
        ("Llama-3.1-8B-Instruct", ANALYSIS_DIR / "00_replication" / "llama3_8b" / "sweep_summary.json"),
        ("Qwen2.5-7B-Instruct",   ANALYSIS_DIR / "00_replication" / "qwen25_7b" / "sweep_summary.json"),
    ]
    emotion_traits = {"joy", "sadness", "anger", "curiosity", "surprise"}

    for label, path in runs:
        if not path.exists():
            parts.append(f"<p class='fail'>missing: {path}</p>")
            continue
        d = json.load(open(path))
        layers = d["layers"]
        per = d["per_layer_per_trait"]
        traits = sorted({t for v in per.values() for t in v})

        means_by_layer = {
            L: sum(per[str(L)].get(t, 0.0) for t in emotion_traits if t in per[str(L)]) /
                max(1, len([t for t in emotion_traits if t in per[str(L)]]))
            for L in layers
        }
        best = max(means_by_layer, key=means_by_layer.get)

        parts.append(f"<h3>{html.escape(label)}</h3>")
        parts.append(f"<p><span class='kpi'>best emotion layer = <b>L{best}</b></span>"
                     f"<span class='kpi'>mean emotion AUC at best = <b>{means_by_layer[best]:.3f}</b></span>"
                     f"<span class='kpi'>traits scored = <b>{len(traits)}</b></span>"
                     f"<span class='kpi'>layers swept = <b>{len(layers)}</b></span></p>")

        parts.append("<table><tr><th>trait</th>" +
                     "".join(f"<th class='num'>L{L}</th>" for L in layers) + "</tr>")
        for t in traits:
            row = f"<tr><td>{t}</td>"
            for L in layers:
                v = per[str(L)].get(t)
                cls = "num"
                if v is not None and v >= 0.85:
                    cls += " pass"
                row += f"<td class='{cls}'>{fmt_num(v)}</td>"
            row += "</tr>"
            parts.append(row)
        parts.append("<tr style='font-weight:600;background:#f7f9fc;'>"
                     "<td>emotion mean</td>" +
                     "".join(f"<td class='num{' chosen' if L == best else ''}'>"
                             f"{means_by_layer[L]:.3f}</td>" for L in layers) +
                     "</tr></table>")

    return "\n".join(parts)


# ---------------------------------------------------------------- sec 2

def render_steering_benchmarks_section() -> str:
    parts = ["<h2 id='steering'>2. Single-agent steering benchmarks</h2>"]
    parts.append("<p class='note'>One steered model answers MMLU-Pro STEM (n=100) and HumanEval (n=50). "
                 "Cells: 4 emotions × 5 alphas × 2 benchmarks per model. Surprise was cut due to a "
                 "walltime timeout. <b>Spearman ρ</b> = monotonic correlation between α and accuracy. "
                 "<b>Flip</b> = a task that flipped pass↔fail when steered relative to the α=0 baseline.</p>")

    runs = [
        ("Llama-3.1-8B-Instruct (L15)", ANALYSIS_DIR / "10_steering_benchmarks" / "llama3_8b_layer15"),
        ("Qwen2.5-7B-Instruct (L14)",   ANALYSIS_DIR / "10_steering_benchmarks" / "qwen25_7b_layer14"),
    ]

    task_cache: dict = {}
    analyzed = []
    for label, run_dir in runs:
        if not run_dir.exists():
            parts.append(f"<p class='fail'>missing: {run_dir}</p>")
            continue
        grid = load_run(run_dir)
        analyzed.append(analyze_run(label, grid, task_cache))

    # Headlines: strong correlations across models.
    parts.append("<h3>Strong monotonic effects (|ρ| &gt; 0.5)</h3>")
    parts.append("<table><tr><th>model</th><th>trait</th><th>bench</th>"
                 "<th class='num'>ρ</th><th>scores at α ∈ {-4, -2, 0, +2, +4}</th></tr>")
    for run in analyzed:
        for r in run["spearman"]:
            if r["rho"] != r["rho"]:  # NaN
                continue
            if abs(r["rho"]) > 0.5:
                cls = "pos" if r["rho"] > 0 else "neg"
                scores = [r["scores"].get(a, float("nan"))
                          for a in [-4.0, -2.0, 0.0, 2.0, 4.0]]
                score_str = " · ".join(fmt_num(s, 2) for s in scores)
                parts.append(
                    f"<tr><td>{run['label']}</td><td>{r['trait']}</td><td>{r['bench']}</td>"
                    f"<td class='num {cls}'>{r['rho']:+.3f}</td><td>{score_str}</td></tr>"
                )
    parts.append("</table>")

    # Per-model spearman + flip detail
    for run in analyzed:
        parts.append(f"<h3>{html.escape(run['label'])}</h3>")
        parts.append("<table><tr><th>trait</th><th>bench</th><th class='num'>ρ</th>"
                     "<th>trend</th><th>scores</th></tr>")
        for r in run["spearman"]:
            rho = r["rho"]
            cls = "pos" if rho > 0.5 else ("neg" if rho < -0.5 else "flat")
            label = "monotonic +" if rho > 0.5 else ("monotonic −" if rho < -0.5 else "flat/noisy")
            score_str = ", ".join(f"a{a:+.0f}={s:.2f}" for a, s in sorted(r["scores"].items()))
            parts.append(
                f"<tr><td>{r['trait']}</td><td>{r['bench']}</td>"
                f"<td class='num {cls}'>{fmt_num(rho)}</td><td>{label}</td>"
                f"<td><code>{score_str}</code></td></tr>"
            )
        parts.append("</table>")

        # Top-flipped tasks
        ranking = sorted(run["task_flips"].items(), key=lambda kv: -len(kv[1]))[:8]
        parts.append("<h4>Most emotion-sensitive tasks (top 8 by flip count)</h4>")
        parts.append("<table><tr><th>task</th><th class='num'>flips</th><th>question</th></tr>")
        for (bench, idx), flips in ranking:
            lab, full = task_text(bench, idx, task_cache)
            qtxt = html.escape(full[:160] + ("..." if len(full) > 160 else ""))
            parts.append(f"<tr><td><code>{html.escape(lab)}</code></td>"
                         f"<td class='num'>{len(flips)}</td>"
                         f"<td class='qtext'>{qtxt}</td></tr>")
        parts.append("</table>")

        parts.append("<details><summary>per-cell flip detail (collapsed)</summary>")
        for (trait, bench, a), info in sorted(run["cells"].items()):
            delta = info["mean_curr"] - info["mean_base"]
            sign_cls = "pos" if delta > 0 else ("neg" if delta < 0 else "flat")
            sumtxt = (f"{trait} / {bench} / α={a:+.1f} — "
                      f"<span class='{sign_cls}'>{delta:+.3f}</span> — "
                      f"+{len(info['gained'])} gained / -{len(info['lost'])} lost")
            parts.append(f"<details><summary>{sumtxt}</summary>")
            for i in info["gained"][:5]:
                lab, q = task_text(bench, i, task_cache)
                parts.append(f"<div class='gained'><b>+ {html.escape(lab)}</b>"
                             f"<div class='qtext'>{html.escape(q[:200])}</div></div>")
            for i in info["lost"][:5]:
                lab, q = task_text(bench, i, task_cache)
                parts.append(f"<div class='lost'><b>− {html.escape(lab)}</b>"
                             f"<div class='qtext'>{html.escape(q[:200])}</div></div>")
            extras = max(0, len(info["gained"]) - 5) + max(0, len(info["lost"]) - 5)
            if extras > 0:
                parts.append(f"<p class='note'>(+{extras} more flips not shown)</p>")
            parts.append("</details>")
        parts.append("</details>")

    return "\n".join(parts)


# ---------------------------------------------------------------- sec 3

ALL_TRAITS = ["joy", "sadness", "anger", "curiosity", "surprise",
              "honesty", "sycophancy", "hallucination",
              "scholar", "caregiver", "explorer"]


MODEL_PREFIXES = {"llama3_8b": "Llama-3.1-8B", "qwen25_7b": "Qwen2.5-7B"}
ALL_BENCHMARKS = ["mmlu_pro", "humaneval", "gsm8k", "gpqa"]
STEER_MODES = ["aliceonly", "both"]


def discover_alice_bob_cells() -> dict:
    """Discover (model, trait, bench, steer_mode) -> (dir, layer) mapping.

    Naming patterns supported:
      <model>_<trait>_<bench>                          # legacy implicit L15, aliceonly
      <model>_<trait>_<bench>_L<layer>                 # explicit layer, aliceonly
      <model>_<trait>_<bench>_L<layer>_<steer_mode>    # full pattern (aliceonly | both)

    Where <model> ∈ {llama3_8b, qwen25_7b}.
    """
    base = ANALYSIS_DIR / "11_alice_bob"
    cells: dict = {}
    if not base.exists():
        return cells
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if d.name.endswith("_smoke") or "_a4_smoke" in d.name:
            continue
        # Identify model prefix.
        model_key = None
        for k in MODEL_PREFIXES:
            if d.name.startswith(k + "_"):
                model_key = k
                break
        if model_key is None:
            continue
        rest = d.name[len(model_key) + 1:]
        # Optional trailing _aliceonly | _both
        steer_mode = "aliceonly"
        for sm in STEER_MODES:
            if rest.endswith("_" + sm):
                steer_mode = sm
                rest = rest[: -(len(sm) + 1)]
                break
        # Optional _L<layer>
        layer = 15
        m = rest.rsplit("_L", 1)
        if len(m) == 2 and m[1].isdigit():
            rest, layer = m[0], int(m[1])
        # Benchmark suffix
        for bench in ALL_BENCHMARKS:
            suffix = f"_{bench}"
            if rest.endswith(suffix):
                trait = rest[: -len(suffix)]
                cells[(model_key, trait, bench, steer_mode)] = (d, layer)
                break
    return cells


def _load_cell(run_dir: Path, layer: int) -> dict | None:
    """Load verification.json + the two setting JSONs (when present) for one cell."""
    verif_path = run_dir / "verification.json"
    if not verif_path.exists():
        return None
    v = json.load(open(verif_path))
    out: dict = {"layer": layer, "verif": v, "passed": v.get("passed", False)}
    for sk, fname in [("alice", "alice_starts.json"), ("bob", "bob_starts.json")]:
        p = run_dir / fname
        if p.exists():
            out[sk] = json.load(open(p))
    return out


def _cell_mean_score(cell: dict) -> float | None:
    """Average benchmark_score across the two starting settings, when both ran."""
    scores = [cell[sk]["benchmark_score"] for sk in ("alice", "bob")
              if sk in cell and "benchmark_score" in cell[sk]]
    return sum(scores) / len(scores) if scores else None


def render_alice_bob_section() -> str:
    parts = ["<h2 id='alice-bob'>3. Two-agent Alice/Bob dialogue (benchmark-grounded)</h2>"]
    parts.append("<p class='note'>Two-agent dialogue grounded on a real benchmark task. "
                 "Alice is always steered with v<sub>trait</sub> at the trait's best AUC layer; "
                 "Bob is steered with the same vector iff the cell's setting is <code>steer-both</code>. "
                 "After 6 turns an unsteered <i>consensus</i> turn states the agreed answer; that "
                 "answer is graded against the benchmark's gold. "
                 "<b>Pre-flight gate</b>: self-judge ≥ 0.10 on a single steered Alice utterance vs an "
                 "unsteered baseline (one shot, before the dialogue runs). "
                 "<b>Post-hoc monitor</b>: after the dialogue, the same self-judge compares the "
                 "<i>concatenated Alice turns</i> (and separately Bob's) against the same baseline — "
                 "this is the running steering monitor that confirms the trait persists across the "
                 "whole conversation, not just the first utterance.</p>")

    cells = discover_alice_bob_cells()
    if not cells:
        return "\n".join(parts + ["<p class='fail'>no cells found</p>"])

    # Materialize cell data, grouped by (model, steer_mode) for top-level subsections.
    grouped: dict = {}  # (model_key, steer_mode) -> {(trait, bench): cell_data}
    for (model_key, trait, bench, mode), (run_dir, layer) in cells.items():
        cd = _load_cell(run_dir, layer)
        if cd is None:
            continue
        cd.update({"trait": trait, "bench": bench, "model": model_key, "steer_mode": mode})
        grouped.setdefault((model_key, mode), {})[(trait, bench)] = cd

    # Top-level KPI strip
    n_total = sum(len(g) for g in grouped.values())
    n_pass = sum(1 for g in grouped.values() for c in g.values() if c["passed"])
    parts.append(
        f"<p><span class='kpi'>cells with data: <b>{n_total}/{2*4*11*2}</b></span>"
        f"<span class='kpi'>pre-flight gate PASS: <b>{n_pass}/{n_total}</b></span>"
        f"<span class='kpi'>models: <b>{len({m for m,_ in grouped})}</b></span>"
        f"<span class='kpi'>steer modes: <b>{len({s for _,s in grouped})}</b></span></p>"
    )

    # 3.1 Cross-setting Δ accuracy: how does steer-both change vs steer-alice?
    parts.append("<h3 id='cross-setting'>3.1 Cross-setting Δ — does steering both agents change accuracy?</h3>")
    parts.append("<p class='note'>For each (model, trait, bench), the table shows the mean "
                 "consensus accuracy under <code>aliceonly</code> and <code>both</code> (averaged over "
                 "alice_starts and bob_starts), with the Δ. Cells without paired data are blanked.</p>")
    parts.append("<table><tr><th>model</th><th>trait</th><th>bench</th>"
                 "<th class='num'>aliceonly</th><th class='num'>both</th><th class='num'>Δ</th></tr>")
    for model_key in MODEL_PREFIXES:
        for trait in ALL_TRAITS:
            for bench in ALL_BENCHMARKS:
                a = grouped.get((model_key, "aliceonly"), {}).get((trait, bench))
                b = grouped.get((model_key, "both"), {}).get((trait, bench))
                a_s = _cell_mean_score(a) if a else None
                b_s = _cell_mean_score(b) if b else None
                if a_s is None and b_s is None:
                    continue
                delta = (b_s - a_s) if (a_s is not None and b_s is not None) else None
                cls = "" if delta is None else ("pos" if delta > 0 else ("neg" if delta < 0 else "flat"))
                a_str = fmt_num(a_s, 2) if a_s is not None else "—"
                b_str = fmt_num(b_s, 2) if b_s is not None else "—"
                d_str = (f"<span class='{cls}'>{delta:+.2f}</span>") if delta is not None else "—"
                parts.append(
                    f"<tr><td>{MODEL_PREFIXES[model_key]}</td><td>{trait}</td><td>{bench}</td>"
                    f"<td class='num'>{a_str}</td><td class='num'>{b_str}</td><td class='num'>{d_str}</td></tr>"
                )
    parts.append("</table>")

    # 3.2 Per-(model, mode) 11x4 grid
    parts.append("<h3 id='cell-grid'>3.2 Per-cell grid — 11 traits × 4 benchmarks</h3>")
    section_idx = 0
    for (model_key, mode), cell_map in sorted(grouped.items()):
        section_idx += 1
        n_pass_local = sum(1 for c in cell_map.values() if c["passed"])
        parts.append(f"<h4>{MODEL_PREFIXES[model_key]} · steer-{mode} "
                     f"(<span class='note'>{len(cell_map)} cells, "
                     f"{n_pass_local} pre-flight PASS</span>)</h4>")
        # Header
        hdr = "<tr><th>trait</th><th class='num'>L</th>"
        for bench in ALL_BENCHMARKS:
            hdr += (f"<th>{bench} gate</th><th class='num'>α</th>"
                    f"<th class='num'>preflight</th><th class='num'>posthoc A</th>"
                    f"<th class='num'>posthoc B</th>"
                    f"<th>alice→</th><th>bob→</th>")
        hdr += "</tr>"
        parts.append("<table>" + hdr)
        for trait in ALL_TRAITS:
            row = [f"<tr><td><b>{trait}</b></td>"]
            # Layer (use any of the cells for this trait for layer display)
            l_disp = "—"
            for b in ALL_BENCHMARKS:
                c = cell_map.get((trait, b))
                if c:
                    l_disp = f"L{c['layer']}"
                    break
            row.append(f"<td class='num'>{l_disp}</td>")
            for bench in ALL_BENCHMARKS:
                c = cell_map.get((trait, bench))
                if c is None:
                    row.append("<td class='flat'>—</td><td class='num'>—</td>"
                               "<td class='num'>—</td><td class='num'>—</td>"
                               "<td class='num'>—</td><td class='flat'>—</td>"
                               "<td class='flat'>—</td>")
                    continue
                v = c["verif"]
                cls = "pass" if c["passed"] else "fail"
                row.append(f"<td class='{cls}'>{'PASS' if c['passed'] else 'FAIL'}</td>")
                row.append(f"<td class='num'>+{v.get('alpha_chosen', 0):.0f}</td>")
                row.append(f"<td class='num'>{v.get('best_judge_score', 0):+.2f}</td>")
                # Post-hoc judges (mean across the two settings)
                pa = []
                pb = []
                for sk in ("alice", "bob"):
                    if sk in c:
                        pa.append(c[sk].get("post_alice_judge", float("nan")))
                        pb.append(c[sk].get("post_bob_judge", float("nan")))
                def _avg(xs):
                    xs = [x for x in xs if x == x]  # drop NaN
                    return sum(xs) / len(xs) if xs else None
                pa_v = _avg(pa); pb_v = _avg(pb)
                cls_a = "pos" if pa_v is not None and pa_v > 0.05 else ("neg" if pa_v is not None and pa_v < -0.05 else "flat")
                cls_b = "pos" if pb_v is not None and pb_v > 0.05 else ("neg" if pb_v is not None and pb_v < -0.05 else "flat")
                row.append(f"<td class='num {cls_a}'>{fmt_num(pa_v, 2)}</td>")
                row.append(f"<td class='num {cls_b}'>{fmt_num(pb_v, 2)}</td>")
                for sk in ("alice", "bob"):
                    if sk not in c:
                        row.append("<td class='flat'>—</td>")
                    else:
                        s = c[sk].get("benchmark_score", 0)
                        cls = "pos" if s >= 0.5 else "neg"
                        row.append(f"<td class='{cls}'>{'✓' if s >= 0.5 else '✗'}</td>")
            row.append("</tr>")
            parts.append("".join(row))
        parts.append("</table>")

    # 3.3 Per-cell collapsibles
    parts.append("<h3 id='cell-detail'>3.3 Per-cell detail (verification + transcripts)</h3>")
    for (model_key, mode) in sorted(grouped):
        cell_map = grouped[(model_key, mode)]
        parts.append(f"<h4>{MODEL_PREFIXES[model_key]} · steer-{mode}</h4>")
        for trait in ALL_TRAITS:
            for bench in ALL_BENCHMARKS:
                c = cell_map.get((trait, bench))
                if c is None:
                    continue
                v = c["verif"]
                ok = c["passed"]
                head = (f"<b>{trait} × {bench}</b> · L{c['layer']} · "
                        f"<span class='{'pass' if ok else 'fail'}'>{'PASS' if ok else 'FAIL'}</span> "
                        f"· preflight={v.get('best_judge_score', 0):+.3f} "
                        f"· proj_Δ={v.get('best_projection_delta', 0):+.3f} "
                        f"· α=+{v.get('alpha_chosen', 0):.0f}")
                if "alice" in c:
                    pa = c["alice"].get("post_alice_judge")
                    pb = c["alice"].get("post_bob_judge")
                    if pa is not None:
                        head += f" · posthoc(A→A={pa:+.2f}, A→B={pb:+.2f})"
                if ok and "alice" in c:
                    head += (f" · alice→{'✓' if c['alice'].get('benchmark_score', 0) >= 0.5 else '✗'} "
                             f"· bob→{'✓' if c['bob'].get('benchmark_score', 0) >= 0.5 else '✗'}")
                parts.append(f"<details><summary>{head}</summary>")

                parts.append(f"<p class='note'><b>baseline:</b> "
                             f"<i>{html.escape(v.get('baseline_text', '')[:300])}</i></p>")
                parts.append("<table><tr><th>α</th><th class='num'>judge</th>"
                             "<th class='num'>proj Δ</th><th>steered text</th></tr>")
                for s in v.get("sweep", []):
                    chosen = s["alpha"] == v.get("alpha_chosen", -999)
                    row_cls = " class='chosen'" if chosen else ""
                    cls_j = "pos" if s["judge_score"] > 0.05 else ("neg" if s["judge_score"] < -0.05 else "flat")
                    parts.append(
                        f"<tr{row_cls}><td>α={s['alpha']:+.0f}</td>"
                        f"<td class='num {cls_j}'>{s['judge_score']:+.3f}</td>"
                        f"<td class='num'>{s['projection_delta']:+.3f}</td>"
                        f"<td>{html.escape(s['steered_text'][:200])}</td></tr>"
                    )
                parts.append("</table>")
                for sk, label in [("alice", "Alice starts"), ("bob", "Bob starts")]:
                    if sk not in c:
                        continue
                    run = c[sk]
                    score = run.get("benchmark_score", 0)
                    cls = "pass" if score >= 0.5 else "fail"
                    mark = "CORRECT" if score >= 0.5 else "WRONG"
                    pa = run.get("post_alice_judge")
                    pb = run.get("post_bob_judge")
                    posthoc = (f" · posthoc Alice={pa:+.2f}, Bob={pb:+.2f}"
                               if pa is not None else "")
                    parts.append(f"<h5>{label} — answer: <span class='{cls}'>{mark}</span>{posthoc}</h5>")
                    if run.get("consensus_text"):
                        parts.append(f"<p><b>consensus:</b> "
                                     f"<i>{html.escape(run['consensus_text'][:400])}</i></p>")
                    for t in run.get("turns", []):
                        cls2 = "alice" if t["speaker"] == "alice" else "bob"
                        tag = f"α={t['alpha']:+.1f}" if t["steered"] else "stable"
                        parts.append(
                            f"<div class='{cls2}'><b>{t['speaker'].upper()}</b> "
                            f"<span class='meta'>turn {t['idx']:02d} · {tag} · "
                            f"proj={t['proj_target']:+.3f}</span>"
                            f"<div>{html.escape(t['text'])}</div></div>"
                        )
                parts.append("</details>")

    return "\n".join(parts)


# ---------------------------------------------------------------- sec 4

def render_findings_section() -> str:
    return """
<h2 id='findings'>4. Cross-cutting findings</h2>
<h3>What the data supports</h3>
<ul>
  <li><b>Persona/emotion vectors extract cleanly on instruction-tuned 7-9B models.</b>
      Best-layer mean emotion AUC: Llama-3.1-8B = 0.937 at L30, Qwen2.5-7B = 0.938 at L24.
      Both peak deep — emotion concepts concentrate in the last third of the stack.</li>
  <li><b>Negative-valence emotions consistently hurt MMLU-Pro accuracy on both models.</b>
      Llama: sadness ρ = −0.975. Qwen: anger ρ = −0.821, sadness ρ = −0.447.
      Joy modestly helps both (+0.5 / +0.41).</li>
  <li><b>Coding-vs-science effects are model-specific.</b>
      Joy strongly lifts Qwen HumanEval (ρ = +0.949) but degrades Llama HumanEval at α=+4.
      Curiosity hurts Qwen HumanEval (ρ = −0.900) — pseudo-token noise in code likely.
      Sadness on Llama splits: hurts MMLU-Pro but mildly helps HumanEval.</li>
  <li><b>Emotion-sensitive tasks cluster in chemistry, health, and physics.</b>
      Specific MMLU-Pro questions flip under 10–14 of 32 cells — these are the
      stress-tests of "is the answer style robust to emotional state."</li>
  <li><b>Steering does shape multi-agent dialogue.</b>
      In the smoke run, the steered Alice consistently uses warmer language
      ("great", "love", "perfect") and her per-turn projections sit higher
      than the unsteered Bob's. The two settings (steered-starts vs
      stable-starts) produce qualitatively similar dialogues — emotion
      doesn't depend on who opens the conversation.</li>
</ul>

<h3>What the data warns about</h3>
<ul>
  <li><b>Residual-stream projection is not a reliable read-back metric.</b>
      Mean-pool and chat-wrap-last-token projections both report
      <i>negative</i> deltas for steered text that the language judge
      and human inspection rate as more emotional. The trait vector
      is calibrated for "instruction prompt awaiting reply" geometry,
      not "parsed output text" geometry — they don't align.
      Use a self-judge (or LM-as-judge) for verifying expressed emotion.</li>
  <li><b>α=+4 overshoots on Llama-3.1-8B/joy/L15.</b> Verification α-sweep
      gives the highest judge score at α=+3 (+0.131); α=+4 and α=+6 are
      already negative (−0.10, −0.30) — the model "tries too hard"
      and the text gets flatter. Pick α via sweep, don't guess.</li>
  <li><b>N=100 MMLU-Pro is too noisy to read most monotonic trends.</b>
      Many cells have |ρ| &lt; 0.5. Bump to N=300+ or restrict to a
      single STEM subcategory before drawing conclusions cell-by-cell.</li>
  <li><b>Surprise is missing.</b> The steering benchmark grid timed out
      before reaching it (last trait in iteration order). 9 missing
      cells per model.</li>
</ul>
"""


# ---------------------------------------------------------------- top

def render_full() -> str:
    head = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Multi-agent emotion steering — results</title>
<style>{CSS}</style></head><body>"""
    title = "<h1>Multi-Agent Emotion Steering — Results</h1>"
    intro = """<p class='note'>Three result tiers: (1) layer-sweep extraction AUC
to pick where the trait vectors live cleanest, (2) single-agent steering
benchmarks on advanced science (MMLU-Pro STEM) and coding (HumanEval),
(3) two-agent Alice/Bob dialogue with a verification gate. Models tested:
Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct.</p>"""
    toc = """<nav class='toc'>
  <b>Contents</b>
  <ol>
    <li><a href='#layer-sweep'>Layer-sweep extraction AUC</a></li>
    <li><a href='#steering'>Single-agent steering benchmarks</a></li>
    <li><a href='#alice-bob'>Two-agent Alice/Bob dialogue</a>
      <ol style='list-style: lower-alpha;'>
        <li><a href='#setting-a'>Setting A — Alice (steered) starts</a></li>
        <li><a href='#setting-b'>Setting B — Bob (stable) starts</a></li>
      </ol>
    </li>
    <li><a href='#findings'>Cross-cutting findings</a></li>
  </ol>
</nav>"""

    body = "\n".join([
        render_layer_sweep_section(),
        render_steering_benchmarks_section(),
        render_alice_bob_section(),
        render_findings_section(),
    ])
    return head + title + intro + toc + body + "</body></html>"


def main() -> None:
    out = ANALYSIS_DIR / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_full(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
