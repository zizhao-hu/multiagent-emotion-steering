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


def discover_alice_bob_cells() -> dict:
    """Discover (trait, benchmark) -> dir mapping under analysis/11_alice_bob/.

    Handles both naming patterns:
      llama3_8b_<trait>_<bench>             (older, implicit L15)
      llama3_8b_<trait>_<bench>_L<layer>    (newer, explicit layer)
    """
    base = ANALYSIS_DIR / "11_alice_bob"
    cells: dict = {}
    if not base.exists():
        return cells
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.startswith("llama3_8b_"):
            continue
        if d.name.endswith("_smoke") or "_a4_smoke" in d.name:
            continue  # the older Saturday-afternoon smokes
        rest = d.name[len("llama3_8b_"):]  # e.g. "joy_mmlu_pro" or "joy_mmlu_pro_L30"
        layer = 15
        m = rest.rsplit("_L", 1)
        if len(m) == 2 and m[1].isdigit():
            rest, layer = m[0], int(m[1])
        # Find the benchmark suffix (mmlu_pro or humaneval).
        for bench in ("mmlu_pro", "humaneval"):
            suffix = f"_{bench}"
            if rest.endswith(suffix):
                trait = rest[: -len(suffix)]
                cells[(trait, bench)] = (d, layer)
                break
    return cells


def render_alice_bob_section() -> str:
    parts = ["<h2 id='alice-bob'>3. Two-agent Alice/Bob dialogue (benchmark-grounded)</h2>"]
    parts.append("<p class='note'>Two-agent dialogue grounded on a real benchmark task — "
                 "Alice (steered with v_trait at the trait's best AUC layer) and Bob (unsteered) "
                 "discuss a single MMLU-Pro STEM question or HumanEval problem. After 6 turns, "
                 "an unsteered consensus turn states the agreed answer. <b>verification gate</b>: "
                 "self-judge score ≥ 0.10 (same Llama compares baseline vs steered text). "
                 "α is picked from a sweep over [+2, +3, +4, +6]. Steering is applied at every "
                 "forward pass during Alice's turn (prompt encoding + generation).</p>")

    cells = discover_alice_bob_cells()
    if not cells:
        return "\n".join(parts + ["<p class='fail'>no cells found</p>"])

    # Aggregate stats across all cells.
    n_total = len(cells)
    n_passed = 0
    n_alice_correct_per_bench = {"mmlu_pro": 0, "humaneval": 0}
    n_bob_correct_per_bench = {"mmlu_pro": 0, "humaneval": 0}
    n_complete_per_bench = {"mmlu_pro": 0, "humaneval": 0}

    cell_data: dict = {}
    for (trait, bench), (run_dir, layer) in cells.items():
        verif_path = run_dir / "verification.json"
        if not verif_path.exists():
            continue
        v = json.load(open(verif_path))
        passed = v.get("passed", False)
        if passed:
            n_passed += 1
        d = {"trait": trait, "bench": bench, "layer": layer, "verif": v, "passed": passed}
        for setting_key, fname in [("alice", "alice_starts.json"), ("bob", "bob_starts.json")]:
            p = run_dir / fname
            if p.exists():
                d[setting_key] = json.load(open(p))
        if "alice" in d and "bob" in d:
            n_complete_per_bench[bench] += 1
            if d["alice"].get("benchmark_score", 0) >= 0.5:
                n_alice_correct_per_bench[bench] += 1
            if d["bob"].get("benchmark_score", 0) >= 0.5:
                n_bob_correct_per_bench[bench] += 1
        cell_data[(trait, bench)] = d

    # KPI strip
    parts.append(
        f"<p>"
        f"<span class='kpi'>cells with data: <b>{n_total}/22</b></span>"
        f"<span class='kpi'>verification PASS: <b>{n_passed}/{n_total}</b></span>"
        f"<span class='kpi'>completed dialogues (mmlu_pro): <b>{n_complete_per_bench['mmlu_pro']}/11</b></span>"
        f"<span class='kpi'>completed dialogues (humaneval): <b>{n_complete_per_bench['humaneval']}/11</b></span>"
        f"</p>"
    )

    # Aggregate accuracy.
    parts.append("<h3>3.1 Aggregate accuracy</h3>")
    parts.append("<p>Among cells where verification passed and both dialogues ran, "
                 "did Alice's-turn-first or Bob's-turn-first reach a correct consensus?</p>")
    parts.append("<table><tr><th>benchmark</th><th class='num'>cells completed</th>"
                 "<th class='num'>alice_starts correct</th>"
                 "<th class='num'>bob_starts correct</th></tr>")
    for bench in ["mmlu_pro", "humaneval"]:
        nc = n_complete_per_bench[bench]
        a = n_alice_correct_per_bench[bench]
        b = n_bob_correct_per_bench[bench]
        parts.append(
            f"<tr><td>{bench}</td><td class='num'>{nc}/11</td>"
            f"<td class='num'>{a}/{nc} ({100*a/max(1,nc):.0f}%)</td>"
            f"<td class='num'>{b}/{nc} ({100*b/max(1,nc):.0f}%)</td></tr>"
        )
    parts.append("</table>")

    # 11x2 grid
    parts.append("<h3>3.2 Per-cell results — 11 traits × 2 benchmarks</h3>")
    parts.append("<table><tr><th>trait</th><th>L</th>"
                 "<th>verif (mmlu)</th><th class='num'>α</th><th class='num'>judge</th>"
                 "<th>mmlu alice→</th><th>mmlu bob→</th>"
                 "<th>verif (humaneval)</th><th class='num'>α</th><th class='num'>judge</th>"
                 "<th>humaneval alice→</th><th>humaneval bob→</th></tr>")
    for trait in ALL_TRAITS:
        cells_for_trait = {b: cell_data.get((trait, b)) for b in ("mmlu_pro", "humaneval")}
        layer_disp = "—"
        for d in cells_for_trait.values():
            if d:
                layer_disp = str(d["layer"])
                break
        row = [f"<tr><td><b>{trait}</b></td><td class='num'>L{layer_disp}</td>"]
        for bench in ("mmlu_pro", "humaneval"):
            d = cells_for_trait[bench]
            if d is None:
                row.append("<td class='flat'>—</td><td class='num'>—</td><td class='num'>—</td>"
                           "<td class='flat'>—</td><td class='flat'>—</td>")
                continue
            v = d["verif"]
            cls = "pass" if d["passed"] else "fail"
            mark = "PASS" if d["passed"] else "FAIL"
            row.append(f"<td class='{cls}'>{mark}</td>")
            row.append(f"<td class='num'>+{v.get('alpha_chosen', 0):.0f}</td>")
            row.append(f"<td class='num'>{v.get('best_judge_score', 0):+.3f}</td>")
            for sk in ("alice", "bob"):
                if sk not in d:
                    row.append("<td class='flat'>—</td>")
                else:
                    score = d[sk].get("benchmark_score", 0)
                    cls = "pos" if score >= 0.5 else "neg"
                    mark = "✓" if score >= 0.5 else "✗"
                    row.append(f"<td class='{cls}'>{mark}</td>")
        row.append("</tr>")
        parts.append("".join(row))
    parts.append("</table>")

    # Per-cell collapsible details
    parts.append("<h3>3.3 Per-cell verification + transcripts</h3>")
    for trait in ALL_TRAITS:
        for bench in ("mmlu_pro", "humaneval"):
            d = cell_data.get((trait, bench))
            if d is None:
                continue
            v = d["verif"]
            ok = d["passed"]
            head = (f"<b>{trait} × {bench}</b> · L{d['layer']} · "
                    f"<span class='{'pass' if ok else 'fail'}'>{'PASS' if ok else 'FAIL'}</span> "
                    f"· judge={v.get('best_judge_score', 0):+.3f} "
                    f"· proj_Δ={v.get('best_projection_delta', 0):+.3f} "
                    f"· α=+{v.get('alpha_chosen', 0):.0f}")
            if ok and "alice" in d:
                a_score = d["alice"].get("benchmark_score", 0)
                b_score = d["bob"].get("benchmark_score", 0)
                head += (f" · alice→{'✓' if a_score >= 0.5 else '✗'} "
                         f"· bob→{'✓' if b_score >= 0.5 else '✗'}")
            parts.append(f"<details><summary>{head}</summary>")

            parts.append(f"<p class='note'><b>baseline:</b> "
                         f"<i>{html.escape(v.get('baseline_text', '')[:300])}</i></p>")
            # Verification sweep
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

            # Transcripts (if present)
            for sk, label in [("alice", "Alice (steered) starts"),
                              ("bob", "Bob (stable) starts")]:
                if sk not in d:
                    continue
                run = d[sk]
                score = run.get("benchmark_score", 0)
                cls = "pass" if score >= 0.5 else "fail"
                mark = "CORRECT" if score >= 0.5 else "WRONG"
                parts.append(f"<h4>{label} — answer: <span class='{cls}'>{mark}</span></h4>")
                if run.get("consensus_text"):
                    parts.append(f"<p><b>consensus:</b> "
                                 f"<i>{html.escape(run['consensus_text'][:400])}</i></p>")
                for t in run.get("turns", []):
                    cls = "alice" if t["speaker"] == "alice" else "bob"
                    tag = f"α={t['alpha']:+.1f}" if t["steered"] else "stable"
                    parts.append(
                        f"<div class='{cls}'><b>{t['speaker'].upper()}</b> "
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
