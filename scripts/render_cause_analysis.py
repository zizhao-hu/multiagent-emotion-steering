"""Render analysis/causes_n200.json as a self-contained HTML report.

Layout:
  - Header: dataset summary, methodology blurb, cause-taxonomy legend.
  - Global cause distribution table.
  - 2 ordering sections (alice-first, bob-first), each with 6 emotion panels
    (joy+/joy-/sadness+/anger+/curiosity+/surprise+).
  - Per panel: outcome counts (stay_correct/stay_wrong/helped/hurt), cause
    distribution bar, and a collapsible per-sample table for the cells where
    the outcome actually flipped (helped + hurt).
  - Per sample row: question, gold, control vs emotion predicted, cause tag,
    and click-to-expand side-by-side transcripts + features.

Output:
  analysis/n200_cause_analysis.html
"""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANA = REPO / "analysis"

EMOTIONS = ["joy+", "joy-", "sadness+", "anger+", "curiosity+", "surprise+"]
ORDERINGS = ["alice", "bob"]

CAUSE_ORDER = [
    "A_decomposition_shift",
    "B_arithmetic_divergence",
    "C_verification_bypass",
    "D_disagreement_loop",
    "E_early_termination",
    "F_magnitude_error",
    "G_semantic_reframing",
    "I_extraction_artifact",
    "J_other",
    "H_stylistic_only",
]

CAUSE_LABEL = {
    "A_decomposition_shift": "A. Decomposition shift",
    "B_arithmetic_divergence": "B. Arithmetic divergence",
    "C_verification_bypass": "C. Verification bypass",
    "D_disagreement_loop": "D. Disagreement loop",
    "E_early_termination": "E. Early termination",
    "F_magnitude_error": "F. Magnitude error",
    "G_semantic_reframing": "G. Semantic reframing",
    "I_extraction_artifact": "I. Extraction artifact",
    "J_other": "J. Other",
    "H_stylistic_only": "H. Stylistic only",
}

CAUSE_BLURB = {
    "A_decomposition_shift": "Alice opened the problem with very different sub-steps (first-turn token Jaccard < 0.30 vs control).",
    "B_arithmetic_divergence": "Same plan as control (Jaccard ≥ 0.55) but landed on a different final number.",
    "C_verification_bypass": "Outcome flipped, emotion ended sooner than control, and emotion has more agreement-words — Bob accepted without re-deriving.",
    "D_disagreement_loop": "Hit the 10-turn cap with ≥ 6 distinct large numbers in play — agents oscillated.",
    "E_early_termination": "Emotion stopped ≥ 2 turns earlier than control AND was wrong — premature halt.",
    "F_magnitude_error": "Predicted is off from gold by ≥10× (decimal-shift / unit slip) and emotion is wrong.",
    "G_semantic_reframing": "Moderate first-turn similarity (0.30–0.55), different answer — interpretation drifted.",
    "I_extraction_artifact": "Gold appears in the transcript text but the extractor returned a different number.",
    "J_other": "No diagnostic feature matched — flag for manual review.",
    "H_stylistic_only": "Outcome unchanged and predicted number matched control — emotion only changed surface form.",
}

# Cause colour palette (used for the bar segments).  Colourblind-aware-ish.
CAUSE_COLOR = {
    "A_decomposition_shift":  "#1f77b4",
    "B_arithmetic_divergence": "#ff7f0e",
    "C_verification_bypass":   "#2ca02c",
    "D_disagreement_loop":     "#d62728",
    "E_early_termination":     "#9467bd",
    "F_magnitude_error":       "#8c564b",
    "G_semantic_reframing":    "#e377c2",
    "I_extraction_artifact":   "#bcbd22",
    "J_other":                 "#7f7f7f",
    "H_stylistic_only":        "#c7c7c7",
}

OUTCOME_COLOR = {
    "helped":       "#2ca02c",
    "hurt":         "#d62728",
    "stay_correct": "#a8d5b3",
    "stay_wrong":   "#e8a8a8",
}


def esc(s) -> str:
    return html.escape(str(s) if s is not None else "—")


def render_transcript(transcript: list[str]) -> str:
    rows = []
    for line in transcript:
        # Lines look like "[01] alice: ..."
        if line.startswith("["):
            speaker = "alice" if "] alice:" in line.lower() else (
                "bob" if "] bob:" in line.lower() else "?"
            )
            rows.append(f'<div class="t t-{speaker}">{esc(line)}</div>')
        else:
            rows.append(f"<div class=\"t t-other\">{esc(line)}</div>")
    return "\n".join(rows)


def render_cause_ranking_section(by_panel: dict, ordering: str) -> str:
    """For one ordering, show 6 diverging bar charts (one per emotion).

    Each row: helped count grows leftward from the centre, hurt count grows
    rightward, the cause label sits in the middle. stay_correct and stay_wrong
    cells with the same cause label are reported as a small grey count after
    the cause name (they did not flip the outcome). H_stylistic_only is
    excluded from the chart entirely (no causal change).

    Bars share the same x-axis scale across all 6 emotions in one ordering so
    the panels are visually comparable side-by-side.
    """
    # Build per-(emo, cause) outcome counter and find the global max single-side
    # bar (helped or hurt) across this ordering, so we can scale all bars to a
    # common pixel width.
    per_emo: dict[str, dict[str, Counter]] = {}
    max_side = 0
    for emo in EMOTIONS:
        sub = by_panel[(ordering, emo)]
        emo_counter: dict[str, Counter] = defaultdict(Counter)
        for r in sub:
            if r["cause"] == "H_stylistic_only":
                continue
            emo_counter[r["cause"]][r["outcome_change"]] += 1
        for cause, oc in emo_counter.items():
            max_side = max(max_side, oc.get("helped", 0), oc.get("hurt", 0))
        per_emo[emo] = emo_counter
    if max_side == 0:
        max_side = 1

    parts = [f'<h3 class="rank-ord-h">{esc(ordering)}-first</h3>']
    for emo in EMOTIONS:
        sub = by_panel[(ordering, emo)]
        n_total = len(sub)
        counts = Counter(r["cause"] for r in sub)
        n_changed = n_total - counts.get("H_stylistic_only", 0)
        outcomes = Counter(r["outcome_change"] for r in sub)
        net = outcomes.get("helped", 0) - outcomes.get("hurt", 0)
        net_class = "net-pos" if net > 0 else "net-neg" if net < 0 else "net-zero"

        emo_counter = per_emo[emo]
        # Order causes by total impact (helped + hurt) descending. Ties broken
        # by hurt count (more disruptive shown first), then by alphabetical key.
        ranked = sorted(
            emo_counter.items(),
            key=lambda kv: (
                -(kv[1].get("helped", 0) + kv[1].get("hurt", 0)),
                -kv[1].get("hurt", 0),
                kv[0],
            ),
        )

        rows = []
        for cause, oc in ranked:
            helped = oc.get("helped", 0)
            hurt = oc.get("hurt", 0)
            stay = oc.get("stay_correct", 0) + oc.get("stay_wrong", 0)
            if helped == 0 and hurt == 0 and stay == 0:
                continue
            helped_w = 100.0 * helped / max_side
            hurt_w = 100.0 * hurt / max_side
            stay_note = (
                f' <span class="rank-stay" title="{stay} cell(s) with this cause but '
                f'outcome unchanged (stay_correct/stay_wrong)">+{stay}</span>'
                if stay > 0 else ""
            )
            rows.append(f"""
<div class="div-row">
  <div class="side helped-side">
    <div class="bar helped-bar" style="width:{helped_w:.2f}%;background:{OUTCOME_COLOR['helped']}">
      {f'<span class="bar-num">{helped}</span>' if helped > 0 else ''}
    </div>
  </div>
  <div class="centre" style="border-left:4px solid {CAUSE_COLOR[cause]}">
    <span class="cname">{esc(CAUSE_LABEL[cause])}</span>{stay_note}
  </div>
  <div class="side hurt-side">
    <div class="bar hurt-bar" style="width:{hurt_w:.2f}%;background:{OUTCOME_COLOR['hurt']}">
      {f'<span class="bar-num">{hurt}</span>' if hurt > 0 else ''}
    </div>
  </div>
</div>
""")

        emo_anchor_link = f"#{ordering}-{emo.replace('+','p').replace('-','m')}"
        if not rows:
            body = '<p class="muted" style="padding:0.4em">no changed cells</p>'
        else:
            body = (
                '<div class="div-header"><div class="hdr-l">helped (ctrl wrong → emo right)</div>'
                '<div class="hdr-c">cause</div>'
                '<div class="hdr-r">hurt (ctrl right → emo wrong)</div></div>'
                + "".join(rows)
            )
        parts.append(
            f'<div class="rank-emo">'
            f'<div class="rank-emo-head">'
            f'<strong>{esc(emo)}</strong> '
            f'<span class="muted">{n_changed}/{n_total} cells changed</span> · '
            f'<span class="muted">{outcomes.get("helped",0)} helped, {outcomes.get("hurt",0)} hurt, '
            f'<span class="{net_class}">net Δ {net:+d}</span></span> · '
            f'<a class="jump" href="{emo_anchor_link}">jump to transcripts ↓</a>'
            f'</div>'
            f'<div class="div-bars">{body}</div>'
            f'</div>'
        )
    return "<div class=\"rank-ord\">" + "".join(parts) + "</div>"


def cause_bar(counts: dict[str, int], total: int) -> str:
    if total == 0:
        return '<div class="cause-bar empty">no cells</div>'
    segs = []
    for cause in CAUSE_ORDER:
        n = counts.get(cause, 0)
        if n == 0:
            continue
        pct = 100.0 * n / total
        segs.append(
            f'<span class="seg" style="width:{pct:.2f}%;background:{CAUSE_COLOR[cause]}" '
            f'title="{esc(CAUSE_LABEL[cause])}: {n} ({pct:.1f}%)"></span>'
        )
    return f'<div class="cause-bar">{"".join(segs)}</div>'


def outcome_pill(outcome: str) -> str:
    label = {"helped": "helped", "hurt": "hurt", "stay_correct": "stay✓", "stay_wrong": "stay✗"}.get(
        outcome, outcome
    )
    return (
        f'<span class="pill" style="background:{OUTCOME_COLOR.get(outcome, "#ccc")}">'
        f"{esc(label)}</span>"
    )


def cause_pill(cause: str) -> str:
    return (
        f'<span class="pill" style="background:{CAUSE_COLOR.get(cause, "#999")}">'
        f"{esc(CAUSE_LABEL.get(cause, cause))}</span>"
    )


def render_sample_row(rec: dict, ctrl_rec: dict | None, emo_rec: dict, problem: dict) -> str:
    """Collapsed row + expandable details with side-by-side transcripts."""
    cause = rec["cause"]
    feats = {k: rec[k] for k in (
        "outcome_change",
        "control_predicted",
        "emotion_predicted",
        "control_n_turns",
        "emotion_n_turns",
        "first_turn_jaccard",
        "mag_ratio",
        "distinct_big_nums",
        "agreement_words_emotion",
        "agreement_words_control",
        "disagreement_words_emotion",
        "gold_in_last_turns",
        "emotion_extract_strategy",
        "control_extract_strategy",
        "alice_trait_mean_emotion",
        "alice_trait_mean_control",
        "bob_trait_drift_emotion",
        "control_failure",
        "emotion_failure",
    ) if k in rec}
    # Summary line
    summary = (
        f'#{rec["idx"]}  '
        f'gold={esc(rec["gold"])}  '
        f'ctrl_pred={esc(rec["control_predicted"])}  emo_pred={esc(rec["emotion_predicted"])}  '
        f'(turns {rec["control_n_turns"]}→{rec["emotion_n_turns"]}, '
        f'J={rec["first_turn_jaccard"]:.2f})'
    )
    feats_lines = [f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in feats.items()]
    feats_html = '<dl class="feats">' + "".join(feats_lines) + "</dl>"

    ctrl_transcript = render_transcript(ctrl_rec["transcript"] if ctrl_rec else [])
    emo_transcript = render_transcript(emo_rec["transcript"])

    return f"""
<details class="sample">
  <summary>
    {outcome_pill(rec["outcome_change"])}
    {cause_pill(cause)}
    <code class="snip">{esc(summary)}</code>
    <span class="qpreview">{esc(rec["question"][:140])}{"…" if len(rec["question"])>140 else ""}</span>
  </summary>
  <div class="sample-body">
    <div class="qfull"><strong>Q (idx {rec["idx"]}, gold={esc(rec["gold"])}):</strong> {esc(problem["question"])}</div>
    {feats_html}
    <div class="cmp">
      <div class="col">
        <h4>control transcript ({rec["control_n_turns"]} turns)</h4>
        {ctrl_transcript}
      </div>
      <div class="col">
        <h4>{esc(rec["emotion"])} transcript ({rec["emotion_n_turns"]} turns)</h4>
        {emo_transcript}
      </div>
    </div>
  </div>
</details>
"""


def build():
    causes = json.loads((ANA / "causes_n200.json").read_text())
    alice_src = json.loads((ANA / "snapshot_alice_n200.json").read_text())
    bob_src = json.loads((ANA / "snapshot_bob_n200.json").read_text())
    problems = json.loads((ANA / "snapshot_problems_n200.json").read_text())

    # Index causes by (ordering, emotion).
    by_panel: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in causes:
        by_panel[(r["ordering"], r["emotion"])].append(r)
    for k in by_panel:
        by_panel[k].sort(key=lambda r: (r["outcome_change"] == "stay_correct",
                                         r["outcome_change"] == "stay_wrong",
                                         CAUSE_ORDER.index(r["cause"]),
                                         r["idx"]))

    # ---- Build nav + panels ----
    nav_links = []
    for ordering in ORDERINGS:
        nav_links.append(f'<span class="nav-group">{ordering}-first:</span>')
        for emo in EMOTIONS:
            anchor = f"{ordering}-{emo.replace('+','p').replace('-','m')}"
            nav_links.append(f'<a href="#{anchor}">{esc(emo)}</a>')
    nav_html = " · ".join(nav_links)

    # Global cause distribution
    global_counts = Counter(r["cause"] for r in causes)
    total = len(causes)
    global_table_rows = []
    for cause in CAUSE_ORDER:
        n = global_counts.get(cause, 0)
        pct = 100.0 * n / total if total else 0
        global_table_rows.append(
            f"<tr>"
            f'<td><span class="swatch" style="background:{CAUSE_COLOR[cause]}"></span>{esc(CAUSE_LABEL[cause])}</td>'
            f"<td>{n}</td><td>{pct:.1f}%</td>"
            f"<td class=\"blurb\">{esc(CAUSE_BLURB[cause])}</td>"
            f"</tr>"
        )

    # Outcome-change table
    outcome_counts = Counter(r["outcome_change"] for r in causes)
    outcome_rows = "".join(
        f"<tr><td>{outcome_pill(o)}</td><td>{outcome_counts.get(o, 0)}</td>"
        f"<td>{100.0 * outcome_counts.get(o,0) / total:.1f}%</td></tr>"
        for o in ("helped", "hurt", "stay_correct", "stay_wrong")
    )

    # Per-emotion delta-acc summary (helped − hurt) per ordering
    helped_hurt_rows = []
    for ordering in ORDERINGS:
        helped_hurt_rows.append(f'<tr><td colspan="5"><strong>{ordering}-first</strong></td></tr>')
        for emo in EMOTIONS:
            sub = by_panel[(ordering, emo)]
            h = sum(1 for r in sub if r["outcome_change"] == "helped")
            x = sum(1 for r in sub if r["outcome_change"] == "hurt")
            sc = sum(1 for r in sub if r["outcome_change"] == "stay_correct")
            sw = sum(1 for r in sub if r["outcome_change"] == "stay_wrong")
            net = h - x
            delta_class = "net-pos" if net > 0 else "net-neg" if net < 0 else "net-zero"
            helped_hurt_rows.append(
                f"<tr>"
                f"<td>{esc(emo)}</td>"
                f"<td>{h}</td><td>{x}</td>"
                f"<td>{sc}</td><td>{sw}</td>"
                f'<td class="{delta_class}">{net:+d}</td>'
                f"</tr>"
            )

    # Build panels
    panel_html_parts = []
    for ordering in ORDERINGS:
        panel_html_parts.append(f'<h2 class="ord-h">{esc(ordering)}-first</h2>')
        src = alice_src if ordering == "alice" else bob_src
        for emo in EMOTIONS:
            anchor = f"{ordering}-{emo.replace('+','p').replace('-','m')}"
            sub = by_panel[(ordering, emo)]
            counts = Counter(r["cause"] for r in sub)
            outcomes = Counter(r["outcome_change"] for r in sub)
            n = len(sub)
            # cause distribution stacked bar
            bar = cause_bar(counts, n)

            # Cause-count list, sorted by frequency descending (excluding the
            # H_stylistic_only "no-change" bucket from the headline ranking).
            ranked = sorted(
                ((c, counts.get(c, 0)) for c in CAUSE_ORDER if counts.get(c, 0) > 0),
                key=lambda kv: (kv[0] == "H_stylistic_only", -kv[1]),
            )
            cause_chips = []
            for c, cnt in ranked:
                cause_chips.append(
                    f'<span class="chip" style="border-left:6px solid {CAUSE_COLOR[c]}">'
                    f'<strong>{cnt}</strong> {esc(CAUSE_LABEL[c])}</span>'
                )
            chips_html = " ".join(cause_chips)

            # Per-sample rows for FLIPS only (helped + hurt).
            ctrl_rs = src.get("control", [])
            emo_rs = src.get(emo, [])
            sample_rows = []
            for rec in sub:
                if rec["outcome_change"] not in ("helped", "hurt"):
                    continue
                idx = rec["idx"]
                # `idx` is the problem id, but the snapshot lists are aligned by
                # position. Find the position whose problems[i]['idx'] == idx.
                # In practice they're sequential, so just look up.
                ctrl_r = ctrl_rs[idx] if idx < len(ctrl_rs) else None
                emo_r = emo_rs[idx] if idx < len(emo_rs) else None
                problem = problems[idx] if idx < len(problems) else {"question": ""}
                if ctrl_r is None or emo_r is None:
                    continue
                sample_rows.append(render_sample_row(rec, ctrl_r, emo_r, problem))

            flips_inner = "".join(sample_rows) if sample_rows else '<p class="muted">no flips</p>'
            panel_html_parts.append(f"""
<section class="panel" id="{anchor}">
  <header class="panel-head">
    <h3>{esc(ordering)}-first / {esc(emo)} <span class="muted">({n} cells)</span></h3>
    <div class="panel-outcomes">
      {outcome_pill("helped")} {outcomes.get("helped",0)}
      {outcome_pill("hurt")} {outcomes.get("hurt",0)}
      {outcome_pill("stay_correct")} {outcomes.get("stay_correct",0)}
      {outcome_pill("stay_wrong")} {outcomes.get("stay_wrong",0)}
      <span class="net">net Δ-acc: <strong>{outcomes.get("helped",0)-outcomes.get("hurt",0):+d}</strong></span>
    </div>
  </header>
  {bar}
  <div class="chips">{chips_html}</div>
  <details class="flips" open>
    <summary>{len(sample_rows)} flipped samples (helped + hurt) — click each to expand transcripts</summary>
    {flips_inner}
  </details>
</section>
""")

    # ---- Stylesheet (inline for self-contained file) ----
    css = """
:root {
  --fg: #222;
  --muted: #666;
  --bg: #fff;
  --bg2: #f6f6f8;
  --line: #ddd;
  --code-bg: #f1f3f5;
}
* { box-sizing: border-box; }
body {
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--fg); background: var(--bg);
  max-width: 1400px; margin: 0 auto; padding: 1.5em;
}
h1, h2, h3, h4 { margin: 0.6em 0 0.4em; }
h1 { font-size: 1.6em; }
h2.ord-h { font-size: 1.3em; border-bottom: 2px solid var(--line); padding-bottom: 0.2em; margin-top: 1.5em; }
.muted { color: var(--muted); }
nav.top { position: sticky; top: 0; background: var(--bg); padding: 0.5em 0; border-bottom: 1px solid var(--line); margin-bottom: 1em; font-size: 13px; z-index: 10; }
nav.top a { margin: 0 0.3em; color: #1f77b4; text-decoration: none; }
nav.top .nav-group { font-weight: 600; margin-left: 0.6em; color: var(--fg); }
section.summary { background: var(--bg2); padding: 1em; border-radius: 6px; margin: 1em 0; }
table { border-collapse: collapse; margin: 0.5em 0; }
table.gtable td, table.gtable th { padding: 4px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
table.gtable th { text-align: left; background: var(--bg2); }
.swatch { display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle; border-radius: 2px; }
.cause-bar { display: flex; height: 18px; width: 100%; border-radius: 3px; overflow: hidden; background: var(--bg2); margin: 0.5em 0; }
.cause-bar .seg { display: block; height: 100%; }
.cause-bar.empty { color: var(--muted); padding: 0.2em 0.5em; }
.chips { display: flex; flex-wrap: wrap; gap: 4px; margin: 0.4em 0 0.8em; }
.chip { background: var(--bg2); padding: 2px 8px; border-radius: 3px; font-size: 12px; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; color: #1a1a1a; margin-right: 4px; }
.panel { border: 1px solid var(--line); border-radius: 6px; padding: 0.8em 1em; margin: 0.7em 0; background: #fafafa; }
.panel-head { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; }
.panel-outcomes { font-size: 12px; color: var(--muted); }
.panel-outcomes .net { margin-left: 1em; }
.net-pos { color: #2ca02c; }
.net-neg { color: #d62728; }
.net-zero { color: var(--muted); }
details.flips > summary { cursor: pointer; padding: 0.3em 0; color: var(--muted); }
details.sample { border-top: 1px solid var(--line); padding: 0.3em 0; }
details.sample > summary { cursor: pointer; list-style: none; }
details.sample > summary::marker, details.sample > summary::-webkit-details-marker { display: none; }
details.sample > summary { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.sample-body { padding: 0.6em 0.4em; border-top: 1px dashed var(--line); margin-top: 0.4em; background: #fff; }
.snip { font-family: ui-monospace, SF Mono, Menlo, Consolas, monospace; font-size: 11px; background: var(--code-bg); padding: 1px 6px; border-radius: 3px; }
.qpreview { font-style: italic; color: var(--muted); flex: 1 1 50%; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.qfull { background: var(--bg2); padding: 0.5em 0.8em; border-left: 4px solid var(--line); margin: 0.4em 0; }
dl.feats { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; font-size: 12px; margin: 0.4em 0; }
dl.feats dt { color: var(--muted); }
.cmp { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 0.4em; }
.cmp .col { background: #fff; border: 1px solid var(--line); padding: 0.5em 0.8em; border-radius: 4px; max-height: 60vh; overflow-y: auto; }
.cmp .col h4 { font-size: 12px; color: var(--muted); margin: 0 0 0.4em; }
.t { font-family: ui-monospace, SF Mono, Menlo, Consolas, monospace; font-size: 11.5px; padding: 2px 4px; margin: 1px 0; border-radius: 2px; white-space: pre-wrap; word-break: break-word; }
.t-alice { background: #fff5f0; }
.t-bob { background: #f0f5ff; }
.t-other { color: var(--muted); }
table.gtable td.blurb { color: var(--muted); font-size: 12.5px; max-width: 600px; }
section.ranking { margin: 1.5em 0; padding: 1em 1.2em; background: var(--bg2); border-radius: 6px; }
.rank-ord { margin: 0.6em 0; }
.rank-ord-h { font-size: 1.05em; margin: 0.6em 0 0.4em; padding: 0.2em 0.6em; background: #2c3e50; color: #fff; border-radius: 4px; display: inline-block; }
.rank-emo { margin: 0.5em 0 1em; padding: 0.5em 0.7em; background: #fff; border: 1px solid var(--line); border-radius: 5px; }
.rank-emo-head { font-size: 13px; margin-bottom: 0.4em; }
.rank-emo-head .jump { float: right; font-size: 12px; color: #1f77b4; text-decoration: none; }
.rank-emo-head .jump:hover { text-decoration: underline; }
.rank-bars { display: flex; flex-direction: column; gap: 3px; }
.rank-row { display: grid; grid-template-columns: 200px 1fr 60px; align-items: center; gap: 8px; font-size: 12px; }
.rank-label { color: var(--fg); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rank-track { background: #eee; border-radius: 2px; height: 16px; position: relative; overflow: hidden; }
.rank-fill { height: 100%; min-width: 18px; display: flex; align-items: center; justify-content: flex-end; padding: 0 6px; box-sizing: border-box; }
.rank-num { color: #fff; font-weight: 600; font-size: 11px; text-shadow: 0 1px 1px rgba(0,0,0,0.4); }
.rank-pct { font-size: 11px; color: var(--muted); text-align: right; }
/* Diverging (helped left, hurt right) bar chart */
.div-bars { font-size: 12px; }
.div-header { display: grid; grid-template-columns: 1fr 220px 1fr; align-items: center; padding: 0 0.4em 0.3em; color: var(--muted); font-size: 11px; border-bottom: 1px solid var(--line); margin-bottom: 4px; }
.div-header .hdr-l { text-align: right; padding-right: 8px; }
.div-header .hdr-c { text-align: center; }
.div-header .hdr-r { text-align: left; padding-left: 8px; }
.div-row { display: grid; grid-template-columns: 1fr 220px 1fr; align-items: center; gap: 0; padding: 1px 0; }
.div-row .side { height: 18px; display: flex; align-items: center; }
.div-row .helped-side { justify-content: flex-end; padding-right: 0; }
.div-row .hurt-side { justify-content: flex-start; padding-left: 0; }
.div-row .bar { height: 100%; min-width: 0; display: flex; align-items: center; padding: 0 6px; box-sizing: border-box; border-radius: 2px; }
.div-row .helped-bar { justify-content: flex-start; border-top-right-radius: 0; border-bottom-right-radius: 0; }
.div-row .hurt-bar { justify-content: flex-end; border-top-left-radius: 0; border-bottom-left-radius: 0; }
.div-row .bar-num { color: #1a1a1a; font-weight: 600; font-size: 11px; }
.div-row .centre { padding: 0 8px; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; background: #fafafa; }
.div-row .cname { font-size: 12px; color: var(--fg); }
.div-row .rank-stay { font-size: 10px; color: var(--muted); margin-left: 4px; }
"""

    # ---- Methodology blurb ----
    methodology = """
<p>Each <em>cell</em> is one (problem, ordering, emotion) triple compared against the same-ordering
<code>control</code>. We compute a small set of mechanical features and apply a priority-ordered
heuristic decision tree to attach a single high-level cause label per cell. Cells whose outcome did
not flip and whose predicted number matches control are tagged <code>H_stylistic_only</code>; cells
where the outcome flipped (helped / hurt) get the most diagnostic non-stylistic tag that fires.
First-turn similarity is computed on <strong>Alice's</strong> first speaking turn (always — she's
the steered agent), so the metric is symmetric across alice-first and bob-first orderings.</p>
"""

    # ---- Final HTML ----
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>n=200 cause analysis — task_sweep</title>
<style>{css}</style>
</head>
<body>
<h1>n=200 cause analysis — task_sweep (first 200 GSM8K problems)</h1>
<p class="muted">2 orderings (alice-first, bob-first) × 6 emotion conditions × 200 problems = 2,400 cells.
Each cell is paired with the same-ordering <code>control</code> for diff-based cause attribution.</p>

<nav class="top">{nav_html}</nav>

<section class="summary">
  <h2>Methodology</h2>
  {methodology}

  <h2>Cause taxonomy</h2>
  <table class="gtable">
    <thead><tr><th>cause</th><th>n</th><th>%</th><th>definition</th></tr></thead>
    <tbody>{"".join(global_table_rows)}</tbody>
  </table>

  <h2>Outcome distribution</h2>
  <table class="gtable">
    <thead><tr><th>outcome</th><th>n</th><th>%</th></tr></thead>
    <tbody>{outcome_rows}</tbody>
  </table>

  <h2>Per-(ordering, emotion) helped vs hurt</h2>
  <table class="gtable">
    <thead><tr><th>condition</th><th>helped</th><th>hurt</th><th>stay✓</th><th>stay✗</th><th>net Δ</th></tr></thead>
    <tbody>{"".join(helped_hurt_rows)}</tbody>
  </table>
</section>

<section class="ranking">
  <h2>Cause ranking — what each emotion changed most vs control</h2>
  <p class="muted">For each (ordering, emotion), the 9 non-stylistic causes are ranked by how often they explain the change vs the same-ordering <code>control</code>. <code>H. Stylistic only</code> (no outcome change, same predicted) is excluded from these bars but reported in the header as <em>n cells changed / total</em>. Bars share an x-axis scale per ordering so emotions are comparable side-by-side.</p>
  {render_cause_ranking_section(by_panel, "alice")}
  {render_cause_ranking_section(by_panel, "bob")}
</section>

{"".join(panel_html_parts)}

<footer style="margin-top:2em;color:var(--muted);font-size:12px">
  Generated from <code>analysis/causes_n200.json</code>; transcripts from
  <code>analysis/snapshot_*_n200.json</code>.
</footer>
</body></html>"""

    out = ANA / "n200_cause_analysis.html"
    out.write_text(html_doc, encoding="utf-8")
    print(f"wrote {out}  ({len(html_doc):,} bytes)")


if __name__ == "__main__":
    build()
