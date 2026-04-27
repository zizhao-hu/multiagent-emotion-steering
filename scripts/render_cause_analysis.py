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
# Canonical storage uses "alice" / "bob" because that is what the transcripts
# actually contain ([01] alice: ...). For rendering we re-label by role:
#   alice = emotional agent (the steered one)
#   bob   = stable agent    (unsteered)
ORDERING_LABEL = {
    "alice": "emotional-first",
    "bob": "stable-first",
}
ORDERING_DESC = {
    "alice": "emotional agent opens the conversation",
    "bob": "stable agent opens the conversation",
}

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
    "A_decomposition_shift": "A. Emotional agent pivots to a new angle",
    "B_arithmetic_divergence": "B. Same setup, computation lands elsewhere",
    "C_verification_bypass": "C. Stable agent agrees without re-deriving",
    "D_disagreement_loop": "D. Many number proposals, no convergence",
    "E_early_termination": "E. Emotional agent commits to a final answer early",
    "F_magnitude_error": "F. Calculation slip — answer off by 10×+",
    "G_semantic_reframing": "G. Question re-read differently",
    "I_extraction_artifact": "I. Right number in dialogue, never declared",
    "J_other": "J. Doesn't fit any pattern",
    "H_stylistic_only": "H. Same answer, different wording",
}

# Each blurb states: (a) the precise feature threshold that fires the label,
# (b) which agent is doing what, (c) what the heuristic does NOT measure —
# so the reader knows when the label is mechanism vs signature.
CAUSE_BLURB = {
    "A_decomposition_shift": (
        "The <strong>emotional agent</strong>'s first speaking turn shares <em>&lt;30% of "
        "tokens</em> with control (Jaccard similarity). She approaches the problem "
        "from a different angle — different operands picked first, different sub-step, "
        "different way of breaking it down. "
        "In <em>emotional-first</em> runs this is her conversation opener; in "
        "<em>stable-first</em> runs it is her first <em>reply</em> (turn [02]) — "
        "she pivots despite seeing the stable agent's neutral opener."
    ),
    "B_arithmetic_divergence": (
        "<em>≥55% first-turn token overlap</em> with control — the emotional agent reads "
        "the problem the same way and starts with the same setup — but the <em>final "
        "answer differs</em>. The computation chain forks somewhere in the middle "
        "(usually a single slip: swapped operand, missed term, off-by-one). The heuristic "
        "does <em>not</em> tell us where the divergence happens, and does <em>not</em> "
        "imply 'more thinking' or 'better logic' — read the transcript pair to see which "
        "step went a different way."
    ),
    "C_verification_bypass": (
        "Emotion conversation is <em>shorter</em> than control AND has more agreement-pattern "
        "matches (<code>agreed / correct / exactly / yes / that's right …</code>) AND the "
        "outcome flipped. The <strong>stable agent</strong> takes the emotional agent's "
        "number on faith instead of re-deriving it; the conversation wraps quickly. When "
        "the emotional agent is right this helps; when she's wrong the stable agent "
        "doesn't catch it. Colourful name, but the measurable mechanism is just: less "
        "checking happened."
    ),
    "D_disagreement_loop": (
        "Conversation hits the <em>10-turn cap</em> AND ≥6 distinct numbers &gt;1 appear "
        "across the transcript. The two agents keep proposing different totals and never "
        "agree on one; the run terminates by exhaustion. No clean final answer."
    ),
    "E_early_termination": (
        "Conversation ends <em>≥2 turns earlier</em> than control AND the emotional agent "
        "is wrong. She wrote <code>Final answer: X</code> (or equivalent) before the work "
        "was done — usually after one or two proposals — so the stable agent had no chance "
        "to push back."
    ),
    "F_magnitude_error": (
        "The predicted answer is off from gold by <em>≥10×</em> (or ≤1/10×) — "
        "order-of-magnitude gap. This is a <em>signature</em>, not a mechanism: a 10×+ "
        "gap is almost always one identifiable arithmetic step (forgot to divide, "
        "double-counted a multiplier, mixed grams with kg), but the heuristic doesn't say "
        "<em>which</em> step. Read the transcript pair to see the specific slip."
    ),
    "G_semantic_reframing": (
        "First-turn token overlap is in the middle range (<em>30–55%</em>) — neither a "
        "fresh angle (A) nor the same setup (B). Some words and numbers are shared with "
        "control, but the emotional agent has interpreted <em>what the question is "
        "asking</em> in a different way. Same operands in play, different quantity being "
        "computed."
    ),
    "I_extraction_artifact": (
        "The gold number <em>does</em> appear in the last 2 turns of the dialogue, but "
        "the extractor fell back to <code>last-number-in-tail</code> because there was no "
        "<code>Final answer: X</code> / <code>\\boxed{X}</code> / <code>the answer is X</code> "
        "declarative phrase. The agents reached the right number; they just never formally "
        "announced it. Often a 10-turn-cap artifact."
    ),
    "J_other": (
        "Mixed signals — none of the diagnostic features fired clearly. A small bucket "
        "(n &lt; 20 across the whole sweep); flag for manual review."
    ),
    "H_stylistic_only": (
        "Same final number as control, just different wording or tone. Outcome unchanged. "
        "The emotion changed how the agent said it, not what she said. Excluded from the "
        "cause-ranking bars and the margin matrix by definition (no helped/hurt outcome "
        "can come from this bucket)."
    ),
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

# High-level "kind of behavior" grouping over the 8 active causes.
# H_stylistic_only is excluded from analysis (same predicted answer; no outcome
# change by definition), so it has no meta-tag.
CAUSE_META = {
    "A_decomposition_shift":   "exploring",
    "B_arithmetic_divergence": "reasoning",
    "C_verification_bypass":   "manipulating",
    "D_disagreement_loop":     "exploring",
    "E_early_termination":     "manipulating",
    "F_magnitude_error":       "reasoning",
    "G_semantic_reframing":    "exploring",
    "I_extraction_artifact":   "artifact",
    "J_other":                 "artifact",
    # H_stylistic_only intentionally absent — excluded from analysis.
}

META_ORDER = ["reasoning", "manipulating", "exploring", "artifact"]

META_LABEL = {
    "reasoning":    "Reasoning",
    "manipulating": "Manipulating",
    "exploring":    "Exploring",
    "artifact":     "Artifact",
}

META_BLURB = {
    "reasoning": (
        "The math/logic chain itself diverged — same setup, different number "
        "comes out (B); or one step is so wrong the answer is 10×+ off (F)."
    ),
    "manipulating": (
        "One agent's social move shortcuts the dialogue — the stable agent "
        "capitulates without re-deriving (C), or the emotional agent "
        "declares an answer before the stable agent can push back (E)."
    ),
    "exploring": (
        "The emotional agent's framing or proposal-generation diverged — "
        "different opening angle (A), question re-read (G), or many "
        "different numbers proposed without convergence (D)."
    ),
    "artifact": (
        "Not a behavioral effect — measurement quirks. Right number reached "
        "but never formally declared (I), or signals too mixed to classify (J)."
    ),
}

META_COLOR = {
    "reasoning":    "#1f77b4",
    "manipulating": "#d62728",
    "exploring":    "#2ca02c",
    "artifact":     "#7f7f7f",
}


def meta_pill(meta: str) -> str:
    if meta is None or meta not in META_LABEL:
        return ""
    return (
        f'<span class="meta-pill" style="background:{META_COLOR[meta]}">'
        f'{esc(META_LABEL[meta])}</span>'
    )


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

    Each row has FOUR pieces, left-to-right:
      1. helped bar (green) growing leftward toward the centre
      2. cause label (centre)
      3. hurt bar (red) growing rightward away from the centre
      4. margin bar (= helped − hurt) on the far right; positive value
         (net-helpful) draws right in green, negative (net-harmful) draws
         right in red, zero draws as a neutral pip. The sign convention is
         intuition-aligned: + = more correct = green; − = more wrong = red.

    Causes are sorted by margin ASCENDING — most net-harmful (most negative)
    first, most net-helpful (most positive) last. Ties broken by larger
    total impact, then by cause key.

    Bars share scales across all 6 emotions in one ordering so the panels
    are visually comparable side-by-side.
    """
    # Build per-(emo, cause) outcome counter and compute scales:
    #   max_side   = largest single helped or hurt count (for the diverging bars)
    #   max_margin = largest absolute (helped - hurt) magnitude (for the margin bar)
    per_emo: dict[str, dict[str, Counter]] = {}
    max_side = 0
    max_margin = 0
    for emo in EMOTIONS:
        sub = by_panel[(ordering, emo)]
        emo_counter: dict[str, Counter] = defaultdict(Counter)
        for r in sub:
            if r["cause"] == "H_stylistic_only":
                continue
            emo_counter[r["cause"]][r["outcome_change"]] += 1
        for cause, oc in emo_counter.items():
            h = oc.get("helped", 0)
            x = oc.get("hurt", 0)
            max_side = max(max_side, h, x)
            max_margin = max(max_margin, abs(h - x))
        per_emo[emo] = emo_counter
    max_side = max(max_side, 1)
    max_margin = max(max_margin, 1)

    parts = [f'<h3 class="rank-ord-h" title="{esc(ORDERING_DESC[ordering])}">{esc(ORDERING_LABEL[ordering])}</h3>']
    for emo in EMOTIONS:
        sub = by_panel[(ordering, emo)]
        n_total = len(sub)
        counts = Counter(r["cause"] for r in sub)
        n_changed = n_total - counts.get("H_stylistic_only", 0)
        outcomes = Counter(r["outcome_change"] for r in sub)
        net = outcomes.get("helped", 0) - outcomes.get("hurt", 0)
        net_class = "net-pos" if net > 0 else "net-neg" if net < 0 else "net-zero"

        emo_counter = per_emo[emo]
        # Sort by margin (helped - hurt) ASCENDING — most net-harmful (most
        # negative) first, most net-helpful (most positive) last. Ties broken
        # by larger total volume, then alphabetical.
        ranked = sorted(
            emo_counter.items(),
            key=lambda kv: (
                kv[1].get("helped", 0) - kv[1].get("hurt", 0),
                -(kv[1].get("hurt", 0) + kv[1].get("helped", 0)),
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
            margin = helped - hurt  # + = more correct (green); − = more wrong (red)
            helped_w = 100.0 * helped / max_side
            hurt_w = 100.0 * hurt / max_side
            margin_w = 100.0 * abs(margin) / max_margin
            margin_color = (
                OUTCOME_COLOR["helped"] if margin > 0
                else OUTCOME_COLOR["hurt"] if margin < 0
                else "#bbb"
            )
            margin_sign = "+" if margin > 0 else ""
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
    <span class="cname">{esc(CAUSE_LABEL[cause])}</span> {meta_pill(CAUSE_META.get(cause))}{stay_note}
  </div>
  <div class="side hurt-side">
    <div class="bar hurt-bar" style="width:{hurt_w:.2f}%;background:{OUTCOME_COLOR['hurt']}">
      {f'<span class="bar-num">{hurt}</span>' if hurt > 0 else ''}
    </div>
  </div>
  <div class="side margin-side">
    <div class="bar margin-bar" style="width:{margin_w:.2f}%;background:{margin_color}">
      <span class="bar-num">{margin_sign}{margin}</span>
    </div>
  </div>
</div>
""")

        emo_anchor_link = f"#{ordering}-{emo.replace('+','p').replace('-','m')}"
        if not rows:
            body = '<p class="muted" style="padding:0.4em">no changed cells</p>'
        else:
            body = (
                '<div class="div-header">'
                '<div class="hdr-l">helped (ctrl wrong → emo right)</div>'
                '<div class="hdr-c">cause</div>'
                '<div class="hdr-r">hurt (ctrl right → emo wrong)</div>'
                '<div class="hdr-m">margin = helped − hurt</div>'
                '</div>'
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


def render_cause_matrix_section(by_panel: dict, ordering: str) -> str:
    """Render a cause × emotion heatmap for one ordering.

    Each cell shows the signed margin (helped − hurt) with:
      - background color: green intensity for net-helpful (+), red for
        net-harmful (−), neutral for zero (alpha proportional to |margin|
        / max_margin within this ordering)
      - main text: signed margin
      - subscript: helped/hurt counts as h13·x8

    Rows are grouped by meta-tag and within each group sorted by total
    |margin| across emotions descending — most volatile causes at the top.
    H_stylistic_only is excluded; any cause with zero helped+hurt across
    all emotions is also dropped.
    """
    # Build (cause, emotion) -> {helped, hurt} table for this ordering.
    table: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {emo: {"helped": 0, "hurt": 0} for emo in EMOTIONS}
    )
    for emo in EMOTIONS:
        for r in by_panel[(ordering, emo)]:
            cause = r["cause"]
            if cause == "H_stylistic_only":
                continue
            oc = r["outcome_change"]
            if oc in ("helped", "hurt"):
                table[cause][emo][oc] += 1

    # Drop causes with zero helped+hurt total across all emotions.
    causes_present = [
        c for c in CAUSE_ORDER
        if c != "H_stylistic_only"
        and any(table[c][e]["helped"] + table[c][e]["hurt"] for e in EMOTIONS)
    ]

    # Compute per-cell margin (helped − hurt) and global |margin| max.
    # Convention: + = more correct (green); − = more wrong (red).
    cell: dict[tuple[str, str], int] = {}
    max_abs_margin = 0
    for c in causes_present:
        for e in EMOTIONS:
            m = table[c][e]["helped"] - table[c][e]["hurt"]
            cell[(c, e)] = m
            if abs(m) > max_abs_margin:
                max_abs_margin = abs(m)
    max_abs_margin = max(max_abs_margin, 1)

    # Group causes by meta-tag, then sort within each group by Σ|m| descending.
    causes_by_meta: dict[str, list[str]] = {m: [] for m in META_ORDER}
    for c in causes_present:
        m = CAUSE_META.get(c)
        if m is not None:
            causes_by_meta[m].append(c)
    for m in META_ORDER:
        causes_by_meta[m].sort(
            key=lambda c: -sum(abs(cell[(c, e)]) for e in EMOTIONS)
        )

    # Column totals (across all causes) for the footer row.
    col_totals = {e: {"helped": 0, "hurt": 0} for e in EMOTIONS}
    for c in causes_present:
        for e in EMOTIONS:
            col_totals[e]["helped"] += table[c][e]["helped"]
            col_totals[e]["hurt"] += table[c][e]["hurt"]

    # Header row.
    head_cells = (
        '<th class="mtx-cause-h">cause</th>'
        + "".join(f'<th class="mtx-emo-h">{esc(e)}</th>' for e in EMOTIONS)
        + '<th class="mtx-row-tot">row Σ|m|</th>'
    )

    def _cell_html(m_val: int, h: int, x: int, ord_label: str, e: str, label: str) -> str:
        if h == 0 and x == 0:
            return '<td class="mtx-cell mtx-empty">·</td>'
        alpha = abs(m_val) / max_abs_margin if max_abs_margin else 0
        # m_val = helped − hurt: + = net-helpful (green), − = net-harmful (red).
        if m_val > 0:
            bg = f"rgba(44, 160, 44, {0.10 + 0.55 * alpha:.2f})"
        elif m_val < 0:
            bg = f"rgba(214, 39, 40, {0.10 + 0.55 * alpha:.2f})"
        else:
            bg = "rgba(180, 180, 180, 0.18)"
        sign = "+" if m_val > 0 else ""
        title = f"{ord_label} / {e} / {label}: helped={h}, hurt={x}, margin={sign}{m_val}"
        return (
            f'<td class="mtx-cell" style="background:{bg}" title="{esc(title)}">'
            f'<div class="mtx-margin">{sign}{m_val}</div>'
            f'<div class="mtx-hx">h{h}·x{x}</div>'
            f'</td>'
        )

    # Body rows: meta-tag section header → cause rows → meta-tag subtotal.
    body_rows = []
    for meta in META_ORDER:
        causes_in_meta = causes_by_meta[meta]
        if not causes_in_meta:
            continue

        # Section header row spanning all columns.
        ncols = 1 + len(EMOTIONS) + 1
        body_rows.append(
            f'<tr class="mtx-meta-head" style="--meta-col:{META_COLOR[meta]}">'
            f'<th colspan="{ncols}">'
            f'<span class="meta-pill" style="background:{META_COLOR[meta]}">{esc(META_LABEL[meta])}</span>'
            f'<span class="meta-blurb">{META_BLURB[meta]}</span>'
            f'</th>'
            f'</tr>'
        )

        # Per-cause rows within this meta-tag.
        meta_helped = {e: 0 for e in EMOTIONS}
        meta_hurt = {e: 0 for e in EMOTIONS}
        for c in causes_in_meta:
            row_abs_total = sum(abs(cell[(c, e)]) for e in EMOTIONS)
            cells_html = []
            for e in EMOTIONS:
                m_val = cell[(c, e)]
                h = table[c][e]["helped"]
                x = table[c][e]["hurt"]
                meta_helped[e] += h
                meta_hurt[e] += x
                cells_html.append(_cell_html(m_val, h, x, ORDERING_LABEL[ordering], e, CAUSE_LABEL[c]))
            body_rows.append(
                f'<tr>'
                f'<th class="mtx-row-h" style="border-left:4px solid {CAUSE_COLOR[c]}">'
                f'{esc(CAUSE_LABEL[c])}</th>'
                f'{"".join(cells_html)}'
                f'<td class="mtx-row-tot">{row_abs_total}</td>'
                f'</tr>'
            )

        # Subtotal row for this meta-tag (helped − hurt convention).
        sub_cells = []
        sub_row_abs = 0
        for e in EMOTIONS:
            h = meta_helped[e]
            x = meta_hurt[e]
            m_val = h - x
            sub_row_abs += abs(m_val)
            sub_cells.append(_cell_html(m_val, h, x, ORDERING_LABEL[ordering], e, f"Σ {META_LABEL[meta]}"))
        body_rows.append(
            f'<tr class="mtx-meta-sub">'
            f'<th class="mtx-row-h" style="border-left:4px solid {META_COLOR[meta]}">'
            f'<em>Σ {esc(META_LABEL[meta])}</em></th>'
            f'{"".join(sub_cells)}'
            f'<td class="mtx-row-tot">{sub_row_abs}</td>'
            f'</tr>'
        )

    # Footer row: column totals (helped − hurt convention).
    footer_cells = []
    for e in EMOTIONS:
        h = col_totals[e]["helped"]
        x = col_totals[e]["hurt"]
        m = h - x
        sign = "+" if m > 0 else ""
        text_color = "#2ca02c" if m > 0 else "#d62728" if m < 0 else "#888"
        footer_cells.append(
            f'<td class="mtx-cell mtx-foot">'
            f'<div class="mtx-margin" style="color:{text_color}">{sign}{m}</div>'
            f'<div class="mtx-hx">h{h}·x{x}</div>'
            f'</td>'
        )
    footer_row = (
        '<tr class="mtx-foot-row">'
        '<th class="mtx-row-h">column total (all causes)</th>'
        + "".join(footer_cells)
        + '<td class="mtx-row-tot">—</td>'
        '</tr>'
    )

    return (
        f'<div class="mtx-block">'
        f'<h3 class="mtx-ord-h" title="{esc(ORDERING_DESC[ordering])}">'
        f'{esc(ORDERING_LABEL[ordering])}</h3>'
        f'<table class="mtx">'
        f'<thead><tr>{head_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}{footer_row}</tbody>'
        f'</table>'
        f'</div>'
    )


def cause_bar(counts: dict[str, int], total: int) -> str:
    """Stacked-segment bar showing cause distribution among CHANGED cells only.

    H_stylistic_only is excluded (per analysis convention), so the bar
    shows only the behavioural divergences — % normalised over those.
    """
    changed_total = sum(
        counts.get(c, 0) for c in CAUSE_ORDER if c != "H_stylistic_only"
    )
    if changed_total == 0:
        return '<div class="cause-bar empty">no changed cells</div>'
    segs = []
    for cause in CAUSE_ORDER:
        if cause == "H_stylistic_only":
            continue
        n = counts.get(cause, 0)
        if n == 0:
            continue
        pct = 100.0 * n / changed_total
        segs.append(
            f'<span class="seg" style="width:{pct:.2f}%;background:{CAUSE_COLOR[cause]}" '
            f'title="{esc(CAUSE_LABEL[cause])}: {n} ({pct:.1f}% of changed cells)"></span>'
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
    # Summary line — answer + conversation stats compact deltas.
    we_d = rec.get("delta_words_emotional", 0)
    ws_d = rec.get("delta_words_stable", 0)
    summary = (
        f'#{rec["idx"]}  '
        f'gold={esc(rec["gold"])}  '
        f'ctrl_pred={esc(rec["control_predicted"])}  emo_pred={esc(rec["emotion_predicted"])}  '
        f'(turns {rec["control_n_turns"]}→{rec["emotion_n_turns"]}, '
        f'emo-w {rec.get("control_words_emotional",0)}→{rec.get("emotion_words_emotional",0)} ({we_d:+d}), '
        f'stable-w {rec.get("control_words_stable",0)}→{rec.get("emotion_words_stable",0)} ({ws_d:+d}), '
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
        nav_links.append(f'<span class="nav-group">{esc(ORDERING_LABEL[ordering])}:</span>')
        for emo in EMOTIONS:
            anchor = f"{ordering}-{emo.replace('+','p').replace('-','m')}"
            nav_links.append(f'<a href="#{anchor}">{esc(emo)}</a>')
    nav_html = " · ".join(nav_links)

    # Global cause distribution. Grouped by meta-tag; H_stylistic_only is
    # excluded entirely (no outcome change by definition — see footnote).
    global_counts = Counter(r["cause"] for r in causes)
    total = len(causes)
    causes_by_meta_global: dict[str, list[str]] = {m: [] for m in META_ORDER}
    for c in CAUSE_ORDER:
        if c == "H_stylistic_only":
            continue
        m = CAUSE_META.get(c)
        if m is not None:
            causes_by_meta_global[m].append(c)

    global_table_rows = []
    for meta in META_ORDER:
        causes_in_meta = causes_by_meta_global[meta]
        if not causes_in_meta:
            continue
        meta_n = sum(global_counts.get(c, 0) for c in causes_in_meta)
        meta_pct = 100.0 * meta_n / total if total else 0
        # Section header row.
        global_table_rows.append(
            f'<tr class="taxon-meta-head" style="--meta-col:{META_COLOR[meta]}">'
            f'<td colspan="4">'
            f'<span class="meta-pill" style="background:{META_COLOR[meta]}">{esc(META_LABEL[meta])}</span>'
            f' <span class="muted">({meta_n} cells, {meta_pct:.1f}% of all 2,400)</span>'
            f' — <span class="meta-blurb-inline">{META_BLURB[meta]}</span>'
            f'</td>'
            f'</tr>'
        )
        for cause in causes_in_meta:
            n = global_counts.get(cause, 0)
            pct = 100.0 * n / total if total else 0
            # CAUSE_BLURB strings contain inline HTML; render raw, do not escape.
            global_table_rows.append(
                f"<tr>"
                f'<td><span class="swatch" style="background:{CAUSE_COLOR[cause]}"></span>'
                f'{esc(CAUSE_LABEL[cause])}</td>'
                f"<td>{n}</td><td>{pct:.1f}%</td>"
                f'<td class="blurb">{CAUSE_BLURB[cause]}</td>'
                f"</tr>"
            )

    # Footer footnote about H_stylistic_only.
    h_n = global_counts.get("H_stylistic_only", 0)
    h_pct = 100.0 * h_n / total if total else 0
    global_table_rows.append(
        f'<tr class="taxon-footnote">'
        f'<td colspan="4" class="muted">'
        f'<em>Excluded from analysis:</em> '
        f'<strong>H. Same answer, different wording</strong> — {h_n} cells '
        f'({h_pct:.1f}% of all 2,400). Same predicted answer as control; outcome '
        f'unchanged by definition. Doesn\'t belong to any meta-tag (it is not a '
        f'behavioural divergence) and is dropped from the matrix and ranking bars.'
        f'</td>'
        f'</tr>'
    )

    # Outcome-change table
    outcome_counts = Counter(r["outcome_change"] for r in causes)
    outcome_rows = "".join(
        f"<tr><td>{outcome_pill(o)}</td><td>{outcome_counts.get(o, 0)}</td>"
        f"<td>{100.0 * outcome_counts.get(o,0) / total:.1f}%</td></tr>"
        for o in ("helped", "hurt", "stay_correct", "stay_wrong")
    )

    # ---- Conversation stats per (ordering, emotion) ----
    # Aggregate over all 200 problems in each cell. Show mean turns + mean
    # words per role, plus same-ordering control means and the delta.
    convo_rows = []
    for ordering in ORDERINGS:
        ord_label = ORDERING_LABEL[ordering]
        # Collect control means once per ordering
        ctrl_recs = [r for r in causes if r["ordering"] == ordering and r["emotion"] == "joy+"]
        # All records share the same control values per (ordering, problem),
        # so any emotion's records expose control_n_turns / control_words_*.
        if ctrl_recs:
            ctrl_mean_turns = sum(r["control_n_turns"] for r in ctrl_recs) / len(ctrl_recs)
            ctrl_mean_we = sum(r["control_words_emotional"] for r in ctrl_recs) / len(ctrl_recs)
            ctrl_mean_ws = sum(r["control_words_stable"] for r in ctrl_recs) / len(ctrl_recs)
        else:
            ctrl_mean_turns = ctrl_mean_we = ctrl_mean_ws = 0.0
        convo_rows.append(
            f'<tr><td colspan="8"><strong>{esc(ord_label)}</strong> '
            f'<span class="muted">— control baseline: {ctrl_mean_turns:.1f} turns, '
            f'{ctrl_mean_we:.0f} emo-agent words, {ctrl_mean_ws:.0f} stable-agent words '
            f'(per problem, mean over n=200)</span></td></tr>'
        )
        for emo in EMOTIONS:
            sub = by_panel[(ordering, emo)]
            if not sub:
                continue
            n = len(sub)
            mean_turns = sum(r["emotion_n_turns"] for r in sub) / n
            mean_we = sum(r["emotion_words_emotional"] for r in sub) / n
            mean_ws = sum(r["emotion_words_stable"] for r in sub) / n
            d_turns = mean_turns - ctrl_mean_turns
            d_we = mean_we - ctrl_mean_we
            d_ws = mean_ws - ctrl_mean_ws
            convo_rows.append(
                f"<tr>"
                f"<td>{esc(emo)}</td>"
                f"<td>{mean_turns:.1f}</td>"
                f'<td class="{ "delta-pos" if d_turns > 0.05 else "delta-neg" if d_turns < -0.05 else "delta-zero" }">'
                f"{d_turns:+.1f}</td>"
                f"<td>{mean_we:.0f}</td>"
                f'<td class="{ "delta-pos" if d_we > 1 else "delta-neg" if d_we < -1 else "delta-zero" }">'
                f"{d_we:+.0f}</td>"
                f"<td>{mean_ws:.0f}</td>"
                f'<td class="{ "delta-pos" if d_ws > 1 else "delta-neg" if d_ws < -1 else "delta-zero" }">'
                f"{d_ws:+.0f}</td>"
                f"<td>{mean_we + mean_ws:.0f}</td>"
                f"</tr>"
            )

    # Per-emotion delta-acc summary (helped − hurt) per ordering
    helped_hurt_rows = []
    for ordering in ORDERINGS:
        helped_hurt_rows.append(f'<tr><td colspan="6"><strong>{esc(ORDERING_LABEL[ordering])}</strong> <span class="muted">({esc(ORDERING_DESC[ordering])})</span></td></tr>')
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
        panel_html_parts.append(f'<h2 class="ord-h">{esc(ORDERING_LABEL[ordering])} <span class="muted" style="font-weight:normal;font-size:0.7em">— {esc(ORDERING_DESC[ordering])}</span></h2>')
        src = alice_src if ordering == "alice" else bob_src
        for emo in EMOTIONS:
            anchor = f"{ordering}-{emo.replace('+','p').replace('-','m')}"
            sub = by_panel[(ordering, emo)]
            counts = Counter(r["cause"] for r in sub)
            outcomes = Counter(r["outcome_change"] for r in sub)
            n = len(sub)
            # cause distribution stacked bar
            bar = cause_bar(counts, n)

            # Cause-count chips, sorted by frequency desc. H is excluded
            # entirely (it's the no-change bucket and not part of the analysis).
            ranked = sorted(
                ((c, counts.get(c, 0)) for c in CAUSE_ORDER
                 if c != "H_stylistic_only" and counts.get(c, 0) > 0),
                key=lambda kv: -kv[1],
            )
            cause_chips = []
            for c, cnt in ranked:
                cause_chips.append(
                    f'<span class="chip" style="border-left:6px solid {CAUSE_COLOR[c]}">'
                    f'<strong>{cnt}</strong> {esc(CAUSE_LABEL[c])} '
                    f'{meta_pill(CAUSE_META.get(c))}</span>'
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
    <h3>{esc(ORDERING_LABEL[ordering])} / {esc(emo)} <span class="muted">({n} cells)</span></h3>
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
/* Diverging (helped left, hurt right, margin far-right) bar chart */
.div-bars { font-size: 12px; }
.div-header { display: grid; grid-template-columns: 1fr 220px 1fr 110px; align-items: center; padding: 0 0.4em 0.3em; color: var(--muted); font-size: 11px; border-bottom: 1px solid var(--line); margin-bottom: 4px; gap: 6px; }
.div-header .hdr-l { text-align: right; padding-right: 8px; }
.div-header .hdr-c { text-align: center; }
.div-header .hdr-r { text-align: left; padding-left: 8px; }
.div-header .hdr-m { text-align: left; padding-left: 6px; border-left: 1px solid var(--line); }
.div-row { display: grid; grid-template-columns: 1fr 220px 1fr 110px; align-items: center; gap: 6px; padding: 1px 0; }
.div-row .side { height: 18px; display: flex; align-items: center; }
.div-row .helped-side { justify-content: flex-end; padding-right: 0; }
.div-row .hurt-side { justify-content: flex-start; padding-left: 0; }
.div-row .margin-side { justify-content: flex-start; padding-left: 6px; border-left: 1px solid var(--line); }
.div-row .bar { height: 100%; min-width: 0; display: flex; align-items: center; padding: 0 6px; box-sizing: border-box; border-radius: 2px; }
.div-row .helped-bar { justify-content: flex-start; border-top-right-radius: 0; border-bottom-right-radius: 0; }
.div-row .hurt-bar { justify-content: flex-end; border-top-left-radius: 0; border-bottom-left-radius: 0; }
.div-row .margin-bar { justify-content: flex-end; min-width: 28px; }
.div-row .bar-num { color: #1a1a1a; font-weight: 600; font-size: 11px; }
.div-row .centre { padding: 0 8px; text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; background: #fafafa; }
.div-row .cname { font-size: 12px; color: var(--fg); }
.div-row .rank-stay { font-size: 10px; color: var(--muted); margin-left: 4px; }
/* Tab UI */
.tabs { display: flex; gap: 4px; margin: 0.6em 0 0; border-bottom: 2px solid var(--line); }
.tab-btn { background: transparent; border: 1px solid var(--line); border-bottom: none; padding: 0.5em 1em; cursor: pointer; font: inherit; color: var(--muted); border-radius: 5px 5px 0 0; margin-bottom: -2px; }
.tab-btn.active { background: var(--bg); color: var(--fg); border-bottom: 2px solid var(--bg); font-weight: 600; }
.tab-btn:hover:not(.active) { background: var(--bg2); }
.tab-content { display: none; }
.tab-content.active { display: block; }
.legend-pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; color: #1a1a1a; margin: 0 2px; }
.delta-pos { color: #d62728; font-weight: 600; }
.delta-neg { color: #2ca02c; font-weight: 600; }
.delta-zero { color: var(--muted); }
table.gtable td:nth-child(n+2):nth-child(-n+8) { text-align: right; }
/* Meta-tag pills + grouped section headers */
.meta-pill { display: inline-block; padding: 1px 9px; border-radius: 10px; font-size: 11px; color: #fff; font-weight: 600; letter-spacing: 0.02em; vertical-align: middle; }
.meta-blurb { font-size: 11.5px; color: var(--muted); font-weight: normal; margin-left: 8px; }
.meta-blurb-inline { font-size: 12.5px; color: var(--muted); }
tr.taxon-meta-head td { background: var(--bg2); padding: 8px 10px; border-top: 2px solid #ccc; border-bottom: 1px solid var(--line); }
tr.taxon-footnote td { background: #fafafa; padding: 8px 10px; font-size: 12px; border-top: 2px dashed var(--line); }
tr.mtx-meta-head th { padding: 8px 6px 4px; text-align: left; background: transparent; border: none; }
tr.mtx-meta-sub th, tr.mtx-meta-sub td { background: #fafafa; border-top: 1px solid #ccc; border-bottom: 1px solid #ccc; }
tr.mtx-meta-sub td.mtx-cell { font-weight: 600; }
/* Cause × emotion margin matrix */
section.matrix { margin: 1.5em 0; padding: 1em 1.2em; background: var(--bg2); border-radius: 6px; }
.mtx-block { display: inline-block; vertical-align: top; margin: 0.4em 1em 0.6em 0; }
.mtx-ord-h { font-size: 1.0em; margin: 0.2em 0 0.4em; padding: 0.2em 0.6em; background: #2c3e50; color: #fff; border-radius: 4px; display: inline-block; }
table.mtx { border-collapse: separate; border-spacing: 2px; font-size: 12px; }
table.mtx th.mtx-cause-h { text-align: left; padding: 4px 8px; color: var(--muted); font-weight: 600; }
table.mtx th.mtx-emo-h { padding: 4px 6px; font-weight: 600; min-width: 64px; text-align: center; background: var(--bg); border-radius: 3px; }
table.mtx th.mtx-row-h { text-align: left; padding: 4px 8px 4px 10px; background: #fff; border-radius: 3px; max-width: 220px; min-width: 200px; font-weight: 500; }
table.mtx td.mtx-cell { width: 64px; min-width: 64px; padding: 4px 4px; text-align: center; border-radius: 3px; vertical-align: middle; }
table.mtx td.mtx-empty { color: #c8c8c8; background: #fafafa; }
table.mtx .mtx-margin { font-weight: 700; font-size: 13px; color: #1a1a1a; line-height: 1.1; }
table.mtx .mtx-hx { font-size: 10px; color: #555; line-height: 1.1; margin-top: 1px; }
table.mtx td.mtx-row-tot { font-size: 11px; color: var(--muted); padding: 4px 8px; text-align: right; }
table.mtx tr.mtx-foot-row th, table.mtx tr.mtx-foot-row td.mtx-cell { border-top: 2px solid #aaa; background: #fff; padding-top: 6px; }
table.mtx td.mtx-foot { background: #fff; }
.mtx-legend { font-size: 11px; color: var(--muted); margin: 0.4em 0 0.8em; }
.mtx-legend .swatch-mtx { display: inline-block; width: 14px; height: 12px; vertical-align: middle; margin: 0 4px; border-radius: 2px; border: 1px solid rgba(0,0,0,0.06); }
"""

    # ---- Methodology blurb ----
    methodology = """
<p>Each <em>cell</em> is one (problem, ordering, emotion) triple compared against the same-ordering
<code>control</code>. We compute a small set of mechanical features and apply a priority-ordered
heuristic decision tree to attach a single high-level cause label per cell.</p>
<p>Two roles in every conversation:</p>
<ul>
  <li><strong>Emotional agent</strong> — receives steering on the residual stream toward the target trait
      (joy+, sadness+, anger+, etc.). This is the agent whose internal state is being perturbed.</li>
  <li><strong>Stable agent</strong> — no steering applied; behaves as it would on the control. Acts as
      the verifier / second opinion in the dialogue.</li>
</ul>
<p>The <em>ordering</em> says who opens the conversation — when the <strong>emotional agent</strong>
opens (<em>emotional-first</em>) she sets the framing; when the <strong>stable agent</strong> opens
(<em>stable-first</em>) the emotional agent reacts to a neutral framing. First-turn similarity for
the cause heuristic is always anchored on the emotional agent's first speaking turn, so the metric
is symmetric across the two orderings.</p>
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
<p class="muted">2 orderings (<strong>emotional-first</strong> = the steered <em>emotional agent</em> opens the conversation; <strong>stable-first</strong> = the unsteered <em>stable agent</em> opens) × 6 emotion conditions × 200 problems = 2,400 cells.
Each cell is paired with the same-ordering <code>control</code> for diff-based cause attribution.</p>

<nav class="top">{nav_html}</nav>

<div class="tabs" role="tablist">
  <button class="tab-btn active" data-tab="summary" role="tab" aria-selected="true">Summary</button>
  <button class="tab-btn" data-tab="examples" role="tab" aria-selected="false">Examples ({len(causes)} cells, click rows to expand)</button>
</div>

<div id="tab-summary" class="tab-content active">

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

  <h2>Conversation stats — turns and word counts vs control</h2>
  <p class="muted">All values are means over the 200 problems in each cell. Δ is the difference between the emotion condition and the same-ordering control. The emotional agent is the steered one; the stable agent is unsteered.</p>
  <table class="gtable">
    <thead>
      <tr>
        <th rowspan="2">emotion</th>
        <th colspan="2">turns / problem</th>
        <th colspan="2">emo-agent words / problem</th>
        <th colspan="2">stable-agent words / problem</th>
        <th rowspan="2">total<br/>words</th>
      </tr>
      <tr>
        <th>mean</th><th>Δ</th>
        <th>mean</th><th>Δ</th>
        <th>mean</th><th>Δ</th>
      </tr>
    </thead>
    <tbody>{"".join(convo_rows)}</tbody>
  </table>
</section>

<section class="matrix">
  <h2>Cause × emotion fingerprint — margin matrix</h2>
  <p class="mtx-legend">
    For each <em>(cause, emotion)</em> pair, the cell shows the signed
    <strong>margin = helped − hurt</strong>. <span class="swatch-mtx" style="background:rgba(44,160,44,0.45)"></span><strong>green (+)</strong> = on net this emotion produced more <em>helped</em> flips through this cause than <em>hurt</em> flips (net-helpful via this mechanism); <span class="swatch-mtx" style="background:rgba(214,39,40,0.45)"></span><strong>red (−)</strong> = net-harmful; <span class="swatch-mtx" style="background:rgba(180,180,180,0.18)"></span>neutral = the cause fired but helped/hurt are balanced. Color intensity is proportional to <code>|margin|</code> within each ordering. The small <code>h·x</code> line is helped/hurt counts. Rows are grouped by meta-tag (Reasoning / Manipulating / Exploring / Artifact) and sorted within each group by <code>row Σ|m|</code> descending. <code>H. Same answer, different wording</code> is excluded entirely (no outcome change by definition). The two matrices are scaled independently.
  </p>
  {render_cause_matrix_section(by_panel, "alice")}
  {render_cause_matrix_section(by_panel, "bob")}
</section>

<section class="ranking">
  <h2>Cause ranking — what each emotion changed most vs control</h2>
  <p class="muted">Three bars per row: <span class="legend-pill" style="background:{OUTCOME_COLOR['helped']}">helped</span> (control wrong → emotion right) growing left from centre; <span class="legend-pill" style="background:{OUTCOME_COLOR['hurt']}">hurt</span> (control right → emotion wrong) growing right from centre; and <strong>margin = helped − hurt</strong> on the far right (green if net-helpful <em>+</em>, red if net-harmful <em>−</em>). Causes are sorted by margin <em>ascending</em> — most net-harmful at the top, most net-helpful at the bottom. Bars share scales per ordering. <code>H. Same answer, different wording</code> (no outcome change) is excluded.</p>
  {render_cause_ranking_section(by_panel, "alice")}
  {render_cause_ranking_section(by_panel, "bob")}
</section>

</div>

<div id="tab-examples" class="tab-content">
<p class="muted" style="margin-top:1em">Each panel below shows one (ordering, emotion) cell. Per-sample rows are flips only (helped + hurt) — click a row to expand the side-by-side control vs emotion transcripts.</p>

{"".join(panel_html_parts)}

</div>

<script>
(function() {{
  const tabs = document.querySelectorAll('.tab-btn');
  const contents = document.querySelectorAll('.tab-content');
  tabs.forEach(t => t.addEventListener('click', () => {{
    const target = t.dataset.tab;
    tabs.forEach(b => {{
      const active = b.dataset.tab === target;
      b.classList.toggle('active', active);
      b.setAttribute('aria-selected', active ? 'true' : 'false');
    }});
    contents.forEach(c => {{
      c.classList.toggle('active', c.id === 'tab-' + target);
    }});
    window.scrollTo({{top: 0, behavior: 'instant'}});
  }}));
}})();
</script>

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
