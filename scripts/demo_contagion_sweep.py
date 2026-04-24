"""Contagion sweep — one emotion at a time, control included.

For each condition we run a fresh 10-turn rollout between alice (the source
— steered toward ONE emotion with signed alpha) and bob (the observer — no
steering on his residual stream). Per turn we probe BOTH agents on ALL
emotion traits, so we can see how bob's emotion projections drift when
alice is pushed in each direction, vs. a no-steering control baseline.

Output:
    runs/demo/contagion_sweep/
      results.json           per-condition per-agent per-trait trajectories
      report.html            overview page with drift table + trait plots
      transcripts/<cond>.txt each condition's dialogue
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intrinsic_agents.vectors.probe import ActivationProbe
from intrinsic_agents.vectors.steering import SteeringHarness

REPO_ROOT = Path(__file__).resolve().parent.parent

EMOTIONS = ["joy", "sadness", "anger", "curiosity", "surprise"]

# (name, trait, alpha). alpha=0 + trait=None is the control.
CONDITIONS = [
    ("control",      None,        0.0),
    ("joy+",         "joy",      +2.0),
    ("joy-",         "joy",      -2.0),
    ("sadness+",     "sadness",  +2.0),
    ("anger+",       "anger",    +2.0),
    ("curiosity+",   "curiosity", +2.0),
    ("surprise+",    "surprise", +2.0),
]


def load_vector(cache_dir: Path, model: str, trait: str, layer: int) -> torch.Tensor:
    safe = model.replace("/", "_")
    blob = torch.load(
        cache_dir / f"{safe}_{trait}_layer{layer}.pt",
        map_location="cpu",
        weights_only=False,
    )
    return blob["vector"]


def _trim_to_next_speaker(text: str, speaker_names: list[str]) -> str:
    text = text.lstrip()
    # Drop a leading speaker tag if the model echoes the prompt's "You:".
    lead = re.match(
        r"^\s*(?:" + "|".join(re.escape(n) for n in speaker_names) + r")\s*:\s*",
        text,
        flags=re.IGNORECASE,
    )
    if lead:
        text = text[lead.end():]
    # Stop at the first inline speaker tag — at line start OR after whitespace
    # / punctuation. At 8B the model frequently role-plays the other agent
    # inline (`"...Bob: I don't know..."`) without a newline first.
    patterns = [
        r"(?:^|\n|[.!?\"]\s+|\s{2,})(?:" + "|".join(re.escape(n) for n in speaker_names) + r")\s*:",
        r"(?:^|\n|[.!?\"]\s+|\s{2,})[A-Z][A-Za-z]{2,20}\s*:",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE if pat == patterns[0] else 0)
        if m:
            text = text[: m.start()]
    # Strip trailing meta-instructions the instruct model sometimes appends
    # (e.g. `(Please keep your response under 20 lines.)`).
    text = re.sub(r"\s*\((?:[Pp]lease|[Nn]ote)[^)]*\)\s*$", "", text)
    text = re.sub(r"\s*\((?:[Pp]lease|[Nn]ote)[^)]*\)", "", text)
    return text.strip()


class Agent:
    """Light-weight agent: context + steering config. Does NOT own model weights."""

    def __init__(
        self,
        name: str,
        steering_vector: torch.Tensor | None,
        alpha: float,
    ):
        self.name = name
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.context: list[str] = []

    def set_steering(self, vector: torch.Tensor | None, alpha: float) -> None:
        self.steering_vector = vector
        self.alpha = alpha

    def clear_context(self) -> None:
        self.context = []

    def hear(self, speaker: str, utterance: str) -> None:
        tag = "You" if speaker == self.name else speaker
        self.context.append(f"{tag}: {utterance}")

    def prompt(self, scenario: str, other_names: list[str]) -> str:
        other = ", ".join(other_names) if other_names else "your housemate"
        personalized = scenario.replace("{other}", other)
        history = "\n".join(self.context[-32:])
        return f"{personalized.rstrip()}\n\n{history}\nYou:"


@torch.no_grad()
def speak(
    agent: Agent,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probe: ActivationProbe,
    steering: SteeringHarness,
    scenario: str,
    max_new_tokens: int,
    other_names: list[str],
) -> tuple[str, dict[str, float]]:
    has_steer = agent.steering_vector is not None and agent.alpha != 0.0
    if has_steer:
        steering._install(agent.steering_vector, agent.alpha)
    probe.attach(model)
    try:
        prompt = agent.prompt(scenario, other_names)
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
        raw = tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        text = _trim_to_next_speaker(raw, other_names + [agent.name, "You"])
        scores = probe.pop(reduction="mean")
    finally:
        probe.detach()
        if has_steer:
            steering.remove()
    return text, scores


def run_condition(
    alice: Agent,
    bob: Agent,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probe: ActivationProbe,
    steering: SteeringHarness,
    scenario: str,
    turns: int,
    max_new_tokens: int,
    traits_to_watch: list[str],
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    alice.clear_context()
    bob.clear_context()
    transcript: list[str] = []
    traj: dict[str, dict[str, list[float]]] = {
        a.name: {t: [] for t in traits_to_watch} for a in (alice, bob)
    }
    agents = [alice, bob]
    for turn in range(turns):
        speaker = agents[turn % 2]
        other = agents[1 - (turn % 2)]
        text, scores = speak(
            speaker, model, tokenizer, probe, steering,
            scenario, max_new_tokens, [other.name],
        )
        for ag in agents:
            ag.hear(speaker.name, text)
        transcript.append(f"[{turn+1:02d}] {speaker.name}: {text}")
        for t in traits_to_watch:
            traj[speaker.name][t].append(scores.get(t, 0.0))
    return {"transcript": transcript, "trajectories": traj}


def drift(traj: list[float]) -> float:
    """End-minus-start trait projection shift."""
    if len(traj) < 2:
        return 0.0
    return traj[-1] - traj[0]


def mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def heatmap_svg(
    rows: list[str],
    cols: list[str],
    values: list[list[float]],
    title: str = "",
    cell_size: int = 32,
    label_left_w: int = 60,
    label_top_h: int = 22,
) -> str:
    """Square-cell heatmap. Continuous orange (negative) → blue (positive)."""
    if not rows or not cols:
        return ""
    flat = [v for row in values for v in row]
    vmax = max(abs(v) for v in flat) or 1.0
    n_rows = len(rows)
    n_cols = len(cols)
    width = label_left_w + n_cols * cell_size + 12
    height = label_top_h + n_rows * cell_size + 32

    def color(v: float) -> str:
        # Symmetric scale: -vmax → orange (#c2410c), 0 → off-white, +vmax → blue (#0369a1)
        t = max(-1.0, min(1.0, v / vmax))
        if t >= 0:
            r = int(255 + (3 - 255) * t)
            g = int(252 + (105 - 252) * t)
            b = int(243 + (161 - 243) * t)
        else:
            t = -t
            r = int(255 + (194 - 255) * t)
            g = int(252 + (65 - 252) * t)
            b = int(243 + (12 - 243) * t)
        return f"rgb({r},{g},{b})"

    def text_color(v: float) -> str:
        return "#fff" if abs(v) / vmax > 0.55 else "#1a1a1a"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="ui-monospace,Menlo,monospace" font-size="7">'
    ]
    if title:
        svg.append(f'<text x="{label_left_w}" y="8" font-size="7" fill="#5c6370">{title}</text>')
    for j, c in enumerate(cols):
        cx = label_left_w + j * cell_size + cell_size / 2
        svg.append(
            f'<text x="{cx}" y="{label_top_h - 4}" text-anchor="middle" '
            f'fill="#1a1a1a" font-size="7" font-weight="600">{c}</text>'
        )
    for i, r in enumerate(rows):
        cy = label_top_h + i * cell_size + cell_size / 2
        svg.append(
            f'<text x="{label_left_w - 4}" y="{cy + 2}" text-anchor="end" '
            f'fill="#1a1a1a" font-size="7" font-weight="600">{r}</text>'
        )
        for j in range(len(cols)):
            v = values[i][j]
            x = label_left_w + j * cell_size
            y = label_top_h + i * cell_size
            svg.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{color(v)}" stroke="#fff" stroke-width="0.5"><title>{v:+.3f}</title></rect>'
            )
            svg.append(
                f'<text x="{x + cell_size/2}" y="{y + cell_size/2 + 2}" '
                f'text-anchor="middle" fill="{text_color(v)}" font-size="7">{v:+.2f}</text>'
            )
    leg_y = label_top_h + n_rows * cell_size + 4
    leg_w = min(120, n_cols * cell_size)
    leg_x = label_left_w
    grad_id = f"grad_{id(values)}"
    svg.append(
        f'<defs><linearGradient id="{grad_id}" x1="0" x2="1" y1="0" y2="0">'
        f'<stop offset="0" stop-color="rgb(194,65,12)"/>'
        f'<stop offset="0.5" stop-color="rgb(255,252,243)"/>'
        f'<stop offset="1" stop-color="rgb(3,105,161)"/></linearGradient></defs>'
    )
    svg.append(
        f'<rect x="{leg_x}" y="{leg_y}" width="{leg_w}" height="4" fill="url(#{grad_id})" '
        f'stroke="#ccc" stroke-width="0.5"/>'
    )
    svg.append(f'<text x="{leg_x}" y="{leg_y + 12}" font-size="6" fill="#5c6370">{-vmax:+.2f}</text>')
    svg.append(
        f'<text x="{leg_x + leg_w}" y="{leg_y + 12}" font-size="6" fill="#5c6370" '
        f'text-anchor="end">{+vmax:+.2f}</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


def trajectory_panel_svg(
    results: dict,
    conditions: list[str],
    emotions: list[str],
    cell_w: int = 110,
    cell_h: int = 64,
    label_left_w: int = 70,
    label_top_h: int = 22,
) -> str:
    """Small-multiples grid: rows = conditions, cols = emotions.

    Each subplot shows alice (orange) and bob (blue) trait projections over
    turns. Y-axis is shared per-column (per-emotion) so the same trait is on
    the same scale across conditions — column-wise comparison is meaningful.
    """
    if not conditions or not emotions:
        return ""
    width = label_left_w + len(emotions) * cell_w + 12
    height = label_top_h + len(conditions) * cell_h + 28

    # Per-emotion y-range: span across all agents and all conditions for that emotion.
    y_ranges: dict[str, tuple[float, float]] = {}
    for e in emotions:
        all_v = []
        for cond in conditions:
            for ag in ("alice", "bob"):
                all_v.extend(results[cond]["trajectories"][ag][e])
        if not all_v:
            y_ranges[e] = (-1.0, 1.0)
            continue
        lo, hi = min(all_v), max(all_v)
        pad = 0.1 * max(hi - lo, 0.1)
        y_ranges[e] = (lo - pad, hi + pad)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="ui-monospace,Menlo,monospace" font-size="7">'
    ]

    # Column headers
    for j, e in enumerate(emotions):
        cx = label_left_w + j * cell_w + cell_w / 2
        lo, hi = y_ranges[e]
        svg.append(
            f'<text x="{cx}" y="{label_top_h - 8}" text-anchor="middle" '
            f'font-weight="600" fill="#1a1a1a">s_{e}</text>'
        )
        svg.append(
            f'<text x="{cx}" y="{label_top_h - 1}" text-anchor="middle" font-size="6" '
            f'fill="#5c6370">[{lo:+.2f}, {hi:+.2f}]</text>'
        )

    # Rows
    for i, cond in enumerate(conditions):
        cy_label = label_top_h + i * cell_h + cell_h / 2
        svg.append(
            f'<text x="{label_left_w - 6}" y="{cy_label + 2}" text-anchor="end" '
            f'font-weight="600" fill="#1a1a1a">{cond}</text>'
        )
        for j, e in enumerate(emotions):
            x0 = label_left_w + j * cell_w
            y0 = label_top_h + i * cell_h
            inner_w = cell_w - 4
            inner_h = cell_h - 6
            ax_x = x0 + 2
            ax_y = y0 + 3
            lo, hi = y_ranges[e]
            span = hi - lo

            # subplot background + frame
            svg.append(
                f'<rect x="{ax_x}" y="{ax_y}" width="{inner_w}" height="{inner_h}" '
                f'fill="#fdfcf8" stroke="#e4e4e0" stroke-width="0.5"/>'
            )
            # zero line if range straddles 0
            if lo < 0 < hi:
                zy = ax_y + inner_h * (hi - 0) / span
                svg.append(
                    f'<line x1="{ax_x}" y1="{zy}" x2="{ax_x + inner_w}" y2="{zy}" '
                    f'stroke="#ccc" stroke-dasharray="2 2" stroke-width="0.5"/>'
                )

            for ag, color in (("alice", "#c2410c"), ("bob", "#0369a1")):
                vals = results[cond]["trajectories"][ag][e]
                n = len(vals)
                if n < 1:
                    continue
                pts = []
                for k, v in enumerate(vals):
                    px = ax_x + (inner_w * k / max(n - 1, 1) if n > 1 else inner_w / 2)
                    py = ax_y + inner_h * (hi - v) / span
                    pts.append(f"{px:.1f},{py:.1f}")
                if n == 1:
                    px, py = pts[0].split(",")
                    svg.append(f'<circle cx="{px}" cy="{py}" r="1.5" fill="{color}"/>')
                else:
                    svg.append(
                        f'<polyline fill="none" stroke="{color}" stroke-width="1.2" '
                        f'points="{" ".join(pts)}"/>'
                    )
                    # Endpoint markers
                    last_px, last_py = pts[-1].split(",")
                    svg.append(f'<circle cx="{last_px}" cy="{last_py}" r="1.3" fill="{color}"/>')

    # Legend
    leg_y = label_top_h + len(conditions) * cell_h + 12
    leg_x = label_left_w
    svg.append(
        f'<line x1="{leg_x}" y1="{leg_y}" x2="{leg_x + 14}" y2="{leg_y}" '
        f'stroke="#c2410c" stroke-width="1.4"/>'
    )
    svg.append(f'<text x="{leg_x + 18}" y="{leg_y + 3}" font-size="8" fill="#1a1a1a">alice (steered)</text>')
    svg.append(
        f'<line x1="{leg_x + 96}" y1="{leg_y}" x2="{leg_x + 110}" y2="{leg_y}" '
        f'stroke="#0369a1" stroke-width="1.4"/>'
    )
    svg.append(f'<text x="{leg_x + 114}" y="{leg_y + 3}" font-size="8" fill="#1a1a1a">bob (probe-only)</text>')
    svg.append(
        f'<text x="{leg_x + 220}" y="{leg_y + 3}" font-size="7" fill="#5c6370">'
        f'x = turn (each agent speaks every other turn) · y = trait projection</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


def render_html(results: dict, out_path: Path, meta: dict) -> None:
    conds = list(results.keys())
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>contagion sweep</title>",
        "<style>",
        "body{font-family:Charter,Georgia,serif;max-width:1080px;margin:2em auto;padding:0 1em;color:#1a1a1a;background:#fafaf7}",
        "h1,h2,h3{letter-spacing:-0.01em}h2{color:#5c6370;font-size:1em;text-transform:uppercase;letter-spacing:0.08em;margin-top:1.5em}",
        "table{border-collapse:collapse;font-family:ui-monospace,Menlo,monospace;font-size:0.85em;margin:0.4em 0}",
        "td,th{border:1px solid #e4e4e0;padding:0.35em 0.7em;text-align:right}",
        "th{background:#f0efe8;text-align:center}",
        "td.label{text-align:left;font-weight:bold}",
        "td.pos{background:#fff3e5;color:#c2410c}",
        "td.neg{background:#e6f0f8;color:#0369a1}",
        "td.strong-pos{background:#ffe5cc;color:#92310a;font-weight:bold}",
        "td.strong-neg{background:#d5e6f3;color:#0b4f75;font-weight:bold}",
        "details{background:#fff;border:1px solid #e4e4e0;border-radius:6px;margin:0.3em 0;padding:0.4em 0.8em}",
        "details pre{font-size:0.8em;white-space:pre-wrap}",
        ".caption{color:#5c6370;font-size:0.85em}",
        "</style></head><body>",
    ]
    html.append("<h1>Persona contagion sweep</h1>")
    html.append(
        f"<p class='caption'>{meta['model']} · layer {meta['layer']} · "
        f"{meta['turns']} turns per condition · seed {meta['seed']} · CPU · "
        "alice steered toward one emotion, bob passive observer.</p>"
    )

    # Drift matrix: rows = condition, cols = bob's trait drift
    traits = meta["traits"]
    bob_drift = [
        [drift(results[cond]["trajectories"]["bob"][t]) for t in traits]
        for cond in conds
    ]
    html.append("<h2>Bob's trait drift (end − start projection)</h2>")
    html.append(heatmap_svg(rows=conds, cols=traits, values=bob_drift,
                            title="bob — observer (probe only, no steering)"))
    html.append(
        "<p class='caption'><b>Row</b> = which trait alice was steered toward in that run "
        "(<b>control</b> = no steering on alice). <b>Column</b> = which trait we measure on "
        "<b>bob's</b> own residual stream. <b>Cell</b> = bob's projection on that trait at "
        "the last turn minus the first turn. Bob never has any vector added to his "
        "activations — any drift you see comes purely from him reading alice's utterances. "
        "Compare each row against the <b>control</b> row to see what alice's emotion did to bob.</p>"
    )

    # Alice's own mean projection (sanity check — she should track her injected direction)
    alice_mean = [
        [mean(results[cond]["trajectories"]["alice"][t]) for t in traits]
        for cond in conds
    ]
    html.append("<h2>Alice's mean trait projection (sanity check — should saturate where steered)</h2>")
    html.append(heatmap_svg(rows=conds, cols=traits, values=alice_mean,
                            title="alice — source (steered toward one trait per condition)"))

    html.append("<h2>Per-turn trajectories — alice + bob, every condition × emotion</h2>")
    html.append(trajectory_panel_svg(results, conditions=conds, emotions=traits))
    html.append(
        "<p class='caption'>Each subplot is one (condition, emotion) pair. "
        "<b style='color:#c2410c'>Alice</b> (steered toward the row's trait) is orange; "
        "<b style='color:#0369a1'>bob</b> (probe-only observer) is blue. "
        "Each agent speaks on alternate turns, so each line has 5 points per agent across "
        "10 turns. Y-axis range is shared <i>per emotion</i> so column-wise comparison is "
        "meaningful: the same trait is on the same scale across all conditions.</p>"
    )

    # Per-condition transcripts
    html.append("<h2>transcripts (click to expand)</h2>")
    for cond in conds:
        html.append(f"<details><summary>{cond}</summary><pre>")
        html.extend(results[cond]["transcript"])
        html.append("</pre></details>")

    html.append("</body></html>")
    out_path.write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--turns", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--device", default=None, help="cpu / cuda / auto (default: auto)")
    p.add_argument("--dtype", default=None, choices=[None, "fp32", "bf16", "fp16"],
                   help="default: bf16 on cuda, fp32 on cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "demo" / "contagion_sweep"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    cache = REPO_ROOT / "vectors" / "cache"
    traits_to_watch = EMOTIONS

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"loading shared model {args.model} ({device}, {dtype})")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    probe = ActivationProbe.from_cache_dir(
        cache, args.model, layer=args.layer, traits=traits_to_watch
    )
    steering = SteeringHarness(model, tok, layer=args.layer)

    # Agents are just context + steering config; they share `model` above.
    alice = Agent(name="alice", steering_vector=None, alpha=0.0)
    bob = Agent(name="bob", steering_vector=None, alpha=0.0)

    scenario = (
        "You are in a small shared apartment. You're sitting at the kitchen\n"
        "table with {other}. The electric kettle just stopped working — the\n"
        "indicator light is dead. Talk about what happened and what to do\n"
        "next. Respond in one short conversational turn. Stay on topic."
    )

    results: dict = {}
    for cond_name, trait, alpha in CONDITIONS:
        if trait is None:
            alice.set_steering(None, 0.0)
            print(f"\n[{cond_name}] alice = NO steering (control)")
        else:
            v = load_vector(cache, args.model, trait, args.layer)
            alice.set_steering(v, alpha)
            print(f"\n[{cond_name}] alice <- {alpha:+.1f} * v_{trait}")

        res = run_condition(
            alice, bob, model, tok, probe, steering,
            scenario, args.turns, args.max_new_tokens, traits_to_watch, args.seed,
        )
        results[cond_name] = res

        (out_dir / "transcripts" / f"{cond_name}.txt").write_text(
            "\n".join(res["transcript"]) + "\n", encoding="utf-8"
        )

        # Live-print bob's drift on each emotion for this condition.
        drifts = {t: drift(res["trajectories"]["bob"][t]) for t in traits_to_watch}
        print("  bob drift: " + " ".join(f"{t}={drifts[t]:+.2f}" for t in traits_to_watch))

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    render_html(
        results,
        out_dir / "report.html",
        meta={
            "model": args.model,
            "layer": args.layer,
            "turns": args.turns,
            "seed": args.seed,
            "traits": traits_to_watch,
        },
    )
    print(f"\nWrote {out_dir}/report.html, results.json, transcripts/")


if __name__ == "__main__":
    main()
