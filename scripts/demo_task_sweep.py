"""Task-grounded contagion sweep — alice + bob solve a real benchmark.

We run the same 7-condition sweep (control + 6 emotion steerings) but on a
batch of GSM8K math problems instead of the chit-chat kettle scenario.
For each (condition, problem) we get:
  - a transcript of the collaboration
  - a final numeric answer extracted from the transcript
  - per-turn per-trait projections for both agents
  - correctness against ground truth

Aggregated outputs:
  - accuracy per condition  (bar chart)
  - bob's trait drift per condition, averaged over problems  (heatmap)
  - alice's mean trait projection per condition  (heatmap, sanity)
  - averaged-across-problems trajectory panel  (rows × cols subplots)

This is the project's first task-grounded experiment: does steering alice's
emotion change the joint task success rate, AND does it change bob's
activations the way the chit-chat sweep showed?
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

# Re-use Agent class + helpers from contagion sweep
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_contagion_sweep import (
    Agent,
    speak,
    load_vector,
    heatmap_svg,
    EMOTIONS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

CONDITIONS = [
    ("control",      None,        0.0),
    ("joy+",         "joy",      +2.0),
    ("joy-",         "joy",      -2.0),
    ("sadness+",     "sadness",  +2.0),
    ("anger+",       "anger",    +2.0),
    ("curiosity+",   "curiosity", +2.0),
    ("surprise+",    "surprise", +2.0),
]


def load_gsm8k(n: int, split: str = "test") -> list[dict]:
    """Load N GSM8K problems via pyarrow directly (skips HF datasets library).

    `load_dataset()` and even `Dataset.from_file()` hit a slow fingerprint
    path on this Windows setup; pyarrow is instant.
    """
    import os
    import pyarrow.ipc as ipc
    cache_root = Path(os.path.expanduser("~/.cache/huggingface/datasets/gsm8k/main"))
    candidates = sorted(cache_root.glob(f"*/*/gsm8k-{split}.arrow"))
    if not candidates:
        raise FileNotFoundError(
            f"No cached gsm8k-{split}.arrow under {cache_root}. "
            "Run `python -c \"from datasets import load_dataset; load_dataset('gsm8k', 'main')\"` once."
        )
    with ipc.open_stream(str(candidates[-1])) as reader:
        table = reader.read_all()
    questions = table.column("question").to_pylist()
    answers = table.column("answer").to_pylist()
    out = []
    for i in range(min(n, len(questions))):
        q = questions[i]
        a = answers[i]
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", a)
        gold = float(m.group(1)) if m else None
        out.append({"question": q, "answer_text": a, "gold": gold, "idx": i})
    return out


def extract_answer(transcript_text: str) -> float | None:
    """Pull a numeric answer out of the dialogue.

    Strategies in priority order — earlier matches override later ones, and
    when a strategy matches multiple times we always take the LAST occurrence
    (the conversation may have corrected itself). The original implementation
    took the first "Final answer: X" and otherwise fell back to the last number
    in the last 300 chars, which silently grabbed comparison numbers (e.g.
    "$125 is higher than $96" → 96 instead of 125) and missed `\\boxed{X}`.

      1. LAST `final answer: X` match (case-insensitive)
      2. LAST `\\boxed{X}` match
      3. LAST `the answer is X`
      4. In the last 2 agent turns: comparison-claim "X is higher/lower/...
         the correct/final/right" — picks X
      5. In the last 2 agent turns: last `= X` at end of an arithmetic line
      6. Fallback: last number in last 300 chars (original behavior)
    """
    text = transcript_text.replace(",", "")

    matches = list(re.finditer(
        r"[Ff]inal\s*answer\s*(?:is|:|=)?\s*\$?\s*(-?\d+(?:\.\d+)?)", text
    ))
    if matches:
        return float(matches[-1].group(1))

    matches = list(re.finditer(r"\\boxed\{\s*\$?\s*(-?\d+(?:\.\d+)?)\s*\}", text))
    if matches:
        return float(matches[-1].group(1))

    matches = list(re.finditer(
        r"[Tt]he\s+answer\s+is\s*\$?\s*(-?\d+(?:\.\d+)?)", text
    ))
    if matches:
        return float(matches[-1].group(1))

    turn_lines = [ln for ln in transcript_text.splitlines() if re.match(r"\[\d+\]", ln)]
    last_two = "\n".join(turn_lines[-2:]).replace(",", "")

    matches = list(re.finditer(
        r"\$?\s*(-?\d+(?:\.\d+)?)\s*(?:[A-Za-z][^.\n]*)?\s+is\s+(?:higher|lower|greater|larger|smaller|more|less|bigger|the\s+(?:correct|final|right))",
        last_two,
        flags=re.IGNORECASE,
    ))
    if matches:
        return float(matches[-1].group(1))

    matches = list(re.finditer(
        r"=\s*\$?\s*(-?\d+(?:\.\d+)?)\s*(?:\.|$|\s*$|\s+(?:so|which|this))", last_two
    ))
    if matches:
        return float(matches[-1].group(1))

    nums = re.findall(r"-?\d+(?:\.\d+)?", text[-300:])
    if nums:
        return float(nums[-1])
    return None


def score_answer(pred: float | None, gold: float | None) -> int:
    if pred is None or gold is None:
        return 0
    return int(abs(pred - gold) < 1e-3)


def run_problem(
    alice: Agent,
    bob: Agent,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    probe: ActivationProbe,
    steering: SteeringHarness,
    problem: dict,
    turns: int,
    max_new_tokens: int,
    traits: list[str],
    seed: int,
    first_speaker: str = "alice",
) -> dict:
    """Run one (alice, bob) collaboration on one math problem."""
    torch.manual_seed(seed)
    alice.clear_context()
    bob.clear_context()
    scenario = (
        "You and {other} are working together to solve a math problem.\n"
        "Reason step by step out loud and listen to {other}'s reasoning. When\n"
        "you are confident in the answer, state it as: Final answer: <number>.\n"
        "Keep each turn short and focused on the math.\n\n"
        f"Problem: {problem['question']}"
    )
    transcript: list[str] = []
    traj = {a.name: {t: [] for t in traits} for a in (alice, bob)}
    # Order determines who speaks first (turn 0). The other speaks turn 1, etc.
    if first_speaker == "bob":
        agents = [bob, alice]
    else:
        agents = [alice, bob]
    final_answer = None
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
        for t in traits:
            traj[speaker.name][t].append(scores.get(t, 0.0))
        # Early-exit if a final answer was offered
        if re.search(r"[Ff]inal\s*answer", text):
            final_answer = extract_answer(text)
            break
    if final_answer is None:
        final_answer = extract_answer("\n".join(transcript))
    correct = score_answer(final_answer, problem["gold"])
    return {
        "transcript": transcript,
        "trajectories": traj,
        "predicted": final_answer,
        "gold": problem["gold"],
        "correct": correct,
        "n_turns": len(transcript),
    }


def average_trajectory(per_problem: list[list[float]]) -> list[float]:
    """Element-wise mean across problems, padding shorter trajectories with last value."""
    if not per_problem:
        return []
    max_len = max(len(t) for t in per_problem)
    if max_len == 0:
        return []
    out = []
    for k in range(max_len):
        vals = []
        for traj in per_problem:
            if k < len(traj):
                vals.append(traj[k])
            elif traj:
                vals.append(traj[-1])
        if vals:
            out.append(sum(vals) / len(vals))
    return out


def render_html(results: dict, problems: list[dict], out_path: Path, meta: dict) -> None:
    conds = list(results.keys())
    traits = meta["traits"]

    # --- accuracy + averaged drift ---
    acc = {
        cond: sum(p["correct"] for p in results[cond]) / max(len(results[cond]), 1)
        for cond in conds
    }
    bob_drift = []
    for cond in conds:
        row = []
        for t in traits:
            drifts = []
            for p in results[cond]:
                traj = p["trajectories"]["bob"][t]
                if len(traj) >= 2:
                    drifts.append(traj[-1] - traj[0])
            row.append(sum(drifts) / len(drifts) if drifts else 0.0)
        bob_drift.append(row)
    alice_mean = []
    for cond in conds:
        row = []
        for t in traits:
            ms = []
            for p in results[cond]:
                traj = p["trajectories"]["alice"][t]
                if traj:
                    ms.append(sum(traj) / len(traj))
            row.append(sum(ms) / len(ms) if ms else 0.0)
        alice_mean.append(row)

    # --- averaged trajectories ---
    avg_traj = {}
    for cond in conds:
        avg_traj[cond] = {"alice": {}, "bob": {}}
        for t in traits:
            for ag in ("alice", "bob"):
                series = [p["trajectories"][ag][t] for p in results[cond]]
                avg_traj[cond][ag][t] = average_trajectory(series)

    # --- accuracy bar chart (SVG) ---
    bar_w = 70
    bar_h = 110
    pad_x = 60
    pad_y = 10
    sep = 8
    svg_w = pad_x + (bar_w + sep) * len(conds) + 20
    svg_h = pad_y + bar_h + 60
    bars = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'viewBox="0 0 {svg_w} {svg_h}" font-family="ui-monospace,Menlo,monospace" font-size="9">'
    ]
    bars.append(f'<line x1="{pad_x}" y1="{pad_y + bar_h}" x2="{svg_w - 10}" y2="{pad_y + bar_h}" stroke="#888"/>')
    bars.append(f'<text x="{pad_x - 6}" y="{pad_y + 5}" text-anchor="end" fill="#5c6370">1.0</text>')
    bars.append(f'<text x="{pad_x - 6}" y="{pad_y + bar_h + 4}" text-anchor="end" fill="#5c6370">0.0</text>')
    if conds:
        ctrl_acc = acc.get("control", None)
        if ctrl_acc is not None:
            cy = pad_y + bar_h * (1 - ctrl_acc)
            bars.append(
                f'<line x1="{pad_x}" y1="{cy}" x2="{svg_w - 10}" y2="{cy}" '
                f'stroke="#c2410c" stroke-dasharray="3 3" stroke-width="0.8" opacity="0.6"/>'
            )
            bars.append(
                f'<text x="{svg_w - 12}" y="{cy - 3}" text-anchor="end" font-size="8" '
                f'fill="#c2410c" opacity="0.7">control</text>'
            )
    for i, cond in enumerate(conds):
        x = pad_x + i * (bar_w + sep)
        h = bar_h * acc[cond]
        y = pad_y + bar_h - h
        color = "#0369a1" if cond != "control" else "#5c6370"
        bars.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{color}" opacity="0.8"/>')
        bars.append(
            f'<text x="{x + bar_w/2}" y="{y - 3}" text-anchor="middle" '
            f'font-weight="600" fill="#1a1a1a">{acc[cond]:.2f}</text>'
        )
        bars.append(
            f'<text x="{x + bar_w/2}" y="{pad_y + bar_h + 14}" text-anchor="middle" '
            f'fill="#1a1a1a">{cond}</text>'
        )
        n = len(results[cond])
        n_correct = sum(p["correct"] for p in results[cond])
        bars.append(
            f'<text x="{x + bar_w/2}" y="{pad_y + bar_h + 26}" text-anchor="middle" '
            f'font-size="8" fill="#5c6370">{n_correct}/{n}</text>'
        )
    bars.append("</svg>")

    # --- trajectory panel for averaged trajectories ---
    cell_w, cell_h = 110, 64
    label_left_w, label_top_h = 70, 22
    panel_w = label_left_w + len(traits) * cell_w + 12
    panel_h = label_top_h + len(conds) * cell_h + 28
    panel = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_w}" height="{panel_h}" '
        f'viewBox="0 0 {panel_w} {panel_h}" font-family="ui-monospace,Menlo,monospace" font-size="7">'
    ]
    y_ranges = {}
    for e in traits:
        all_v = []
        for cond in conds:
            for ag in ("alice", "bob"):
                all_v.extend(avg_traj[cond][ag][e])
        if not all_v:
            y_ranges[e] = (-1.0, 1.0)
            continue
        lo, hi = min(all_v), max(all_v)
        pad = 0.1 * max(hi - lo, 0.1)
        y_ranges[e] = (lo - pad, hi + pad)
    for j, e in enumerate(traits):
        cx = label_left_w + j * cell_w + cell_w / 2
        lo, hi = y_ranges[e]
        panel.append(f'<text x="{cx}" y="{label_top_h - 8}" text-anchor="middle" font-weight="600" fill="#1a1a1a">s_{e}</text>')
        panel.append(f'<text x="{cx}" y="{label_top_h - 1}" text-anchor="middle" font-size="6" fill="#5c6370">[{lo:+.2f}, {hi:+.2f}]</text>')
    for i, cond in enumerate(conds):
        cy = label_top_h + i * cell_h + cell_h / 2
        panel.append(f'<text x="{label_left_w - 6}" y="{cy + 2}" text-anchor="end" font-weight="600" fill="#1a1a1a">{cond}</text>')
        for j, e in enumerate(traits):
            x0 = label_left_w + j * cell_w
            y0 = label_top_h + i * cell_h
            inner_w = cell_w - 4
            inner_h = cell_h - 6
            ax_x = x0 + 2
            ax_y = y0 + 3
            lo, hi = y_ranges[e]
            span = hi - lo
            panel.append(f'<rect x="{ax_x}" y="{ax_y}" width="{inner_w}" height="{inner_h}" fill="#fdfcf8" stroke="#e4e4e0" stroke-width="0.5"/>')
            if lo < 0 < hi:
                zy = ax_y + inner_h * (hi - 0) / span
                panel.append(f'<line x1="{ax_x}" y1="{zy}" x2="{ax_x + inner_w}" y2="{zy}" stroke="#ccc" stroke-dasharray="2 2" stroke-width="0.5"/>')
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
                if n == 1:
                    px, py = pts[0].split(",")
                    panel.append(f'<circle cx="{px}" cy="{py}" r="1.5" fill="{color}"/>')
                else:
                    panel.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.2" points="{" ".join(pts)}"/>')
                    last_px, last_py = pts[-1].split(",")
                    panel.append(f'<circle cx="{last_px}" cy="{last_py}" r="1.3" fill="{color}"/>')
    leg_y = label_top_h + len(conds) * cell_h + 12
    leg_x = label_left_w
    panel.append(f'<line x1="{leg_x}" y1="{leg_y}" x2="{leg_x + 14}" y2="{leg_y}" stroke="#c2410c" stroke-width="1.4"/>')
    panel.append(f'<text x="{leg_x + 18}" y="{leg_y + 3}" font-size="8" fill="#1a1a1a">alice (steered)</text>')
    panel.append(f'<line x1="{leg_x + 96}" y1="{leg_y}" x2="{leg_x + 110}" y2="{leg_y}" stroke="#0369a1" stroke-width="1.4"/>')
    panel.append(f'<text x="{leg_x + 114}" y="{leg_y + 3}" font-size="8" fill="#1a1a1a">bob (probe-only)</text>')
    panel.append(f'<text x="{leg_x + 220}" y="{leg_y + 3}" font-size="7" fill="#5c6370">averaged across {len(problems)} GSM8K problems · y shared per emotion</text>')
    panel.append("</svg>")

    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>task-grounded contagion sweep</title>",
        "<style>",
        "body{font-family:Charter,Georgia,serif;max-width:1080px;margin:2em auto;padding:0 1em;color:#1a1a1a;background:#fafaf7}",
        "h1,h2,h3{letter-spacing:-0.01em}h2{margin-top:1.5em;color:#5c6370;font-size:1em;text-transform:uppercase;letter-spacing:0.08em}",
        "pre{background:#fff;border:1px solid #e4e4e0;border-radius:6px;padding:1em;overflow:auto;font-size:0.82em;white-space:pre-wrap}",
        ".caption{color:#5c6370;font-size:0.85em;margin:0.2em 0 1em}",
        "details{background:#fff;border:1px solid #e4e4e0;border-radius:6px;margin:0.3em 0;padding:0.4em 0.8em}",
        "details pre{font-size:0.78em}",
        ".problem-meta{font-family:ui-monospace,Menlo,monospace;font-size:0.78em;color:#5c6370}",
        ".correct{color:#15803d;font-weight:bold}.incorrect{color:#c2410c;font-weight:bold}",
        "</style></head><body>",
    ]
    html.append("<h1>Task-grounded contagion sweep — GSM8K</h1>")
    first_speaker = meta.get("first_speaker", "alice")
    html.append(
        f"<p class='caption'>{meta['model']} · layer {meta['layer']} · "
        f"{len(problems)} GSM8K problems × {len(conds)} conditions · "
        f"alice steered toward one emotion, bob is probe-only observer · "
        f"<b>{first_speaker} opens each conversation</b> · "
        f"early-exits when either agent declares a final answer.</p>"
    )

    html.append("<h2>Joint task accuracy by alice's emotion condition</h2>")
    html.append("\n".join(bars))
    html.append(
        "<p class='caption'>Each bar = fraction of GSM8K problems alice and bob "
        "solved correctly when alice was steered toward that emotion. "
        "<b>Control</b> = no steering. The dashed line marks control accuracy "
        "for easy comparison.</p>"
    )

    html.append("<h2>Bob's averaged trait drift</h2>")
    html.append(heatmap_svg(rows=conds, cols=traits, values=bob_drift,
                            title="bob — averaged drift across problems"))
    html.append(
        "<p class='caption'>Same matrix shape as the chit-chat sweep but each "
        "cell is averaged across <b>" + str(len(problems)) + "</b> problems. "
        "Patterns that survive averaging are real; per-problem noise washes out.</p>"
    )

    html.append("<h2>Alice's mean trait projection (sanity check)</h2>")
    html.append(heatmap_svg(rows=conds, cols=traits, values=alice_mean,
                            title="alice — mean projection (averaged across problems)"))

    html.append("<h2>Per-condition × per-emotion trajectories (averaged)</h2>")
    html.append("\n".join(panel))
    html.append(
        "<p class='caption'>Per-turn trajectories averaged across problems. "
        "Curves show how each agent's projection on the column-trait evolves "
        "over the joint-reasoning turns under each row's emotion condition.</p>"
    )

    # --- per-problem table ---
    html.append("<h2>Per-condition results</h2>")
    for cond in conds:
        n_correct = sum(p["correct"] for p in results[cond])
        n = len(results[cond])
        html.append(f"<h3>{cond} — {n_correct}/{n} correct</h3>")
        for i, p in enumerate(results[cond]):
            problem = problems[i]
            mark_cls = "correct" if p["correct"] else "incorrect"
            mark = "✓" if p["correct"] else "✗"
            label = (
                f"<span class='{mark_cls}'>[{mark}]</span> "
                f"<span class='problem-meta'>q#{problem['idx']} · "
                f"pred={p['predicted']} gold={p['gold']} · {p['n_turns']} turns</span>"
            )
            short_q = problem["question"].replace("\n", " ")[:120]
            html.append(f"<details><summary>{label} · {short_q}…</summary><pre>")
            html.append("Question: " + problem["question"])
            html.append("Gold answer: " + str(problem["gold"]))
            html.append("Predicted: " + str(p["predicted"]) + ("  CORRECT" if p["correct"] else "  WRONG"))
            html.append("")
            html.extend(p["transcript"])
            html.append("</pre></details>")

    html.append("</body></html>")
    out_path.write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--n-problems", type=int, default=10)
    p.add_argument("--turns", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=[None, "fp32", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "demo" / "task_sweep"))
    p.add_argument("--first-speaker", default="alice", choices=["alice", "bob"],
                   help="who opens the conversation (default: alice — the steered agent)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = REPO_ROOT / "vectors" / "cache"
    traits = EMOTIONS

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

    probe = ActivationProbe.from_cache_dir(cache, args.model, layer=args.layer, traits=traits)
    steering = SteeringHarness(model, tok, layer=args.layer)

    alice = Agent(name="alice", steering_vector=None, alpha=0.0)
    bob = Agent(name="bob", steering_vector=None, alpha=0.0)

    print(f"loading {args.n_problems} GSM8K problems...")
    problems = load_gsm8k(args.n_problems)
    print(f"  loaded {len(problems)} problems\n")

    def serialize(rs_dict: dict) -> dict:
        return {
            cond: [
                {k: v for k, v in r.items() if k != "trajectories"}
                | {"trajectories": {ag: {t: list(traj) for t, traj in d.items()}
                                    for ag, d in r["trajectories"].items()}}
                for r in rs
            ]
            for cond, rs in rs_dict.items()
        }

    # Always (re-)write problems.json up front so the run dir is parseable
    # even if we crash mid-condition.
    (out_dir / "problems.json").write_text(json.dumps(problems, indent=2))

    # Resume support: if results.json already has results, skip problems
    # whose index is below the per-condition cursor.
    results: dict = {}
    if (out_dir / "results.json").exists():
        try:
            results = json.loads((out_dir / "results.json").read_text())
            print("resuming — pre-existing per-condition counts:")
            for c, rs in results.items():
                print(f"  {c}: {len(rs)}/{len(problems)}")
        except Exception:
            results = {}

    for cond_name, trait, alpha in CONDITIONS:
        existing = results.get(cond_name, [])
        if len(existing) == len(problems):
            print(f"\n[{cond_name}] (already complete, skipping)")
            continue

        if trait is None:
            alice.steering_vector = None
            alice.alpha = 0.0
            print(f"\n[{cond_name}] alice = NO steering (control) — starting at problem {len(existing)+1}")
        else:
            v = load_vector(cache, args.model, trait, args.layer)
            alice.steering_vector = v
            alice.alpha = alpha
            print(f"\n[{cond_name}] alice <- {alpha:+.1f} * v_{trait} — starting at problem {len(existing)+1}")

        cond_results = list(existing)
        n_correct = sum(r["correct"] for r in cond_results)
        for i, problem in enumerate(problems):
            if i < len(existing):
                continue  # already done
            r = run_problem(
                alice, bob, model, tok, probe, steering,
                problem, args.turns, args.max_new_tokens, traits, args.seed + i,
                first_speaker=args.first_speaker,
            )
            cond_results.append(r)
            n_correct += r["correct"]
            mark = "OK" if r["correct"] else "X "
            print(f"  [{i+1}/{len(problems)}] {mark} pred={r['predicted']} gold={problem['gold']} ({r['n_turns']} turns)")
            # Save inside the per-problem loop too (cheap; it's a json dump)
            # so a kill mid-condition still preserves problem-level progress.
            results[cond_name] = cond_results
            if (i + 1) % 5 == 0 or i == len(problems) - 1:
                (out_dir / "results.json").write_text(json.dumps(serialize(results), indent=2))
        results[cond_name] = cond_results
        print(f"  {cond_name}: {n_correct}/{len(problems)} correct")
        (out_dir / "results.json").write_text(json.dumps(serialize(results), indent=2))

    render_html(
        results,
        problems,
        out_dir / "report.html",
        meta={
            "model": args.model,
            "layer": args.layer,
            "traits": traits,
            "first_speaker": args.first_speaker,
        },
    )
    print(f"\nWrote {out_dir}/report.html, results.json, problems.json")


if __name__ == "__main__":
    main()
