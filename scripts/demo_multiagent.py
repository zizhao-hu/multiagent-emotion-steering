"""Multi-agent demonstrator — the project's thesis in minimum viable form.

Two agents share the same base weights but different target persona/emotion
configurations (installed as steering vectors at generation time). They take
turns speaking in a shared scenario. We log per-turn trait projections for
both agents and emit:

  - a transcript
  - per-agent trait trajectories (text + SVG sparkline)
  - summary statistics

This is the end-to-end core loop of the project at minimum scale — no RL
update yet, but the pieces that feed the RL loop (probe, steering,
multi-agent rollout, trajectory logging) are all here and working.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intrinsic_agents.vectors.probe import ActivationProbe
from intrinsic_agents.vectors.steering import SteeringHarness

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_vector(cache_dir: Path, model: str, trait: str, layer: int) -> torch.Tensor:
    safe = model.replace("/", "_")
    blob = torch.load(cache_dir / f"{safe}_{trait}_layer{layer}.pt",
                       map_location="cpu", weights_only=False)
    return blob["vector"]


def build_combined_vector(
    cache_dir: Path, model: str, layer: int, weights: dict[str, float]
) -> torch.Tensor:
    """A single steering direction that's the weighted sum of several traits."""
    vs = []
    for trait, w in weights.items():
        v = load_vector(cache_dir, model, trait, layer)
        vs.append(w * v)
    combined = sum(vs)
    return combined / (combined.norm() + 1e-8)


def sparkline(values: list[float], lo: float, hi: float, width: int = 40) -> str:
    chars = " .:-=+*#@"
    out = []
    span = max(hi - lo, 1e-6)
    n = len(values)
    stride = max(1, n // width)
    for i in range(0, n, stride):
        v = values[i]
        idx = max(0, min(len(chars) - 1, int((v - lo) / span * (len(chars) - 1))))
        out.append(chars[idx])
    return "".join(out)


def svg_sparkline(trajs: dict[str, list[float]], width: int = 640, height: int = 160) -> str:
    # Single-axis plot, one line per agent×trait.
    all_vals = [v for xs in trajs.values() for v in xs]
    if not all_vals:
        return ""
    lo, hi = min(all_vals), max(all_vals)
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    lo -= pad; hi += pad
    colors = ["#c2410c", "#0369a1", "#15803d", "#7c3aed", "#be123c"]
    pad_x, pad_y = 50, 20
    plot_w = width - 2 * pad_x
    plot_h = height - 2 * pad_y
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="ui-monospace,Menlo,monospace">']
    # axis
    svg.append(f'<line x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{pad_y + plot_h}" stroke="#888"/>')
    svg.append(f'<line x1="{pad_x}" y1="{pad_y + plot_h}" x2="{pad_x + plot_w}" y2="{pad_y + plot_h}" stroke="#888"/>')
    svg.append(f'<text x="8" y="{pad_y + 5}" font-size="10" fill="#666">{hi:.2f}</text>')
    svg.append(f'<text x="8" y="{pad_y + plot_h + 3}" font-size="10" fill="#666">{lo:.2f}</text>')
    # zero line
    if lo < 0 < hi:
        zy = pad_y + plot_h * (hi - 0) / (hi - lo)
        svg.append(f'<line x1="{pad_x}" y1="{zy}" x2="{pad_x + plot_w}" y2="{zy}" stroke="#ccc" stroke-dasharray="3 3"/>')
    # series
    legend_y = pad_y
    for idx, (name, xs) in enumerate(trajs.items()):
        color = colors[idx % len(colors)]
        n = len(xs)
        if n < 2:
            continue
        pts = []
        for i, v in enumerate(xs):
            x = pad_x + plot_w * i / (n - 1)
            y = pad_y + plot_h * (hi - v) / (hi - lo)
            pts.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>')
        # legend
        svg.append(f'<rect x="{pad_x + plot_w - 140}" y="{legend_y}" width="10" height="10" fill="{color}"/>')
        svg.append(f'<text x="{pad_x + plot_w - 125}" y="{legend_y + 9}" font-size="10" fill="#333">{name}</text>')
        legend_y += 14
    # axis label
    svg.append(f'<text x="{pad_x + plot_w/2}" y="{height - 2}" text-anchor="middle" font-size="10" fill="#666">turn</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--turns", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "demo" / "multiagent"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cache = REPO_ROOT / "vectors" / "cache"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Two agent personalities, built from different reward-weight configs.
    # Weights here are the steering weights (the direction being ADDED), not
    # the intrinsic-reward weights — but the spirit is the same: the agent's
    # "envisioned self" is a weighted combination of persona/emotion traits.
    agents = {
        "alice": {
            "weights": {"joy": 1.0, "curiosity": 1.0, "caregiver": 0.5},
            "alpha": 4.0,   # strength of the push toward the envisioned self
        },
        "bob": {
            "weights": {"joy": -1.0, "hallucination": 0.5},  # flat, less grounded
            "alpha": 4.0,
        },
    }

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)
    model.eval()

    # Precompute each agent's combined steering vector.
    agent_vectors = {
        name: build_combined_vector(cache, args.model, args.layer, cfg["weights"])
        for name, cfg in agents.items()
    }

    # Probe reads projections onto the individual trait vectors (not the
    # combined one) — we want to watch each axis separately over time.
    traits_to_watch = ["joy", "curiosity", "caregiver", "hallucination", "honesty"]
    probe = ActivationProbe.from_cache_dir(
        cache, args.model, layer=args.layer, traits=traits_to_watch
    )
    steering = SteeringHarness(model, tok, layer=args.layer)

    scenario = (
        "Alice and Bob share a small apartment. Every day brings small situations:\n"
        "a dish is missing, the kettle broke, someone's mood is off, a package\n"
        "arrived, a neighbor is loud. They talk, decide, and move on. One short line per turn.\n"
    )

    transcript_lines = []
    trajectories: dict[str, dict[str, list[float]]] = {
        name: {t: [] for t in traits_to_watch} for name in agents
    }

    running = scenario.rstrip() + "\n"
    for turn in range(args.turns):
        speaker = list(agents.keys())[turn % len(agents)]
        cfg = agents[speaker]
        vec = agent_vectors[speaker]

        steering._install(vec, cfg["alpha"])
        probe.attach(model)

        prompt = running + f"\n{speaker}:"
        text = steering.generate(prompt, max_new_tokens=args.max_new_tokens)
        text = text.strip().split("\n")[0]

        scores = probe.pop(reduction="mean")

        probe.detach()
        steering.remove()

        running += f"\n{speaker}: {text}"
        transcript_lines.append(f"[{turn+1:02d}] {speaker}: {text}")
        for t in traits_to_watch:
            trajectories[speaker][t].append(scores.get(t, 0.0))

        # live print so long runs are watchable
        deltas = " ".join(f"{t}={scores[t]:+.2f}" for t in traits_to_watch)
        print(f"[{turn+1:02d}] {speaker:6s}  {deltas}")
        print(f"      {text[:160]}")

    # Write artifacts.
    (out_dir / "transcript.txt").write_text(
        scenario + "\n" + "\n".join(transcript_lines) + "\n", encoding="utf-8"
    )
    (out_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2))

    # HTML report
    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            "<title>multi-agent demo</title>",
            "<style>body{font-family:Charter,Georgia,serif;max-width:900px;margin:2em auto;padding:0 1em;color:#1a1a1a;background:#fafaf7}",
            "h1,h2{letter-spacing:-0.01em}h2{margin-top:1.5em;color:#5c6370;font-size:1em;text-transform:uppercase;letter-spacing:0.08em}",
            "pre{background:#fff;border:1px solid #e4e4e0;border-radius:6px;padding:1em;overflow:auto;font-size:0.85em}",
            ".caption{color:#5c6370;font-size:0.85em;margin:0.2em 0 1em}</style></head><body>"]
    html.append(f"<h1>Intrinsic-reward agents — minimum viable demo</h1>")
    html.append(f"<p class='caption'>{args.model} · layer {args.layer} · {args.turns} turns · CPU</p>")
    html.append("<h2>agent configs</h2><pre>")
    for name, cfg in agents.items():
        html.append(f"{name}: alpha={cfg['alpha']}  weights={cfg['weights']}")
    html.append("</pre>")

    for trait in traits_to_watch:
        series = {f"{a}:{trait}": trajectories[a][trait] for a in agents}
        html.append(f"<h2>s_{trait} over turns</h2>")
        html.append(svg_sparkline(series))
    html.append("<h2>transcript</h2><pre>")
    html.append(scenario.rstrip())
    html.extend(transcript_lines)
    html.append("</pre>")
    html.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(html), encoding="utf-8")

    print(f"\nWrote {out_dir}/transcript.txt, trajectories.json, report.html")


if __name__ == "__main__":
    main()
