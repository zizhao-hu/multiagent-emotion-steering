"""Multi-agent demonstrator — separate agents, separate contexts.

Each agent has:
  - its own model instance (own weights in memory, ready to be swapped for
    an independently-trained LoRA adapter in the RL version)
  - its own ActivationProbe and SteeringHarness (bound to its model)
  - its own first-person context: the agent sees "You: ..." for its own
    utterances and "<name>: ..." for the other agents'

Per turn:
  - The speaker generates from its OWN context (first-person view) with its
    OWN steering installed.
  - The utterance is broadcast to every agent's context (including the
    speaker's). Other agents see it tagged with the speaker's name; the
    speaker sees it tagged as "You".

This makes the separation real: no shared network, no shared memory.
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
    vs = [w * load_vector(cache_dir, model, trait, layer) for trait, w in weights.items()]
    combined = sum(vs)
    return combined / (combined.norm() + 1e-8)


class Agent:
    """One agent. Owns its own model weights, probe, steering, and context."""

    def __init__(
        self,
        name: str,
        model_name: str,
        layer: int,
        steering_weights: dict[str, float],
        alpha: float,
        traits_to_watch: list[str],
        device: str,
        dtype: torch.dtype,
        cache_dir: Path,
    ):
        self.name = name
        self.alpha = alpha
        self.layer = layer
        self.context: list[str] = []  # first-person transcript lines

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        self.model.eval()

        # This agent's steering target — the "envisioned self."
        self.steering_vector = build_combined_vector(cache_dir, model_name, layer, steering_weights)

        self.probe = ActivationProbe.from_cache_dir(
            cache_dir, model_name, layer=layer, traits=traits_to_watch
        )
        self.steering = SteeringHarness(self.model, self.tokenizer, layer=layer)

    def hear(self, speaker: str, utterance: str) -> None:
        """Record an utterance into this agent's first-person context."""
        tag = "You" if speaker == self.name else speaker
        self.context.append(f"{tag}: {utterance}")

    def prompt(self, scenario: str) -> str:
        """Build the agent's first-person prompt for its next turn."""
        history = "\n".join(self.context[-32:])  # bounded context window
        return f"{scenario.rstrip()}\n\n{history}\nYou:"

    @torch.no_grad()
    def speak(self, scenario: str, max_new_tokens: int) -> tuple[str, dict[str, float]]:
        # Hook order: steering first, probe second — probe must read the
        # already-steered activation.
        self.steering._install(self.steering_vector, self.alpha)
        self.probe.attach(self.model)
        try:
            text = self.steering.generate(self.prompt(scenario), max_new_tokens=max_new_tokens)
            text = text.strip().split("\n")[0]
            scores = self.probe.pop(reduction="mean")
        finally:
            self.probe.detach()
            self.steering.remove()
        return text, scores


def svg_sparkline(trajs: dict[str, list[float]], width: int = 640, height: int = 160) -> str:
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
    svg.append(f'<line x1="{pad_x}" y1="{pad_y}" x2="{pad_x}" y2="{pad_y + plot_h}" stroke="#888"/>')
    svg.append(f'<line x1="{pad_x}" y1="{pad_y + plot_h}" x2="{pad_x + plot_w}" y2="{pad_y + plot_h}" stroke="#888"/>')
    svg.append(f'<text x="8" y="{pad_y + 5}" font-size="10" fill="#666">{hi:.2f}</text>')
    svg.append(f'<text x="8" y="{pad_y + plot_h + 3}" font-size="10" fill="#666">{lo:.2f}</text>')
    if lo < 0 < hi:
        zy = pad_y + plot_h * (hi - 0) / (hi - lo)
        svg.append(f'<line x1="{pad_x}" y1="{zy}" x2="{pad_x + plot_w}" y2="{zy}" stroke="#ccc" stroke-dasharray="3 3"/>')
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
        svg.append(f'<rect x="{pad_x + plot_w - 140}" y="{legend_y}" width="10" height="10" fill="{color}"/>')
        svg.append(f'<text x="{pad_x + plot_w - 125}" y="{legend_y + 9}" font-size="10" fill="#333">{name}</text>')
        legend_y += 14
    svg.append(f'<text x="{pad_x + plot_w/2}" y="{height - 2}" text-anchor="middle" font-size="10" fill="#666">turn</text>')
    svg.append('</svg>')
    return "\n".join(svg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--turns", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "demo" / "multiagent_sep"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cache = REPO_ROOT / "vectors" / "cache"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traits_to_watch = ["joy", "curiosity", "caregiver", "hallucination", "honesty"]

    default_scenario = (
        "You are in a small shared apartment. You're sitting at the kitchen\n"
        "table with your housemate. The electric kettle just stopped working\n"
        "— the indicator light is dead. Talk about what happened and what\n"
        "to do next. One short line of dialogue per turn. Stay on topic."
    )
    scenario = args.scenario or default_scenario

    agent_configs = [
        ("alice", {"joy": 1.0, "curiosity": 1.0, "caregiver": 0.5}),
        ("bob",   {"joy": -1.0, "hallucination": 0.5}),
    ]

    print(f"loading {len(agent_configs)} separate model instances of {args.model}...")
    agents: list[Agent] = []
    for name, weights in agent_configs:
        print(f"  -> {name}")
        agents.append(Agent(
            name=name,
            model_name=args.model,
            layer=args.layer,
            steering_weights=weights,
            alpha=args.alpha,
            traits_to_watch=traits_to_watch,
            device=args.device,
            dtype=torch.float32,
            cache_dir=cache,
        ))
    print(f"all agents loaded. each has its own weights, probe, steering, context.\n")

    transcript_lines: list[str] = []
    trajectories: dict[str, dict[str, list[float]]] = {
        a.name: {t: [] for t in traits_to_watch} for a in agents
    }

    for turn in range(args.turns):
        speaker = agents[turn % len(agents)]
        text, scores = speaker.speak(scenario, args.max_new_tokens)

        # Broadcast to every agent's context (including the speaker).
        for listener in agents:
            listener.hear(speaker.name, text)

        transcript_lines.append(f"[{turn+1:02d}] {speaker.name}: {text}")
        for t in traits_to_watch:
            trajectories[speaker.name][t].append(scores.get(t, 0.0))

        deltas = " ".join(f"{t}={scores[t]:+.2f}" for t in traits_to_watch)
        print(f"[{turn+1:02d}] {speaker.name:6s}  {deltas}")
        print(f"      {text[:180]}")

    (out_dir / "transcript.txt").write_text(
        scenario + "\n\n" + "\n".join(transcript_lines) + "\n", encoding="utf-8"
    )
    (out_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2))

    # Per-agent context dumps so the separation is auditable.
    for a in agents:
        (out_dir / f"context_{a.name}.txt").write_text(
            "\n".join(a.context) + "\n", encoding="utf-8"
        )

    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            "<title>multi-agent demo (separate contexts)</title>",
            "<style>body{font-family:Charter,Georgia,serif;max-width:960px;margin:2em auto;padding:0 1em;color:#1a1a1a;background:#fafaf7}",
            "h1,h2{letter-spacing:-0.01em}h2{margin-top:1.5em;color:#5c6370;font-size:1em;text-transform:uppercase;letter-spacing:0.08em}",
            "pre{background:#fff;border:1px solid #e4e4e0;border-radius:6px;padding:1em;overflow:auto;font-size:0.85em;white-space:pre-wrap}",
            ".caption{color:#5c6370;font-size:0.85em;margin:0.2em 0 1em}",
            ".two{display:grid;grid-template-columns:1fr 1fr;gap:1em}",
            "</style></head><body>"]
    html.append(f"<h1>Intrinsic-reward agents — separate-context demo</h1>")
    html.append(f"<p class='caption'>{args.model} · layer {args.layer} · {args.turns} turns · alpha={args.alpha} · CPU · separate model instances per agent</p>")
    html.append("<h2>agent configs</h2><pre>")
    for a in agents:
        html.append(f"{a.name}: alpha={a.alpha}  steering_vector ∝ {dict(agent_configs)[a.name]}")
    html.append("</pre>")

    for trait in traits_to_watch:
        series = {f"{a.name}:{trait}": trajectories[a.name][trait] for a in agents}
        html.append(f"<h2>s_{trait} over turns</h2>")
        html.append(svg_sparkline(series))

    html.append("<h2>joint transcript (what a spectator sees)</h2><pre>")
    html.append(scenario)
    html.append("")
    html.extend(transcript_lines)
    html.append("</pre>")

    html.append("<h2>first-person contexts (what each agent sees)</h2><div class='two'>")
    for a in agents:
        html.append(f"<div><h3>{a.name}</h3><pre>")
        html.append(scenario)
        html.append("")
        html.extend(a.context)
        html.append("</pre></div>")
    html.append("</div>")

    html.append("</body></html>")
    (out_dir / "report.html").write_text("\n".join(html), encoding="utf-8")
    print(f"\nWrote {out_dir}/transcript.txt, trajectories.json, report.html, context_*.txt")


if __name__ == "__main__":
    main()
