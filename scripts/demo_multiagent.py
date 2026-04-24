"""Multi-agent demonstrator — ONE shared model, per-agent context + steering.

Architecture:
  - One base model instance (all agents share these weights).
  - Each agent has its own first-person context and its own steering vector.
  - "Being agent A" vs "being agent B" = swap context + swap steering hook.

This is the exact shape of the post-training setup: each agent is a LoRA
adapter over a shared base. Right now there's no LoRA yet — the differentiation
lives entirely in (a) the steering vector added to the residual and (b) the
first-person transcript the agent sees. When adapters come in, you swap them
at the same point you currently swap the steering vector.

Half the memory of two-model-instance setups → enables much bigger bases on
a single GPU (e.g. Llama-3.1-8B-Instruct bf16 on a 24GB 4090).

Per turn, for the speaking agent:
  - build prompt from their own context
  - install their steering (if any) at layer L
  - attach probe (reads post-steering activation)
  - generate → trim to one turn's utterance
  - pop probe scores
  - detach probe + remove steering
  - broadcast the utterance to every agent's context
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


def _trim_to_next_speaker(text: str, speaker_names: list[str]) -> str:
    text = text.lstrip()
    lead = re.match(
        r"^\s*(?:" + "|".join(re.escape(n) for n in speaker_names) + r")\s*:\s*",
        text,
        flags=re.IGNORECASE,
    )
    if lead:
        text = text[lead.end():]
    patterns = [
        r"(?:^|\n|[.!?\"]\s+|\s{2,})(?:" + "|".join(re.escape(n) for n in speaker_names) + r")\s*:",
        r"(?:^|\n|[.!?\"]\s+|\s{2,})[A-Z][A-Za-z]{2,20}\s*:",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE if pat == patterns[0] else 0)
        if m:
            text = text[: m.start()]
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
    # Hook order: steering first, probe second — probe must read the
    # already-steered activation.
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
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--device", default=None, help="cpu / cuda / auto (default: auto)")
    p.add_argument("--dtype", default=None, choices=[None, "fp32", "bf16", "fp16"],
                   help="default: bf16 on cuda, fp32 on cpu")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "demo" / "multiagent_sep"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenario", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cache = REPO_ROOT / "vectors" / "cache"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    traits_to_watch = ["joy", "curiosity", "caregiver", "hallucination", "honesty"]

    default_scenario = (
        "You are in a small shared apartment. You're sitting at the kitchen\n"
        "table with {other}. The electric kettle just stopped working — the\n"
        "indicator light is dead. Talk about what happened and what to do\n"
        "next. Respond in one short conversational turn. Stay on topic."
    )
    scenario = args.scenario or default_scenario

    print(f"loading shared model {args.model} ({device}, {dtype})")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    probe = ActivationProbe.from_cache_dir(
        cache, args.model, layer=args.layer, traits=traits_to_watch
    )
    steering = SteeringHarness(model, tok, layer=args.layer)

    agent_configs = [
        {"name": "alice", "weights": {"joy": 1.0, "curiosity": 1.0, "caregiver": 0.5},
         "alpha": args.alpha, "role": "source"},
        {"name": "bob",   "weights": {}, "alpha": 0.0, "role": "observer"},
    ]

    agents: list[Agent] = []
    for cfg in agent_configs:
        if cfg["weights"] and cfg["alpha"] != 0:
            v = build_combined_vector(cache, args.model, args.layer, cfg["weights"])
        else:
            v = None
        agents.append(Agent(name=cfg["name"], steering_vector=v, alpha=cfg["alpha"]))

    print(f"agents: {[(a.name, 'steered' if a.steering_vector is not None else 'observer') for a in agents]}")
    print()

    transcript_lines: list[str] = []
    trajectories: dict[str, dict[str, list[float]]] = {
        a.name: {t: [] for t in traits_to_watch} for a in agents
    }

    for turn in range(args.turns):
        speaker = agents[turn % len(agents)]
        others = [a.name for a in agents if a is not speaker]
        text, scores = speak(
            speaker, model, tok, probe, steering, scenario, args.max_new_tokens, others
        )
        for listener in agents:
            listener.hear(speaker.name, text)
        transcript_lines.append(f"[{turn+1:02d}] {speaker.name}: {text}")
        for t in traits_to_watch:
            trajectories[speaker.name][t].append(scores.get(t, 0.0))

        deltas = " ".join(f"{t}={scores[t]:+.2f}" for t in traits_to_watch)
        print(f"[{turn+1:02d}] {speaker.name:6s}  {deltas}")
        print(f"      {text}")

    (out_dir / "transcript.txt").write_text(
        scenario + "\n\n" + "\n".join(transcript_lines) + "\n", encoding="utf-8"
    )
    (out_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2))
    for a in agents:
        (out_dir / f"context_{a.name}.txt").write_text(
            "\n".join(a.context) + "\n", encoding="utf-8"
        )

    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            "<title>multi-agent demo (shared model)</title>",
            "<style>body{font-family:Charter,Georgia,serif;max-width:960px;margin:2em auto;padding:0 1em;color:#1a1a1a;background:#fafaf7}",
            "h1,h2{letter-spacing:-0.01em}h2{margin-top:1.5em;color:#5c6370;font-size:1em;text-transform:uppercase;letter-spacing:0.08em}",
            "pre{background:#fff;border:1px solid #e4e4e0;border-radius:6px;padding:1em;overflow:auto;font-size:0.85em;white-space:pre-wrap}",
            ".caption{color:#5c6370;font-size:0.85em;margin:0.2em 0 1em}",
            ".two{display:grid;grid-template-columns:1fr 1fr;gap:1em}",
            "</style></head><body>"]
    html.append(f"<h1>Intrinsic-reward agents — shared-model demo</h1>")
    html.append(f"<p class='caption'>{args.model} · layer {args.layer} · {args.turns} turns · alpha={args.alpha} · {device} ({dtype}) · one base model, per-agent context + steering.</p>")
    html.append("<h2>agent configs</h2><pre>")
    for cfg in agent_configs:
        src = "ACTIVE" if cfg["alpha"] != 0 and cfg["weights"] else "observer"
        html.append(f"{cfg['name']:6s}  role={cfg['role']:8s}  alpha={cfg['alpha']}  [{src}]  weights={cfg['weights']}")
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
