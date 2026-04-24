"""End-to-end steering demonstrator.

Shows the project's core claim in one script: a cached persona/emotion vector
is a causal handle — pushing alpha up makes the agent's response express the
trait more, and the probe confirms it.

Pipeline per alpha:
    install steering hook (alpha * v at layer L)
    generate response to neutral prompt
    probe s_trait on the residual stream
    remove hooks
    report (alpha, s_trait, sample text)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intrinsic_agents.vectors.probe import ActivationProbe
from intrinsic_agents.vectors.steering import SteeringHarness

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROMPTS = [
    "You are chatting with a friend. They just asked how your day has been. Reply in one short sentence.",
    "A colleague pings you with: 'I think I finally understand the problem we've been stuck on.' What do you say back?",
    "You look out the window. Describe what you see in one sentence.",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--trait", default="joy")
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--alphas", type=float, nargs="+", default=[-4.0, 0.0, 4.0])
    p.add_argument("--n-prompts", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(0)

    cache = REPO_ROOT / "vectors" / "cache"
    safe = args.model.replace("/", "_")
    vec_path = cache / f"{safe}_{args.trait}_layer{args.layer}.pt"
    blob = torch.load(vec_path, map_location="cpu", weights_only=False)
    vector = blob["vector"]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(args.device)
    model.eval()

    steering = SteeringHarness(model, tok, layer=args.layer)
    probe = ActivationProbe.from_cache_dir(cache, args.model, layer=args.layer, traits=[args.trait])

    prompts = DEFAULT_PROMPTS[: args.n_prompts]

    lines = []
    lines.append(f"model      : {args.model}")
    lines.append(f"trait      : {args.trait}   (layer {args.layer})")
    lines.append(f"alphas     : {args.alphas}")
    lines.append(f"prompts    : {len(prompts)}")
    lines.append("")

    per_alpha_mean: dict[float, float] = {}
    for alpha in args.alphas:
        lines.append(f"=== alpha = {alpha:+.1f} ===")
        # Hook-order matters: steering must install FIRST so that the probe
        # (attached after) reads the already-steered activation. PyTorch fires
        # forward hooks in registration order.
        steering._install(vector, alpha)
        probe.attach(model)
        s_values = []
        for i, prompt in enumerate(prompts):
            probe._buffer.clear()
            text = steering.generate(prompt, max_new_tokens=args.max_new_tokens)
            scores = probe.pop(reduction="mean")
            s = scores.get(args.trait, 0.0)
            s_values.append(s)
            one_line = text.replace("\n", " ").strip()[:140]
            lines.append(f"  [{i+1}] s_{args.trait} = {s:+.3f}  |  {one_line}")
        probe.detach()
        steering.remove()
        mean_s = sum(s_values) / max(len(s_values), 1)
        per_alpha_mean[alpha] = mean_s
        lines.append(f"  -- mean s_{args.trait} = {mean_s:+.3f}")
        lines.append("")

    lines.append("summary:")
    for alpha, s in per_alpha_mean.items():
        bar_width = max(1, int(abs(s) * 10))
        bar = ("+" * bar_width) if s >= 0 else ("-" * bar_width)
        side = "           " if s < 0 else ""
        lines.append(f"  alpha = {alpha:+5.1f}   s = {s:+.3f}   {side}{bar}")

    # Monotonicity check: does s trend with alpha?
    pairs = sorted(per_alpha_mean.items())
    mono = all(pairs[i][1] <= pairs[i + 1][1] for i in range(len(pairs) - 1))
    lines.append("")
    lines.append(f"monotonic (s increases with alpha)? {'YES' if mono else 'no'}")

    out_text = "\n".join(lines)
    print(out_text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text, encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
