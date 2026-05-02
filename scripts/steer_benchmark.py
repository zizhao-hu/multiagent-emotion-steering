"""Sweep emotion-vector steering against advanced science / coding benchmarks.

For each (trait, alpha) cell:
  1. Load the cached unit vector for the trait at the chosen layer.
  2. Install a forward hook adding alpha*vector to the residual stream at L.
  3. Generate a completion for every benchmark task.
  4. Score, average, write JSON.

Cells with alpha=0 are the unsteered baseline — the same generation path,
hook installed with a zero-magnitude vector, so any diff is purely additive
steering, not e.g. tokenizer or sampling drift.

Usage:
    python scripts/steer_benchmark.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --layer 15 \\
        --vector-cache vectors/cache \\
        --traits joy sadness anger curiosity surprise \\
        --alphas -4 -2 0 2 4 \\
        --benchmarks gpqa humaneval \\
        --n-gpqa 100 --n-humaneval 50 \\
        --out-dir runs/10_steering_benchmarks/llama3_8b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intrinsic_agents.benchmarks import (
    load_gpqa_diamond,
    load_humaneval,
    score_gpqa,
    score_humaneval,
)
from intrinsic_agents.vectors.steering import SteeringHarness

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_vector(cache_dir: Path, model_name: str, trait: str, layer: int) -> torch.Tensor:
    safe = model_name.replace("/", "_")
    path = cache_dir / f"{safe}_{trait}_layer{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"missing vector: {path}. Run scripts/extract_vectors.py "
            f"--model {model_name} --layer {layer} first."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["vector"].float()


def chat_format(tok: AutoTokenizer, user_text: str) -> str:
    """Wrap raw user text in the model's chat template if it has one.

    Most modern instruct models route input through `apply_chat_template`,
    which inserts the right system/user/assistant tags and the assistant
    BOS marker. Falling back to raw text would silently degrade quality
    for chat-tuned bases.
    """
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        return tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_text


def run_benchmark(
    name: str,
    tasks: list,
    score_fn,
    harness: SteeringHarness,
    vector: torch.Tensor,
    alpha: float,
    tok: AutoTokenizer,
    max_new_tokens: int,
) -> dict:
    """Score every task; return per-task scores + mean."""
    scores: list[float] = []
    samples: list[str] = []
    harness._install(vector, alpha)
    try:
        for task in tasks:
            prompt = chat_format(tok, task.prompt)
            completion = harness.generate(prompt, max_new_tokens=max_new_tokens)
            s = score_fn(task, completion)
            scores.append(s)
            samples.append(completion)
    finally:
        harness.remove()
    mean = sum(scores) / max(len(scores), 1)
    return {
        "benchmark": name,
        "alpha": alpha,
        "mean_score": mean,
        "n": len(scores),
        "per_task": scores,
        # Keep first 3 samples for spot-checking; full samples bloat JSON.
        "samples_preview": samples[:3],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--vector-cache", default=str(REPO_ROOT / "vectors" / "cache"))
    p.add_argument("--traits", nargs="+",
                   default=["joy", "sadness", "anger", "curiosity", "surprise"])
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    p.add_argument("--benchmarks", nargs="+", default=["gpqa", "humaneval"],
                   choices=["gpqa", "humaneval"])
    p.add_argument("--n-gpqa", type=int, default=100,
                   help="number of GPQA-Diamond tasks (max ~198)")
    p.add_argument("--n-humaneval", type=int, default=50,
                   help="number of HumanEval tasks (max 164)")
    p.add_argument("--max-new-tokens-gpqa", type=int, default=256)
    p.add_argument("--max-new-tokens-humaneval", type=int, default=512)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.vector_cache)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"loading {args.model} on {device} ({args.dtype})...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    print("loaded.", flush=True)

    # Load benchmarks once. For 'gpqa' alpha=0 is identical across traits, but
    # we re-run anyway because sampling is stochastic — separate trait runs
    # give us a sense of baseline noise floor for each trait sweep.
    benches: dict = {}
    if "gpqa" in args.benchmarks:
        print(f"loading GPQA-Diamond ({args.n_gpqa} tasks)...", flush=True)
        benches["gpqa"] = (
            load_gpqa_diamond(n=args.n_gpqa, seed=args.seed),
            score_gpqa,
            args.max_new_tokens_gpqa,
        )
    if "humaneval" in args.benchmarks:
        print(f"loading HumanEval ({args.n_humaneval} tasks)...", flush=True)
        benches["humaneval"] = (
            load_humaneval(n=args.n_humaneval, seed=args.seed),
            score_humaneval,
            args.max_new_tokens_humaneval,
        )

    harness = SteeringHarness(model, tok, layer=args.layer)

    summary: dict = {
        "model": args.model,
        "layer": args.layer,
        "alphas": args.alphas,
        "traits": args.traits,
        "benchmarks": list(benches.keys()),
        "n": {name: len(tasks) for name, (tasks, _, _) in benches.items()},
        "results": {},  # trait -> bench -> alpha -> mean
    }

    for trait in args.traits:
        try:
            v = load_vector(cache_dir, args.model, trait, args.layer).to(device)
        except FileNotFoundError as e:
            print(f"  skipping {trait}: {e}", flush=True)
            continue
        summary["results"][trait] = {bench: {} for bench in benches}

        for alpha in args.alphas:
            for bench_name, (tasks, score_fn, max_tokens) in benches.items():
                t0 = time.time()
                rec = run_benchmark(
                    bench_name, tasks, score_fn, harness, v, alpha, tok, max_tokens,
                )
                dt = time.time() - t0
                summary["results"][trait][bench_name][str(alpha)] = rec["mean_score"]
                print(f"  {trait:<10} {bench_name:<10} alpha={alpha:+.1f}  "
                      f"mean={rec['mean_score']:.3f}  ({dt:.1f}s)", flush=True)

                cell_path = out_dir / f"{trait}_{bench_name}_alpha{alpha:+.1f}.json"
                cell_path.write_text(json.dumps(rec, indent=2))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
