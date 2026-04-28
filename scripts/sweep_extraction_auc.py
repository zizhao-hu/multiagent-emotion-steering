"""Layer sweep — extraction AUC across multiple layers in one model load.

Loads the model once and evaluates extraction AUC at every requested layer,
saving per-layer JSON. Avoids paying the model-load cost N times when
picking a steering layer for a new base model.

Usage:
    python scripts/sweep_extraction_auc.py \\
        --model google/gemma-3-12b-it \\
        --layers 12 18 24 30 36 42 \\
        --out-dir runs/00_replication/gemma12b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intrinsic_agents.vectors.eval import extraction_auc_all
from intrinsic_agents.vectors.extract import load_traits

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAITS = REPO_ROOT / "src" / "intrinsic_agents" / "vectors" / "traits.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layers", type=int, nargs="+", required=True,
                   help="Layer indices to sweep, e.g. --layers 12 18 24 30 36 42")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--traits-yaml", default=str(DEFAULT_TRAITS))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-auc", type=float, default=0.85)
    p.add_argument("--readout", default="last", choices=["last", "mean"],
                   help="residual-stream readout point: 'last' = last prompt token "
                   "(Anthropic default), 'mean' = mean-pool over all prompt tokens "
                   "(more robust when the trait signal is distributed)")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"loading {args.model} on {device} ({args.dtype})...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    print("loaded.", flush=True)

    traits = load_traits(args.traits_yaml)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"readout: {args.readout}", flush=True)
    summary: dict[int, dict[str, float]] = {}
    for L in args.layers:
        print(f"\n=== layer {L} (readout={args.readout}) ===", flush=True)
        results = extraction_auc_all(model, tok, traits, layer=L, device=device,
                                     readout=args.readout)

        passes = 0
        per_trait = {}
        for name, r in results.items():
            ok = r.passes(args.target_auc)
            passes += int(ok)
            per_trait[name] = round(r.auc, 4)
            mark = "PASS" if ok else "fail"
            print(f"  {name:<16} {r.auc:>6.3f}  {mark}", flush=True)
        print(f"  -> {passes}/{len(results)} pass AUC >= {args.target_auc}", flush=True)

        out_path = out_dir / f"extraction_auc_layer{L}_{args.readout}.json"
        out_path.write_text(json.dumps({
            "model": args.model,
            "layer": L,
            "dtype": args.dtype,
            "readout": args.readout,
            "target_auc": args.target_auc,
            "results": {
                name: {
                    "auc": r.auc,
                    "n_pairs": r.n_pairs,
                    "pos": r.per_pair_pos,
                    "neg": r.per_pair_neg,
                }
                for name, r in results.items()
            },
        }, indent=2))
        summary[L] = per_trait
        print(f"  wrote {out_path}", flush=True)

    # Cross-layer summary table.
    print("\n=== summary across layers ===", flush=True)
    trait_names = sorted({n for d in summary.values() for n in d})
    header = f"  {'trait':<16} " + " ".join(f"L{L:>3d}" for L in args.layers)
    print(header, flush=True)
    print("  " + "-" * (16 + 6 * len(args.layers)), flush=True)
    for name in trait_names:
        cells = " ".join(f"{summary[L].get(name, float('nan')):.3f}" for L in args.layers)
        print(f"  {name:<16} {cells}", flush=True)

    # Pick best layer by mean AUC across emotion traits, since that is the
    # subset we care about for the steering experiments.
    emotion_traits = [n for n in trait_names if n in
                      {"joy", "sadness", "anger", "curiosity", "surprise"}]
    if emotion_traits:
        means = {L: sum(summary[L][n] for n in emotion_traits) / len(emotion_traits)
                 for L in args.layers}
        best = max(means, key=means.get)
        print(f"\n  best layer for emotion traits: L{best}  (mean AUC = {means[best]:.3f})",
              flush=True)
        print(f"  per-layer mean AUC over emotions: " +
              ", ".join(f"L{L}={means[L]:.3f}" for L in args.layers), flush=True)

    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps({
        "model": args.model,
        "layers": args.layers,
        "per_layer_per_trait": summary,
    }, indent=2))
    print(f"\nwrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
