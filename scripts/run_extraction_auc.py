"""Replication gate 1 — leave-one-out extraction AUC on every trait.

Usage:
    python scripts/run_extraction_auc.py --model gpt2 --device cpu
    python scripts/run_extraction_auc.py --model Qwen/Qwen2.5-7B-Instruct --layer 14
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intrinsic_agents.vectors.eval import extraction_auc_all
from intrinsic_agents.vectors.extract import default_layer, load_traits

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAITS = REPO_ROOT / "src" / "intrinsic_agents" / "vectors" / "traits.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--traits-yaml", default=str(DEFAULT_TRAITS))
    p.add_argument("--out", default=None)
    p.add_argument("--target-auc", type=float, default=0.85)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    layer = args.layer if args.layer is not None else default_layer(model)
    traits = load_traits(args.traits_yaml)

    results = extraction_auc_all(model, tok, traits, layer=layer, device=device)

    print(f"\nExtraction AUC · model={args.model} · layer={layer} · device={device}\n")
    print(f"  {'trait':<16} {'AUC':>6}  {'pass':>5}   (n pairs)")
    print("  " + "-" * 46)
    passes = 0
    for name, r in results.items():
        ok = r.passes(args.target_auc)
        passes += int(ok)
        mark = "  PASS" if ok else "  fail"
        print(f"  {name:<16} {r.auc:>6.3f}  {mark:>5}   ({r.n_pairs})")
    print("  " + "-" * 46)
    print(f"  {passes}/{len(results)} traits meet AUC >= {args.target_auc}\n")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "layer": layer,
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
                },
                indent=2,
            )
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
