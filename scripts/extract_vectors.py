"""Extract persona/emotion vectors and cache them to vectors/cache/.

Usage:
    python scripts/extract_vectors.py --model gpt2
    python scripts/extract_vectors.py --model Qwen/Qwen2.5-3B-Instruct --traits honesty joy
    python scripts/extract_vectors.py --model gpt2 --layer 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from intrinsic_agents.vectors.extract import extract_all

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAITS_YAML = REPO_ROOT / "src" / "intrinsic_agents" / "vectors" / "traits.yaml"
DEFAULT_CACHE = REPO_ROOT / "vectors" / "cache"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id, e.g. gpt2 or Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--layer", type=int, default=None, help="Which layer's residual stream to probe (default: middle)")
    p.add_argument("--traits", nargs="+", default=None, help="Subset of trait names (default: all in traits.yaml)")
    p.add_argument("--traits-yaml", default=str(DEFAULT_TRAITS_YAML))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p.add_argument("--device", default=None)
    args = p.parse_args()

    written = extract_all(
        model_name=args.model,
        traits_yaml=args.traits_yaml,
        out_dir=args.cache_dir,
        layer=args.layer,
        device=args.device,
        selected=args.traits,
    )
    for name, path in written.items():
        print(f"  {name:16s} -> {path}")


if __name__ == "__main__":
    main()
