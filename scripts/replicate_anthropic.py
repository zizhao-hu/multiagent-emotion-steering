"""Run the three Anthropic Persona Vectors replication checks.

TODO: implement end-to-end. This script currently defines the shape of the
output report — what PASS/FAIL means, what numbers go in the final table —
so the rest of the code can be filled in against a fixed contract.

Pipeline:
    for each (layer, trait):
      1. Load cached vector from vectors/cache/
      2. Extraction check — AUC on held-out contrastive pairs
      3. Steering check — sweep alpha, generate, judge, Spearman(alpha, score)
      4. Probing check — generate N free responses, Spearman(cos(h, v), judge)
    Write replication_report.json under runs/00_replication/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=str(REPO_ROOT / "runs" / "00_replication"))
    args = p.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scaffold report shape — fill in once extraction/steering/probing eval are
    # wired up. Any check falling below its target is a hard stop for
    # downstream experiments.
    report = {
        "config": str(cfg_path),
        "model": cfg["model"]["name"],
        "per_layer": {},
        "verdict": "UNIMPLEMENTED",
    }

    for layer in cfg.get("layer_sweep", [cfg["model"]["layer"]]):
        report["per_layer"][layer] = {
            trait: {
                "extraction_auc": None,
                "steering_spearman": None,
                "probing_spearman": None,
                "pass": None,
            }
            for trait in cfg["traits"]
        }

    (out_dir / "replication_report.json").write_text(json.dumps(report, indent=2))
    print(f"Replication report stub written to {out_dir/'replication_report.json'}")
    print("TODO: wire up extraction/steering/probing evals.")


if __name__ == "__main__":
    main()
