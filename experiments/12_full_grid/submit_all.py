"""Generate sbatch dispatch commands for the full experiment grid.

Reads layer_map.py for per-trait best layers per model, then enumerates
the requested subset of (model, trait, bench, setting) cells. By default
prints the sbatch commands to stdout (dry run). Pass --execute to actually
ssh and submit.

Usage examples:
    # Dry-run print all 88 single-agent cells per model:
    python experiments/12_full_grid/submit_all.py --settings single

    # Submit only the dialogue cells, both modes, just for joy:
    python experiments/12_full_grid/submit_all.py \\
        --settings dialogue --traits joy --execute

    # Smoke: one cell per setting, on Llama only, joy x gsm8k:
    python experiments/12_full_grid/submit_all.py \\
        --models meta-llama/Llama-3.1-8B-Instruct \\
        --traits joy --benchmarks gsm8k --execute

    # Skip cells whose output dir already exists on the cluster:
    python experiments/12_full_grid/submit_all.py --skip-existing --execute
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from layer_map import (  # noqa: E402
    ALL_BENCHMARKS, ALL_MODELS, ALL_STEER_MODES, ALL_TRAITS,
    LAYER_BY_TRAIT, MODEL_SHORT,
)

REMOTE_REPO = "/project2/jessetho_1732/zizhaoh/multiagent-emotion-steering"


def cell_remote_dir(setting: str, model: str, trait: str, bench: str,
                    layer: int, steer_mode: str | None) -> str:
    short = MODEL_SHORT[model]
    if setting == "single":
        return f"{REMOTE_REPO}/runs/12_full_grid/single/{short}_{trait}_{bench}_L{layer}"
    return f"{REMOTE_REPO}/runs/12_full_grid/dialogue/{short}_{trait}_{bench}_L{layer}_{steer_mode}"


def remote_dir_exists(remote: str) -> bool:
    """Check if a directory exists on Endeavour via ssh."""
    cmd = ["ssh", "endeavour", f"test -d {shlex.quote(remote)} && echo y || echo n"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return out == "y"


def build_sbatch_cmd(setting: str, model: str, trait: str, layer: int,
                     bench: str, steer_mode: str | None) -> str:
    short = MODEL_SHORT[model]
    if setting == "single":
        job = f"fg_single_{short}_{trait}_{bench}"
        export = f"ALL,MODEL={shlex.quote(model)},TRAIT={trait},LAYER={layer},BENCH={bench}"
        script = "experiments/12_full_grid/run_single.sh"
    else:
        job = f"fg_dlg_{short}_{trait}_{bench}_{steer_mode}"
        export = (f"ALL,MODEL={shlex.quote(model)},TRAIT={trait},LAYER={layer},"
                  f"BENCH={bench},STEER_MODE={steer_mode}")
        script = "experiments/12_full_grid/run_dialogue.sh"
    return (f"cd {REMOTE_REPO} && sbatch -J {job} "
            f"--export={export} {script}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--traits", nargs="+", default=ALL_TRAITS,
                   choices=ALL_TRAITS)
    p.add_argument("--benchmarks", nargs="+", default=ALL_BENCHMARKS,
                   choices=ALL_BENCHMARKS)
    p.add_argument("--settings", nargs="+", default=["single", "dialogue"],
                   choices=["single", "dialogue"])
    p.add_argument("--steer-modes", nargs="+", default=ALL_STEER_MODES,
                   choices=ALL_STEER_MODES,
                   help="Only used for dialogue setting.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Probe Endeavour for each cell's output dir; skip if present.")
    p.add_argument("--execute", action="store_true",
                   help="Actually submit. Default is dry-run (print commands).")
    args = p.parse_args()

    cells: list[tuple[str, str, str, int, str, str | None]] = []
    for setting in args.settings:
        for model in args.models:
            for trait in args.traits:
                layer = LAYER_BY_TRAIT.get((model, trait))
                if layer is None:
                    print(f"# missing layer for ({model}, {trait}); skipping",
                          file=sys.stderr)
                    continue
                for bench in args.benchmarks:
                    if setting == "single":
                        cells.append((setting, model, trait, layer, bench, None))
                    else:
                        for sm in args.steer_modes:
                            cells.append((setting, model, trait, layer, bench, sm))

    print(f"# {len(cells)} cells planned", file=sys.stderr)
    submitted = skipped = 0
    for setting, model, trait, layer, bench, sm in cells:
        if args.skip_existing:
            remote = cell_remote_dir(setting, model, trait, bench, layer, sm)
            if remote_dir_exists(remote):
                skipped += 1
                continue
        cmd = build_sbatch_cmd(setting, model, trait, layer, bench, sm)
        if args.execute:
            ssh_cmd = ["ssh", "endeavour", cmd]
            r = subprocess.run(ssh_cmd, capture_output=True, text=True)
            line = r.stdout.strip() or r.stderr.strip()
            print(f"  {line}  ({setting} {MODEL_SHORT[model]} {trait} {bench}"
                  f"{' ' + sm if sm else ''})")
            submitted += 1
        else:
            print(cmd)

    if args.execute:
        print(f"# submitted={submitted}  skipped(existing)={skipped}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
