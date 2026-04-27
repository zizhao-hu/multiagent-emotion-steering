"""Apply the multi-strategy extractor (now built into demo_task_sweep.py) to
the saved n=500 results.json files, in-place.

The seeded portion (problems 0-199) was scored under the old extractor and
needs to be re-scored. The new portion (200-499) was scored with the new
extractor (because the worker process re-imports the module on each retry).
Either way, re-scoring is idempotent: applying the new extractor to a
result that already used it produces the same predicted/correct values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from demo_task_sweep import extract_answer, score_answer  # noqa: E402

CONDITIONS = ["control", "joy+", "joy-", "sadness+", "anger+", "curiosity+", "surprise+"]


def patch_dir(out_dir: Path) -> dict:
    res_path = out_dir / "results.json"
    prob_path = out_dir / "problems.json"
    if not res_path.exists() or not prob_path.exists():
        print(f"  {out_dir}: skipped (missing files)")
        return {}
    res = json.loads(res_path.read_text())
    problems = json.loads(prob_path.read_text())
    summary = {}
    for cond in CONDITIONS:
        rs = res.get(cond, [])
        flips = 0
        for i, r in enumerate(rs):
            if i >= len(problems):
                continue
            txt = "\n".join(r.get("transcript", []))
            new_pred = extract_answer(txt)
            new_correct = score_answer(new_pred, problems[i]["gold"])
            if new_pred != r.get("predicted") or new_correct != int(r.get("correct", 0)):
                if not r.get("correct") and new_correct:
                    flips += 1
                r["predicted"] = new_pred
                r["correct"] = new_correct
        summary[cond] = {
            "n": len(rs),
            "n_correct": sum(int(r.get("correct", 0)) for r in rs),
            "flipped_to_correct": flips,
        }
    res_path.write_text(json.dumps(res, indent=2))
    return summary


def main():
    for sub in ("task_sweep_n500", "task_sweep_n500_bobfirst"):
        d = REPO / "runs" / "demo" / sub
        print(f"\n=== {sub} ===")
        sm = patch_dir(d)
        for cond in CONDITIONS:
            s = sm.get(cond, {})
            if not s:
                continue
            print(f"  {cond}: {s['n_correct']}/{s['n']} (+{s['flipped_to_correct']} fixed)")


if __name__ == "__main__":
    main()
