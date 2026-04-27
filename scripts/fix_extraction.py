"""Re-extract answers from saved transcripts using a multi-strategy parser.

The original extractor (scripts/demo_task_sweep.py:extract_answer) has a known
weakness: when no "Final answer: X" marker is present, it falls back to the
LAST number in the last 300 chars — which often catches a comparison number
(e.g., "$125 is higher than $96" → grabs 96 instead of 125).

This script re-parses every saved transcript with a stronger extractor that:

  1) Picks the LAST "Final answer: X" match (not the first).
  2) Picks the LAST `\\boxed{X}` match.
  3) Picks the LAST "the answer is X" match.
  4) For the last agent turn only: looks for "X is higher / lower / greater /
     larger / smaller / more / less" patterns and picks X.
  5) Looks for "= X" at the end of an arithmetic line in the last 2 turns.
  6) Falls back to last number in last 300 chars (original behavior).

A "fix" is recorded only when:
  - original extractor said WRONG
  - improved extractor produces a number that matches gold (within 1e-3)

Conservative — never flips wrong→right unless the gold appears in a
conclusion-shaped position. Outputs:
  analysis/extract_fixes.json      — list of fixed cases per condition
  analysis/results_alice_fixed.json — patched results.json for alice-first
  analysis/results_bob_fixed.json   — patched results.json for bob-first

Then re-runs the synthesis with fixed numbers.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANA = REPO / "analysis"
CONDITIONS = ["control", "joy+", "joy-", "sadness+", "anger+", "curiosity+", "surprise+"]


def _to_float(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def improved_extract(transcript_text: str) -> tuple[float | None, str]:
    """Return (predicted, strategy_used)."""
    text = transcript_text.replace(",", "")

    # Strategy 1: LAST "Final answer: X" — last, not first
    matches = list(re.finditer(
        r"[Ff]inal\s*answer\s*(?:is|:|=)?\s*\$?\s*(-?\d+(?:\.\d+)?)", text
    ))
    if matches:
        return _to_float(matches[-1].group(1)), "final_answer_last"

    # Strategy 2: LAST \boxed{X}
    matches = list(re.finditer(r"\\boxed\{\s*\$?\s*(-?\d+(?:\.\d+)?)\s*\}", text))
    if matches:
        return _to_float(matches[-1].group(1)), "boxed_last"

    # Strategy 3: LAST "the answer is X"
    matches = list(re.finditer(
        r"[Tt]he\s+answer\s+is\s*\$?\s*(-?\d+(?:\.\d+)?)", text
    ))
    if matches:
        return _to_float(matches[-1].group(1)), "answer_is_last"

    # For the next strategies, isolate the last 2 agent turns
    turn_lines = [ln for ln in transcript_text.splitlines() if re.match(r"\[\d+\]", ln)]
    last_two = "\n".join(turn_lines[-2:]).replace(",", "")

    # Strategy 4: comparison-claim "X is higher/lower/greater/etc"
    matches = list(re.finditer(
        r"\$?\s*(-?\d+(?:\.\d+)?)\s*(?:[A-Za-z][^.\n]*)?\s+is\s+(?:higher|lower|greater|larger|smaller|more|less|bigger|the\s+(?:correct|final|right))",
        last_two,
        flags=re.IGNORECASE,
    ))
    if matches:
        return _to_float(matches[-1].group(1)), "comparison_claim"

    # Strategy 5: last "= X" at end of an arithmetic line in last 2 turns
    matches = list(re.finditer(
        r"=\s*\$?\s*(-?\d+(?:\.\d+)?)\s*(?:\.|$|\s*$|\s+(?:so|which|this))",
        last_two,
    ))
    if matches:
        return _to_float(matches[-1].group(1)), "equals_last"

    # Strategy 6: original fallback — last number in last 300 chars
    nums = re.findall(r"-?\d+(?:\.\d+)?", text[-300:])
    if nums:
        return float(nums[-1]), "fallback_last300"
    return None, "no_match"


def score_against_gold(pred: float | None, gold: float | None) -> int:
    if pred is None or gold is None:
        return 0
    return int(abs(pred - gold) < 1e-3)


def reprocess(results: dict, problems: list[dict]) -> tuple[dict, list[dict]]:
    """Return (patched_results, fix_records)."""
    patched: dict = {}
    fixes: list[dict] = []
    for cond in CONDITIONS:
        rs = results.get(cond, [])
        new_rs = []
        for i, r in enumerate(rs):
            new_r = dict(r)
            if i >= len(problems):
                new_rs.append(new_r)
                continue
            gold = problems[i]["gold"]
            txt = "\n".join(r.get("transcript", []))
            new_pred, strategy = improved_extract(txt)
            new_correct = score_against_gold(new_pred, gold)
            old_correct = bool(r.get("correct", False))
            if not old_correct and new_correct:
                # Conservative: only flip if the new strategy is one of the
                # high-confidence ones (NOT the fallback)
                if strategy != "fallback_last300":
                    new_r["predicted"] = new_pred
                    new_r["correct"] = 1
                    new_r["fixed"] = True
                    new_r["fix_strategy"] = strategy
                    fixes.append(
                        {
                            "condition": cond,
                            "idx": problems[i]["idx"],
                            "gold": gold,
                            "old_predicted": r.get("predicted"),
                            "new_predicted": new_pred,
                            "strategy": strategy,
                            "n_turns": r.get("n_turns"),
                        }
                    )
            new_rs.append(new_r)
        patched[cond] = new_rs
    return patched, fixes


def main():
    alice = json.loads((ANA / "snapshot_alice_n200.json").read_text())
    bob = json.loads((ANA / "snapshot_bob_n200.json").read_text())
    problems = json.loads((ANA / "snapshot_problems_n200.json").read_text())

    alice_fixed, alice_fixes = reprocess(alice, problems)
    bob_fixed, bob_fixes = reprocess(bob, problems)

    (ANA / "results_alice_fixed.json").write_text(json.dumps(alice_fixed, indent=2))
    (ANA / "results_bob_fixed.json").write_text(json.dumps(bob_fixed, indent=2))
    (ANA / "extract_fixes.json").write_text(
        json.dumps({"alice": alice_fixes, "bob": bob_fixes}, indent=2)
    )

    # Summary
    print(f"alice fixes: {len(alice_fixes)}  bob fixes: {len(bob_fixes)}")
    for label, fixes in (("alice", alice_fixes), ("bob", bob_fixes)):
        by_strategy = Counter(f["strategy"] for f in fixes)
        by_cond = Counter(f["condition"] for f in fixes)
        print(f"\n{label}-first strategy distribution:")
        for s, n in by_strategy.most_common():
            print(f"  {s}: {n}")
        print(f"{label}-first fixes per condition:")
        for c in CONDITIONS:
            print(f"  {c}: {by_cond.get(c, 0)}")

    # Updated accuracy table
    print("\n=== updated accuracy (after extraction fixes) ===")
    print(f"{'cond':<11} {'alice old':>14} {'alice new':>14} {'bob old':>14} {'bob new':>14}")
    for cond in CONDITIONS:
        a_old = sum(int(r.get("correct", False)) for r in alice[cond])
        a_new = sum(int(r.get("correct", False)) for r in alice_fixed[cond])
        b_old = sum(int(r.get("correct", False)) for r in bob[cond])
        b_new = sum(int(r.get("correct", False)) for r in bob_fixed[cond])
        print(
            f"{cond:<11} {a_old:3d}/200 ({a_old/200:.2%})  {a_new:3d}/200 ({a_new/200:.2%})  "
            f"{b_old:3d}/200 ({b_old/200:.2%})  {b_new:3d}/200 ({b_new/200:.2%})"
        )


if __name__ == "__main__":
    main()
