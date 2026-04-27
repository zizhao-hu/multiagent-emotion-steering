"""Case-by-case analysis of the GSM8K task sweep.

Loads alice-first and bob-first results.json (n=200 snapshot under analysis/),
classifies every problem outcome, groups failure modes, and computes
trait-projection ↔ correctness correlations to test whether the emotion
steering signal is mechanically connected to the math outcome.

Outputs:
  analysis/case_report.md          — markdown report (the main deliverable)
  analysis/outcome_matrix.json     — per-problem 14-cell outcome record
  analysis/flip_examples.json      — sample transcripts for each interesting flip class
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANA = REPO / "analysis"
ANA.mkdir(exist_ok=True)

ALICE_FILE = ANA / "snapshot_alice_n200.json"
BOB_FILE = ANA / "snapshot_bob_n200.json"
PROBLEMS_FILE = ANA / "snapshot_problems_n200.json"

CONDITIONS = ["control", "joy+", "joy-", "sadness+", "anger+", "curiosity+", "surprise+"]
EMOTIONS = ["joy", "sadness", "anger", "curiosity", "surprise"]
STEERED_EMOTION = {  # condition -> (trait_name, sign)
    "joy+": ("joy", +1),
    "joy-": ("joy", -1),
    "sadness+": ("sadness", +1),
    "anger+": ("anger", +1),
    "curiosity+": ("curiosity", +1),
    "surprise+": ("surprise", +1),
}


def classify_failure(r: dict, gold: float | None) -> str:
    """Heuristic failure-mode classifier.  Returns a short tag."""
    pred = r.get("predicted")
    transcript = "\n".join(r.get("transcript", []))
    n_turns = r.get("n_turns", 0)
    if r.get("correct"):
        if n_turns == 1:
            return "correct_oneturn"
        if n_turns <= 3:
            return "correct_fast"
        return "correct_slow"
    # wrong cases below
    if pred is None:
        return "no_answer"
    if gold is None:
        return "no_gold"
    try:
        pred_f = float(pred)
        gold_f = float(gold)
    except (TypeError, ValueError):
        return "extract_error"
    if gold_f == 0:
        diff_rel = abs(pred_f)
    else:
        diff_rel = abs(pred_f - gold_f) / max(abs(gold_f), 1e-9)
    # magnitude error: off by >= 10x
    if pred_f != 0:
        ratio = abs(pred_f / gold_f) if gold_f != 0 else float("inf")
        if ratio >= 9.5 or ratio <= 1 / 9.5:
            return "magnitude_error"
    if diff_rel < 0.05:
        return "near_miss"  # within 5%
    if n_turns == 10:
        return "didnt_converge"
    if n_turns == 1:
        return "early_exit_wrong"
    # agent disagreement signature: many distinct numeric proposals
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", transcript)
    distinct_big = len({n for n in nums if float(n) > 1})
    if distinct_big >= 6 and n_turns >= 5:
        return "many_proposals"
    return "wrong_other"


def load_results():
    alice = json.loads(ALICE_FILE.read_text())
    bob = json.loads(BOB_FILE.read_text())
    problems = json.loads(PROBLEMS_FILE.read_text())
    return alice, bob, problems


def build_outcome_matrix(alice, bob, problems):
    n = len(problems)
    rows = []
    for i, prob in enumerate(problems):
        cell = {"idx": prob["idx"], "gold": prob["gold"], "question": prob["question"]}
        for ord_name, src in (("alice", alice), ("bob", bob)):
            for cond in CONDITIONS:
                rs = src.get(cond, [])
                if i < len(rs):
                    r = rs[i]
                    cell[f"{ord_name}.{cond}.correct"] = bool(r["correct"])
                    cell[f"{ord_name}.{cond}.predicted"] = r.get("predicted")
                    cell[f"{ord_name}.{cond}.turns"] = r.get("n_turns")
                    cell[f"{ord_name}.{cond}.failure"] = classify_failure(r, prob["gold"])
        rows.append(cell)
    return rows


def flip_pairs(alice, bob, problems, ordering: str):
    """Per-emotion flip lists vs same-order control.

    Returns dict: cond -> {"helped": [idx,...], "hurt": [idx,...]}.
    """
    src = alice if ordering == "alice" else bob
    out = {}
    ctrl = src.get("control", [])
    n = min(len(ctrl), len(problems))
    for cond in CONDITIONS:
        if cond == "control":
            continue
        rs = src.get(cond, [])
        helped, hurt = [], []
        for i in range(min(n, len(rs))):
            if not ctrl[i]["correct"] and rs[i]["correct"]:
                helped.append(i)
            elif ctrl[i]["correct"] and not rs[i]["correct"]:
                hurt.append(i)
        out[cond] = {"helped": helped, "hurt": hurt}
    return out


def ordering_flips(alice, bob, problems):
    """Per-condition: alice-correct ⊕ bob-correct (the ordering changed it)."""
    out = {}
    for cond in CONDITIONS:
        a = alice.get(cond, [])
        b = bob.get(cond, [])
        n = min(len(a), len(b), len(problems))
        a_only, b_only, both_right, both_wrong = [], [], 0, 0
        for i in range(n):
            ca, cb = a[i]["correct"], b[i]["correct"]
            if ca and not cb:
                a_only.append(i)
            elif cb and not ca:
                b_only.append(i)
            elif ca and cb:
                both_right += 1
            else:
                both_wrong += 1
        out[cond] = {
            "alice_only": a_only,
            "bob_only": b_only,
            "both_right": both_right,
            "both_wrong": both_wrong,
        }
    return out


def trait_projection_correlation(alice, bob, problems):
    """For each (ordering, condition), correlate alice's mean projection on the
    steered trait with the per-problem outcome (1=correct, 0=wrong).

    Also: bob's trait drift on the same trait (the contagion target) vs outcome.
    """
    rows = []
    for ordering, src in (("alice", alice), ("bob", bob)):
        for cond, (trait, sign) in STEERED_EMOTION.items():
            rs = src.get(cond, [])
            if not rs:
                continue
            xs_alice, xs_bob_drift, ys = [], [], []
            for r in rs:
                tr = r.get("trajectories", {})
                a_traj = tr.get("alice", {}).get(trait, [])
                b_traj = tr.get("bob", {}).get(trait, [])
                if not a_traj:
                    continue
                a_mean = sum(a_traj) / len(a_traj)
                b_drift = (b_traj[-1] - b_traj[0]) if len(b_traj) >= 2 else 0.0
                xs_alice.append(a_mean)
                xs_bob_drift.append(b_drift)
                ys.append(1.0 if r["correct"] else 0.0)
            rows.append(
                {
                    "ordering": ordering,
                    "condition": cond,
                    "trait": trait,
                    "n": len(ys),
                    "alice_mean": statistics.mean(xs_alice) if xs_alice else 0.0,
                    "bob_drift_mean": statistics.mean(xs_bob_drift) if xs_bob_drift else 0.0,
                    "acc": statistics.mean(ys) if ys else 0.0,
                    "corr_alice_proj_correct": _pearson(xs_alice, ys),
                    "corr_bob_drift_correct": _pearson(xs_bob_drift, ys),
                }
            )
    return rows


def _pearson(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def per_condition_failure_dist(alice, bob, problems):
    """For each (ordering, condition), count failure modes."""
    out = {}
    for ordering, src in (("alice", alice), ("bob", bob)):
        for cond in CONDITIONS:
            rs = src.get(cond, [])
            counts = Counter()
            for i, r in enumerate(rs):
                gold = problems[i]["gold"] if i < len(problems) else None
                counts[classify_failure(r, gold)] += 1
            out[f"{ordering}.{cond}"] = dict(counts)
    return out


def example_transcripts(rows, alice, bob, problems, max_per_class=3):
    """Sample transcripts that illustrate each (ordering × emotion × flip-class)."""
    samples = {}
    for ordering, src in (("alice", alice), ("bob", bob)):
        ctrl = src["control"]
        for cond in CONDITIONS:
            if cond == "control":
                continue
            rs = src.get(cond, [])
            for kind in ("helped", "hurt"):
                key = f"{ordering}.{cond}.{kind}"
                samples[key] = []
                for i, r in enumerate(rs):
                    if i >= len(ctrl):
                        break
                    cr = ctrl[i]["correct"]
                    er = r["correct"]
                    if kind == "helped" and not cr and er:
                        samples[key].append(_sample_pack(i, ctrl[i], r, problems[i]))
                    elif kind == "hurt" and cr and not er:
                        samples[key].append(_sample_pack(i, ctrl[i], r, problems[i]))
                    if len(samples[key]) >= max_per_class:
                        break
    return samples


def _sample_pack(i, ctrl_r, emo_r, prob):
    return {
        "idx": prob["idx"],
        "question": prob["question"],
        "gold": prob["gold"],
        "control": {
            "predicted": ctrl_r.get("predicted"),
            "n_turns": ctrl_r.get("n_turns"),
            "transcript": ctrl_r.get("transcript", []),
            "failure": classify_failure(ctrl_r, prob["gold"]),
        },
        "emotion": {
            "predicted": emo_r.get("predicted"),
            "n_turns": emo_r.get("n_turns"),
            "transcript": emo_r.get("transcript", []),
            "failure": classify_failure(emo_r, prob["gold"]),
        },
    }


def build_report(alice, bob, problems):
    rows = build_outcome_matrix(alice, bob, problems)
    flips_a = flip_pairs(alice, bob, problems, "alice")
    flips_b = flip_pairs(alice, bob, problems, "bob")
    ord_flips = ordering_flips(alice, bob, problems)
    fail_dist = per_condition_failure_dist(alice, bob, problems)
    relevance = trait_projection_correlation(alice, bob, problems)
    examples = example_transcripts(rows, alice, bob, problems, max_per_class=3)

    # Save artifacts
    (ANA / "outcome_matrix.json").write_text(json.dumps(rows, indent=2))
    (ANA / "flip_examples.json").write_text(json.dumps(examples, indent=2))

    # Build markdown report
    lines = []
    lines.append("# Case-by-case analysis — task_sweep n=200")
    lines.append("")
    lines.append("Outcome flips and failure-mode breakdown for the 7-condition × 2-ordering sweep.")
    lines.append("")

    # ---- 1. Per-problem outcome distribution ----
    lines.append("## 1. Outcome distribution across 14 cells per problem")
    lines.append("")
    lines.append("Each problem has 14 outcomes (alice-first/bob-first × 7 conditions). Distribution of \"how many of 14 cells were correct\":")
    lines.append("")
    counts = Counter()
    for r in rows:
        n_correct = sum(
            int(r.get(f"{o}.{c}.correct", False))
            for o in ("alice", "bob")
            for c in CONDITIONS
        )
        counts[n_correct] += 1
    lines.append("| n_correct | n_problems |")
    lines.append("|---|---|")
    for k in sorted(counts):
        lines.append(f"| {k}/14 | {counts[k]} |")
    n_always_right = counts.get(14, 0)
    n_always_wrong = counts.get(0, 0)
    n_sometimes = sum(v for k, v in counts.items() if 0 < k < 14)
    lines.append("")
    lines.append(
        f"**{n_always_right}/{len(rows)} always correct, {n_always_wrong}/{len(rows)} always wrong, "
        f"{n_sometimes}/{len(rows)} sometimes correct (these are where ordering/steering matters).**"
    )
    lines.append("")

    # ---- 2. Per-emotion flip table ----
    lines.append("## 2. Flips vs same-order control")
    lines.append("")
    for ordering, flips in (("alice-first", flips_a), ("bob-first", flips_b)):
        lines.append(f"### {ordering}")
        lines.append("")
        lines.append("| condition | helped (ctrl wrong → emo right) | hurt (ctrl right → emo wrong) | net |")
        lines.append("|---|---:|---:|---:|")
        for cond in CONDITIONS:
            if cond == "control":
                continue
            f = flips[cond]
            net = len(f["helped"]) - len(f["hurt"])
            lines.append(f"| {cond} | {len(f['helped'])} | {len(f['hurt'])} | {net:+d} |")
        lines.append("")

    # ---- 3. Ordering flips per condition ----
    lines.append("## 3. Ordering flips (alice-only correct vs bob-only correct, same condition)")
    lines.append("")
    lines.append("| condition | alice-only | bob-only | both right | both wrong |")
    lines.append("|---|---:|---:|---:|---:|")
    for cond in CONDITIONS:
        of = ord_flips[cond]
        lines.append(
            f"| {cond} | {len(of['alice_only'])} | {len(of['bob_only'])} | {of['both_right']} | {of['both_wrong']} |"
        )
    lines.append("")

    # ---- 4. Failure-mode taxonomy ----
    lines.append("## 4. Failure-mode breakdown")
    lines.append("")
    lines.append("Heuristic classifier:")
    lines.append("")
    lines.append("- `correct_*`: correct (oneturn / fast≤3 / slow>3)")
    lines.append("- `near_miss`: within 5% of gold")
    lines.append("- `magnitude_error`: off by ≥10× (decimal-shift / unit error)")
    lines.append("- `didnt_converge`: 10 turns, no final answer")
    lines.append("- `early_exit_wrong`: 1 turn, wrong")
    lines.append("- `many_proposals`: ≥6 distinct large numbers in transcript (agent disagreement)")
    lines.append("- `wrong_other`: catch-all wrong")
    lines.append("")
    all_modes = set()
    for d in fail_dist.values():
        all_modes.update(d)
    all_modes = sorted(all_modes)
    header = "| condition |" + " | ".join(all_modes) + " |"
    sep = "|---|" + "|".join(["---"] * len(all_modes)) + "|"
    lines.append(header)
    lines.append(sep)
    for ordering in ("alice", "bob"):
        for cond in CONDITIONS:
            row = fail_dist[f"{ordering}.{cond}"]
            cells = [str(row.get(m, 0)) for m in all_modes]
            lines.append(f"| {ordering}/{cond} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- 5. Steering relevance ----
    lines.append("## 5. Steering relevance — does the steered trait actually correlate with outcome?")
    lines.append("")
    lines.append(
        "For each condition, we computed Pearson correlation between (a) alice's mean projection on "
        "the steered trait and (b) the joint correctness (1/0). If the steering had a mechanical "
        "effect on the math outcome through the trait expression, |r| should be non-trivial. "
        "Same for bob's drift on that trait (the contagion target)."
    )
    lines.append("")
    lines.append("| ordering | condition | trait | n | alice μ proj | bob drift μ | acc | r(alice·proj→correct) | r(bob·drift→correct) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in relevance:
        lines.append(
            f"| {r['ordering']} | {r['condition']} | {r['trait']} | {r['n']} | "
            f"{r['alice_mean']:+.3f} | {r['bob_drift_mean']:+.3f} | {r['acc']:.3f} | "
            f"{r['corr_alice_proj_correct']:+.3f} | {r['corr_bob_drift_correct']:+.3f} |"
        )
    lines.append("")
    sig = [r for r in relevance if abs(r["corr_alice_proj_correct"]) > 0.15]
    lines.append(f"**Conditions where |r(alice proj → correct)| > 0.15:** {len(sig)} / {len(relevance)}")
    for r in sig:
        lines.append(
            f"- {r['ordering']}/{r['condition']}: r={r['corr_alice_proj_correct']:+.3f}"
        )
    lines.append("")

    # ---- 6. Spotlight cases ----
    lines.append("## 6. Spotlight cases — sample transcripts of flips")
    lines.append("")
    lines.append("Three problems per (ordering × emotion × helped|hurt) class. The same problem "
                 "appears in both control and emotion-steered transcripts so you can see what "
                 "actually changed in the dialogue.")
    lines.append("")
    for ordering in ("alice", "bob"):
        for cond in CONDITIONS:
            if cond == "control":
                continue
            for kind in ("helped", "hurt"):
                key = f"{ordering}.{cond}.{kind}"
                if not examples.get(key):
                    continue
                lines.append(f"### {ordering}-first / {cond} / {kind}")
                lines.append("")
                for ex in examples[key]:
                    lines.append(f"**Problem #{ex['idx']}** (gold = {ex['gold']})  ")
                    lines.append("")
                    lines.append(f"> {ex['question'].splitlines()[0][:160]}…")
                    lines.append("")
                    lines.append(
                        f"- **control** ({ex['control']['failure']}): predicted={ex['control']['predicted']}, "
                        f"{ex['control']['n_turns']} turns"
                    )
                    lines.append(
                        f"- **{cond}** ({ex['emotion']['failure']}): predicted={ex['emotion']['predicted']}, "
                        f"{ex['emotion']['n_turns']} turns"
                    )
                    lines.append("")
                lines.append("")

    # Final write
    out = ANA / "case_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}  ({len(lines)} lines)")
    print(f"wrote {ANA/'outcome_matrix.json'}")
    print(f"wrote {ANA/'flip_examples.json'}")


def main():
    alice, bob, problems = load_results()
    print(f"loaded {len(problems)} problems, alice conds {list(alice.keys())}, bob conds {list(bob.keys())}")
    build_report(alice, bob, problems)


if __name__ == "__main__":
    main()
