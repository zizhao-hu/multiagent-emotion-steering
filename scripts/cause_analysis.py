"""Per-sample cause attribution for the n=200 task sweep.

For each (problem, ordering, emotion) cell, compare the steered transcript
against the same-ordering control transcript and assign a high-level cause
label that explains what the emotion did to the joint reasoning.

Inputs:
  analysis/snapshot_alice_n200.json
  analysis/snapshot_bob_n200.json
  analysis/snapshot_problems_n200.json

Output:
  analysis/causes_n200.json   — list of records, one per (problem, ordering, emotion) cell

Cause taxonomy (heuristic, applied in priority order):
  F. magnitude_error      pred / gold off by >= 10x (decimal-shift / unit slip)
  I. extraction_artifact  gold appears in transcript but extracted predicted differs
  E. early_termination    emotion ended >=2 turns earlier than control AND emotion wrong
  D. disagreement_loop    emotion hit turn cap AND >=6 distinct numeric proposals
  A. decomposition_shift  first-turn Jaccard similarity < 0.30 (emotion opened differently)
  B. arithmetic_divergence first-turn Jaccard >= 0.55 AND different final answer
  C. verification_bypass  helped/hurt + emotion shorter than control + agreement-word signal
  G. semantic_reframing   moderate similarity (0.30-0.55), different final answer, no other tag fits
  H. stylistic_only       outcome unchanged + same predicted (or both none)
  J. other                catch-all
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANA = REPO / "analysis"
sys.path.insert(0, str(REPO / "scripts"))
from analyze_cases import classify_failure  # noqa: E402
from fix_extraction import improved_extract  # noqa: E402

CONDITIONS = ["control", "joy+", "joy-", "sadness+", "anger+", "curiosity+", "surprise+"]
EMOTIONS_NON_CTRL = [c for c in CONDITIONS if c != "control"]
STEERED_TRAIT = {
    "joy+": ("joy", +1),
    "joy-": ("joy", -1),
    "sadness+": ("sadness", +1),
    "anger+": ("anger", +1),
    "curiosity+": ("curiosity", +1),
    "surprise+": ("surprise", +1),
}

AGREEMENT_PAT = re.compile(
    r"\b(agreed?|correct|exactly|yes,|that'?s right|you'?re right|sounds right|I confirm)\b",
    re.IGNORECASE,
)
DISAGREEMENT_PAT = re.compile(
    r"\b(wait|actually|no,|I disagree|I don'?t think|let me reconsider|hold on|hmm)\b",
    re.IGNORECASE,
)
TOKEN_PAT = re.compile(r"[A-Za-z0-9]+")
NUM_PAT = re.compile(r"-?\d+(?:\.\d+)?")


def first_speaker_turn(transcript: list[str], speaker: str) -> str:
    for line in transcript:
        m = re.match(r"\[\d+\]\s+(\w+):\s*(.*)", line)
        if m and m.group(1).lower() == speaker.lower():
            return m.group(2)
    return ""


def words_per_speaker(transcript: list[str]) -> dict[str, int]:
    """Total word count per speaker across the whole transcript.

    Words are whitespace-separated tokens after stripping the [NN] speaker:
    prefix. Counts only labelled lines like "[01] alice: ...".
    """
    counts: dict[str, int] = {"alice": 0, "bob": 0, "other": 0}
    for line in transcript:
        m = re.match(r"\[\d+\]\s+(\w+):\s*(.*)", line)
        if not m:
            continue
        spk = m.group(1).lower()
        body = m.group(2)
        wc = len(body.split())
        if spk in counts:
            counts[spk] += wc
        else:
            counts["other"] += wc
    return counts


def jaccard(a: str, b: str) -> float:
    ta = set(TOKEN_PAT.findall(a.lower()))
    tb = set(TOKEN_PAT.findall(b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def magnitude_ratio(pred, gold) -> float | None:
    """Return |pred / gold| if both numeric and gold non-zero."""
    try:
        p = float(pred)
        g = float(gold)
    except (TypeError, ValueError):
        return None
    if g == 0:
        return None
    return abs(p / g)


def gold_in_last_turns(transcript: list[str], gold, n_last: int = 2) -> bool:
    """Did the gold number appear in the last `n_last` agent turns?

    We restrict to the tail because that is where the answer should be
    asserted; gold mentioned mid-reasoning and then abandoned does not
    constitute an extraction artifact.
    """
    if gold is None or not transcript:
        return False
    turn_lines = [ln for ln in transcript if re.match(r"\[\d+\]", ln)]
    tail = "\n".join(turn_lines[-n_last:]).replace(",", "")
    try:
        gf = float(gold)
    except (TypeError, ValueError):
        return False
    g_int = int(gf) if gf == int(gf) else None
    if g_int is not None:
        return bool(re.search(rf"\b{g_int}(?:\.\d+)?\b", tail))
    return f"{gf}" in tail


def distinct_big_numbers(transcript: list[str]) -> int:
    text = "\n".join(transcript).replace(",", "")
    nums = NUM_PAT.findall(text)
    return len({n for n in nums if abs(float(n)) > 1})


def cause_label(features: dict) -> str:
    """Apply heuristic decision tree to feature dict."""
    out = features["outcome_change"]  # one of: stay_correct, stay_wrong, helped, hurt
    if out == "stay_correct" and features["same_predicted"]:
        return "H_stylistic_only"
    if out == "stay_wrong" and features["same_predicted"]:
        return "H_stylistic_only"

    # Priority-ordered diagnostics for cells where something changed.
    mag = features["mag_ratio"]
    if mag is not None and (mag >= 9.5 or mag <= 1 / 9.5) and not features["emotion_correct"]:
        return "F_magnitude_error"

    # Extraction artifact = (a) gold IS asserted in the last 2 turns,
    # (b) the agent did NOT conclude clearly (extractor had to fall back to
    # last-number-in-tail because no "Final answer:", "\boxed{}", or
    # "the answer is X" was present), and (c) the predicted answer differs
    # from gold. Cases where the agent did write "Final answer: X" with X !=
    # gold are NOT extraction artifacts — the agent reached a wrong conclusion.
    if (
        features["gold_in_last_turns"]
        and not features["emotion_correct"]
        and features["emotion_extract_strategy"] in ("fallback_last300", "no_match")
    ):
        return "I_extraction_artifact"

    if features["emotion_n_turns_minus_control"] <= -2 and not features["emotion_correct"]:
        return "E_early_termination"

    if features["emotion_n_turns"] >= 10 and features["distinct_big_nums"] >= 6:
        return "D_disagreement_loop"

    if features["first_turn_jaccard"] < 0.30:
        return "A_decomposition_shift"

    if features["first_turn_jaccard"] >= 0.55 and not features["same_predicted"]:
        return "B_arithmetic_divergence"

    if (
        out in ("helped", "hurt")
        and features["emotion_n_turns_minus_control"] < 0
        and features["agreement_words_emotion"] > features["agreement_words_control"]
    ):
        return "C_verification_bypass"

    if 0.30 <= features["first_turn_jaccard"] < 0.55 and not features["same_predicted"]:
        return "G_semantic_reframing"

    if out in ("helped", "hurt"):
        return "J_other"
    return "H_stylistic_only"


def features_for_cell(
    problem: dict, ctrl: dict, emo: dict, condition: str
) -> dict:
    gold = problem.get("gold")
    ctrl_pred = ctrl.get("predicted")
    emo_pred = emo.get("predicted")
    ctrl_correct = bool(ctrl.get("correct"))
    emo_correct = bool(emo.get("correct"))

    if ctrl_correct and emo_correct:
        outcome = "stay_correct"
    elif (not ctrl_correct) and (not emo_correct):
        outcome = "stay_wrong"
    elif (not ctrl_correct) and emo_correct:
        outcome = "helped"
    else:
        outcome = "hurt"

    # Always compare *Alice's* first turn — she is the steered agent, so any
    # decomposition / arithmetic / framing change must originate there.
    # In bob-first runs this is turn [02]; in alice-first it is turn [01].
    ctrl_first = first_speaker_turn(ctrl["transcript"], "alice")
    emo_first = first_speaker_turn(emo["transcript"], "alice")
    j = jaccard(ctrl_first, emo_first)

    ctrl_text = "\n".join(ctrl.get("transcript", []))
    emo_text = "\n".join(emo.get("transcript", []))

    trait, sign = STEERED_TRAIT.get(condition, ("joy", +1))
    a_traj = emo.get("trajectories", {}).get("alice", {}).get(trait, [])
    b_traj = emo.get("trajectories", {}).get("bob", {}).get(trait, [])
    a_traj_ctrl = ctrl.get("trajectories", {}).get("alice", {}).get(trait, [])
    a_mean = sum(a_traj) / len(a_traj) if a_traj else 0.0
    a_mean_ctrl = sum(a_traj_ctrl) / len(a_traj_ctrl) if a_traj_ctrl else 0.0
    b_drift = (b_traj[-1] - b_traj[0]) if len(b_traj) >= 2 else 0.0

    same_predicted = (
        ctrl_pred == emo_pred
        if (ctrl_pred is not None and emo_pred is not None)
        else (ctrl_pred is None and emo_pred is None)
    )

    # Re-extract via the strong extractor so we know which strategy fired —
    # used to distinguish genuine extraction artifacts from agent-was-wrong cases.
    _, emo_strategy = improved_extract(emo_text)
    _, ctrl_strategy = improved_extract(ctrl_text)

    # Conversation stats: word counts per agent role.
    # alice = emotional agent (steered), bob = stable agent.
    ctrl_words = words_per_speaker(ctrl.get("transcript", []))
    emo_words = words_per_speaker(emo.get("transcript", []))

    return {
        "outcome_change": outcome,
        "control_correct": ctrl_correct,
        "emotion_correct": emo_correct,
        "control_predicted": ctrl_pred,
        "emotion_predicted": emo_pred,
        "same_predicted": same_predicted,
        "control_n_turns": ctrl.get("n_turns", 0),
        "emotion_n_turns": emo.get("n_turns", 0),
        "emotion_n_turns_minus_control": (emo.get("n_turns", 0) - ctrl.get("n_turns", 0)),
        # Conversation word counts per role; alice = emotional, bob = stable.
        "control_words_emotional": ctrl_words["alice"],
        "control_words_stable": ctrl_words["bob"],
        "emotion_words_emotional": emo_words["alice"],
        "emotion_words_stable": emo_words["bob"],
        "delta_words_emotional": emo_words["alice"] - ctrl_words["alice"],
        "delta_words_stable": emo_words["bob"] - ctrl_words["bob"],
        "first_turn_jaccard": round(j, 3),
        "mag_ratio": magnitude_ratio(emo_pred, gold),
        "distinct_big_nums": distinct_big_numbers(emo.get("transcript", [])),
        "agreement_words_emotion": len(AGREEMENT_PAT.findall(emo_text)),
        "agreement_words_control": len(AGREEMENT_PAT.findall(ctrl_text)),
        "disagreement_words_emotion": len(DISAGREEMENT_PAT.findall(emo_text)),
        "gold_in_last_turns": gold_in_last_turns(emo.get("transcript", []), gold),
        "emotion_extract_strategy": emo_strategy,
        "control_extract_strategy": ctrl_strategy,
        "alice_trait_mean_emotion": round(a_mean, 4),
        "alice_trait_mean_control": round(a_mean_ctrl, 4),
        "bob_trait_drift_emotion": round(b_drift, 4),
        "control_failure": classify_failure(ctrl, gold),
        "emotion_failure": classify_failure(emo, gold),
    }


def _load_results(stem: str) -> dict:
    """Prefer the post-fix results file; fall back to the raw snapshot."""
    fixed = ANA / f"results_{stem}_fixed.json"
    raw = ANA / f"snapshot_{stem}_n200.json"
    p = fixed if fixed.exists() else raw
    print(f"  using {p.name}")
    return json.loads(p.read_text())


def build():
    print("loading data:")
    alice = _load_results("alice")
    bob = _load_results("bob")
    problems = json.loads((ANA / "snapshot_problems_n200.json").read_text())
    n = len(problems)
    print(f"loaded n={n} problems")

    records = []
    for ordering, src in (("alice", alice), ("bob", bob)):
        ctrl_rs = src.get("control", [])
        for cond in EMOTIONS_NON_CTRL:
            emo_rs = src.get(cond, [])
            for i in range(min(n, len(ctrl_rs), len(emo_rs))):
                feats = features_for_cell(problems[i], ctrl_rs[i], emo_rs[i], cond)
                cause = cause_label(feats)
                rec = {
                    "idx": problems[i]["idx"],
                    "ordering": ordering,
                    "emotion": cond,
                    "cause": cause,
                    **feats,
                    "question": problems[i]["question"].splitlines()[0][:200],
                    "gold": problems[i]["gold"],
                }
                records.append(rec)

    out = ANA / "causes_n200.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"wrote {out}  ({len(records)} cells)")

    # Quick summary
    from collections import Counter
    by_cell = Counter()
    by_outcome = Counter()
    by_oxc = Counter()  # (ordering, emotion, cause)
    for r in records:
        by_cell[r["cause"]] += 1
        by_outcome[r["outcome_change"]] += 1
        by_oxc[(r["ordering"], r["emotion"], r["cause"])] += 1

    print("\n=== Outcome distribution ===")
    for k, v in sorted(by_outcome.items()):
        print(f"  {k:14s}: {v}")
    print("\n=== Cause distribution (all 2400 cells) ===")
    for k, v in sorted(by_cell.items(), key=lambda kv: -kv[1]):
        print(f"  {k:30s}: {v}")
    print("\n=== Per (ordering, emotion) cause distribution ===")
    for ordering in ("alice", "bob"):
        for cond in EMOTIONS_NON_CTRL:
            cell_total = sum(v for (o, c, _), v in by_oxc.items() if o == ordering and c == cond)
            print(f"\n  [{ordering}-first / {cond}] (n={cell_total})")
            sub = [(cause, v) for (o, c, cause), v in by_oxc.items() if o == ordering and c == cond]
            for cause, v in sorted(sub, key=lambda kv: -kv[1]):
                print(f"    {cause:30s}: {v}")


if __name__ == "__main__":
    build()
