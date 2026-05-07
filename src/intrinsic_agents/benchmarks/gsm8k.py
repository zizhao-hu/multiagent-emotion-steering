"""GSM8K loader + scorer.

GSM8K (Cobbe et al., 2021) is a grade-school math word problems dataset
where the model must produce a numeric answer through chain-of-thought.
Each example's gold answer is appended as `#### N` at the end of the
reasoning trace. We extract that number and grade an open-ended completion
by extracting *its* final number and comparing for numeric equality.

Dataset: `openai/gsm8k`, config `main`, splits {train, test}.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from datasets import load_dataset

GSM8K_PROMPT = (
    "Solve the following math word problem. Show your reasoning step by step, "
    "then state your final numeric answer on a new line in the form "
    "'#### N' where N is the answer.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# Capture the gold answer that the dataset ships in the form "#### 72".
_GOLD_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
# Greedy: pull every signed/decimal number out of a completion. We score
# against the LAST number, since chain-of-thought may surface intermediate
# values before the final answer.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class GSM8KTask:
    qid: str
    prompt: str
    correct_answer: str   # canonical numeric string from "#### N"
    gold_raw: str         # full reasoning + answer, for debugging


def _extract_gold(raw_answer: str) -> str:
    """Pull the numeric suffix from a GSM8K answer string."""
    m = _GOLD_RE.search(raw_answer)
    if m is None:
        # Defensive fallback: take the last number in the trace.
        nums = _NUMBER_RE.findall(raw_answer)
        return nums[-1].replace(",", "") if nums else ""
    return m.group(1).strip().replace(",", "")


def _to_float(s: str) -> float | None:
    if not s:
        return None
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def load_gsm8k(
    n: int | None = None,
    seed: int = 0,
    split: str = "test",
    config: str = "main",
) -> list[GSM8KTask]:
    """Return up to `n` GSM8K tasks, deterministically shuffled by `seed`."""
    ds = load_dataset("openai/gsm8k", config, split=split)
    rng = random.Random(seed)

    tasks: list[GSM8KTask] = []
    for i, row in enumerate(ds):
        gold = _extract_gold(row["answer"])
        if not gold:
            continue
        prompt = GSM8K_PROMPT.format(question=row["question"].strip())
        tasks.append(GSM8KTask(
            qid=f"gsm8k_{split}_{i}",
            prompt=prompt,
            correct_answer=gold,
            gold_raw=row["answer"],
        ))

    rng.shuffle(tasks)
    if n is not None:
        tasks = tasks[:n]
    return tasks


def score_gsm8k(task: GSM8KTask, completion: str) -> float:
    """1.0 iff the completion's final extracted number equals the gold.

    Strategy: prefer a `#### N` marker if the model emitted one; otherwise
    fall back to the last number in the completion. We compare numerically
    (strip commas, parse to float) to absorb formatting noise like trailing
    decimals or thousands separators.
    """
    m = _GOLD_RE.search(completion)
    if m is not None:
        pred = m.group(1).strip().replace(",", "")
    else:
        nums = _NUMBER_RE.findall(completion)
        if not nums:
            return 0.0
        pred = nums[-1].replace(",", "")

    pred_v = _to_float(pred)
    gold_v = _to_float(task.correct_answer)
    if pred_v is None or gold_v is None:
        return 1.0 if pred.strip() == task.correct_answer.strip() else 0.0
    return 1.0 if abs(pred_v - gold_v) < 1e-6 else 0.0
