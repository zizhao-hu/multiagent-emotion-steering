"""MMLU-Pro loader + scorer.

MMLU-Pro (Wang et al., 2024, TIGER-Lab/MMLU-Pro) is a harder, 10-choice
re-do of MMLU with reasoning-centric questions. We filter to STEM
categories so this serves as the "advanced science" benchmark in steering
sweeps. Ungated and freely available, unlike GPQA.

Score: letter match. We extract the first A-J the model emits after a
final 'Answer:' marker; if missing, fall back to the first standalone A-J
in the first 200 chars.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass

from datasets import load_dataset

# Default to STEM-heavy categories where steering on emotion vectors might
# plausibly affect reasoning vs. recall.
STEM_CATEGORIES = {
    "physics",
    "chemistry",
    "biology",
    "computer science",
    "math",
    "engineering",
    "health",
}

MMLU_PRO_PROMPT = (
    "Answer the following multiple-choice question. "
    "Reason briefly, then end with a line of the form 'Answer: X' where X "
    "is the letter of the correct option.\n\n"
    "Question: {question}\n\n"
    "{choices}\n\n"
    "Answer:"
)


@dataclass
class MMLUProTask:
    qid: str
    prompt: str
    correct_letter: str
    category: str


def load_mmlu_pro(
    n: int | None = None,
    seed: int = 0,
    categories: set[str] | None = None,
    split: str = "test",
) -> list[MMLUProTask]:
    """Return up to `n` MMLU-Pro tasks, optionally restricted to `categories`.

    The dataset already ships with shuffled options and an 'answer_index'
    column. We don't reshuffle — we just preserve the dataset's letter
    assignment so the gold answer matches.
    """
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    cats = categories if categories is not None else STEM_CATEGORIES

    tasks: list[MMLUProTask] = []
    for row in ds:
        if cats and row["category"] not in cats:
            continue
        opts = row["options"]
        # MMLU-Pro options are up to 10 long; pad letters A..J accordingly.
        letters = list(string.ascii_uppercase[: len(opts)])
        choice_lines = [f"{lt}. {opt}" for lt, opt in zip(letters, opts)]
        correct_letter = letters[row["answer_index"]]
        prompt = MMLU_PRO_PROMPT.format(
            question=row["question"].strip(),
            choices="\n".join(choice_lines),
        )
        tasks.append(MMLUProTask(
            qid=str(row["question_id"]),
            prompt=prompt,
            correct_letter=correct_letter,
            category=row["category"],
        ))

    rng = random.Random(seed)
    rng.shuffle(tasks)
    if n is not None:
        tasks = tasks[:n]
    return tasks


_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\(?([A-J])\)?", re.IGNORECASE)


def score_mmlu_pro(task: MMLUProTask, completion: str) -> float:
    m = _ANSWER_RE.search(completion)
    if m is None:
        for ch in completion[:200]:
            if ch in "ABCDEFGHIJ":
                return 1.0 if ch == task.correct_letter else 0.0
        return 0.0
    return 1.0 if m.group(1).upper() == task.correct_letter else 0.0
