"""GPQA-Diamond loader + scorer.

GPQA (Rein et al., 2023) is a graduate-level science QA set in chemistry,
biology, and physics. The "Diamond" subset (~198 questions) is the
hardest-and-cleanest split that experts roughly agree on. Multi-choice with
four options.

Score: letter match. We extract the first A/B/C/D the model emits after the
final "Answer:" token; if none, the response counts as 0.

Dataset source: `Idavidrein/gpqa`, config `gpqa_diamond`.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from datasets import load_dataset

GPQA_PROMPT = (
    "Answer the following graduate-level science multiple-choice question. "
    "Reason briefly, then end your response with a single line of the form "
    "'Answer: X' where X is one of A, B, C, or D.\n\n"
    "Question: {question}\n\n"
    "{choices}\n\n"
    "Answer:"
)


@dataclass
class GPQATask:
    qid: str
    prompt: str
    correct_letter: str
    domain: str


def load_gpqa_diamond(
    split: str = "train",
    n: int | None = None,
    seed: int = 0,
) -> list[GPQATask]:
    """Return up to `n` GPQA-Diamond tasks, deterministically shuffled by `seed`.

    The Diamond subset is small (~198 examples) so we use the train split and
    sample without replacement.
    """
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split)
    rng = random.Random(seed)

    tasks: list[GPQATask] = []
    for i, row in enumerate(ds):
        # Choices in GPQA are stored as 4 separate columns: Correct Answer +
        # Incorrect Answer 1/2/3. Shuffle them per-row with a deterministic seed.
        opts = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        order = list(range(4))
        rng_row = random.Random(seed + i)
        rng_row.shuffle(order)
        letters = ["A", "B", "C", "D"]
        choice_lines = []
        correct_letter = ""
        for letter, src_idx in zip(letters, order):
            choice_lines.append(f"{letter}. {opts[src_idx]}")
            if src_idx == 0:
                correct_letter = letter

        prompt = GPQA_PROMPT.format(
            question=row["Question"].strip(),
            choices="\n".join(choice_lines),
        )
        tasks.append(GPQATask(
            qid=str(i),
            prompt=prompt,
            correct_letter=correct_letter,
            domain=row.get("High-level domain", "unknown"),
        ))

    if n is not None:
        rng.shuffle(tasks)
        tasks = tasks[:n]
    return tasks


_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\(?([ABCD])\)?", re.IGNORECASE)


def score_gpqa(task: GPQATask, completion: str) -> float:
    """1.0 if the first parsed letter matches the gold letter, else 0.0.

    Looks for 'Answer: X' first; falls back to the first standalone A-D it
    finds. This is the simplest robust rubric for short MC completions.
    """
    m = _ANSWER_RE.search(completion)
    if m is None:
        # Fallback: first standalone capital A-D in the first 200 chars.
        for ch in completion[:200]:
            if ch in "ABCD":
                return 1.0 if ch == task.correct_letter else 0.0
        return 0.0
    return 1.0 if m.group(1).upper() == task.correct_letter else 0.0
