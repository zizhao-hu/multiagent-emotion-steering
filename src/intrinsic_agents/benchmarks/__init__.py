"""Benchmark adapters for emotion-steering causal evaluations.

Each benchmark exposes:
    load(...) -> list[Task]
    score(task: Task, completion: str) -> float in [0, 1]

Where Task is a dataclass holding the input prompt, the gold answer/test, and
metadata. The driver in scripts/steer_benchmark.py loops over (trait, alpha)
pairs and runs every benchmark Task once per cell, recording pass/fail.
"""

from .gpqa import GPQATask, load_gpqa_diamond, score_gpqa
from .gsm8k import GSM8KTask, load_gsm8k, score_gsm8k
from .humaneval import HumanEvalTask, load_humaneval, score_humaneval
from .mmlu_pro import MMLUProTask, load_mmlu_pro, score_mmlu_pro

__all__ = [
    "GPQATask",
    "GSM8KTask",
    "HumanEvalTask",
    "MMLUProTask",
    "load_gpqa_diamond",
    "load_gsm8k",
    "load_humaneval",
    "load_mmlu_pro",
    "score_gpqa",
    "score_gsm8k",
    "score_humaneval",
    "score_mmlu_pro",
]
