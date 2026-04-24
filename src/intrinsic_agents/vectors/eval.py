"""Replication-gate evaluations.

Gate 1 — Extraction AUC: leave-one-out. For each trait with K contrastive
prompt pairs, we hold out one pair at a time, extract the vector from the
remaining K-1 pairs, and score the held-out pos/neg by projection. AUC over
all K held-out scores is the extraction quality.

A vector that generalizes beyond the prompts it was extracted from will have
AUC close to 1.0. A vector that just memorizes its own prompt set will have
near-0.5 AUC on held-out pairs.

Gate 2 (steering) lives in `steering.py`. Gate 3 (probing on free-form
responses) needs a judge and is separate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .extract import TraitSpec, _last_token_hidden


def auc_from_scores(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Rank-based AUC: P(score_pos > score_neg), ties counted as 0.5."""
    if not pos_scores or not neg_scores:
        return float("nan")
    total, wins = 0, 0.0
    for p in pos_scores:
        for n in neg_scores:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / total


@dataclass
class ExtractionAUC:
    trait: str
    auc: float
    n_pairs: int
    per_pair_pos: list[float]
    per_pair_neg: list[float]

    def passes(self, target: float = 0.85) -> bool:
        return self.auc >= target


def extraction_auc_loo(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    trait: TraitSpec,
    layer: int,
    device: str = "cpu",
) -> ExtractionAUC:
    """Leave-one-out extraction AUC for a single trait.

    For each held-out pair i:
        v_{-i} = normalize( mean_{j != i} h_pos_j − mean_{j != i} h_neg_j )
        pos_score_i = <h_pos_i, v_{-i}>
        neg_score_i = <h_neg_i, v_{-i}>
    Returns AUC computed over {pos_score_i}, {neg_score_i}.
    """
    # Cache all activations once; then the LOO loop is just matrix ops.
    pos_acts = torch.stack(
        [_last_token_hidden(model, tokenizer, p["pos"], layer, device) for p in trait.pairs]
    )
    neg_acts = torch.stack(
        [_last_token_hidden(model, tokenizer, p["neg"], layer, device) for p in trait.pairs]
    )
    K = pos_acts.shape[0]

    pos_scores, neg_scores = [], []
    for i in range(K):
        mask = [j for j in range(K) if j != i]
        pos_mean = pos_acts[mask].mean(0)
        neg_mean = neg_acts[mask].mean(0)
        diff = pos_mean - neg_mean
        v = diff / (diff.norm() + 1e-8)
        pos_scores.append(float(torch.dot(pos_acts[i], v).item()))
        neg_scores.append(float(torch.dot(neg_acts[i], v).item()))

    return ExtractionAUC(
        trait=trait.name,
        auc=auc_from_scores(pos_scores, neg_scores),
        n_pairs=K,
        per_pair_pos=pos_scores,
        per_pair_neg=neg_scores,
    )


def extraction_auc_all(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    traits: dict[str, TraitSpec],
    layer: int,
    device: str = "cpu",
) -> dict[str, ExtractionAUC]:
    return {
        name: extraction_auc_loo(model, tokenizer, spec, layer=layer, device=device)
        for name, spec in traits.items()
    }
