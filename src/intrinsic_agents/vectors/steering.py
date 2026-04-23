"""Steering evaluation — the Anthropic Persona Vectors replication metric.

A persona vector is only meaningful if *adding* it to the residual stream at
test time actually makes the model express the trait more. This is the core
causal check from the paper: the vector has to be a handle on the behavior,
not just a correlation.

Usage:
    harness = SteeringHarness(model, tok, layer=14)
    result = harness.compare(
        vector=v_honesty,
        prompts=eval_prompts,
        alphas=[-4, -2, 0, 2, 4],
        judge=lambda text: score_honesty(text),  # e.g. Claude-as-judge
    )
    # result is a dict: alpha -> mean judge score. Expected monotonic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import _layer_module


@dataclass
class SteeringResult:
    alphas: list[float]
    mean_scores: list[float]
    per_prompt_scores: dict[float, list[float]] = field(default_factory=dict)
    samples: dict[float, list[str]] = field(default_factory=dict)

    def monotonic(self) -> bool:
        """True if judge score trends with alpha (loose check: corr > 0.5)."""
        if len(self.alphas) < 3:
            return False
        a = torch.tensor(self.alphas, dtype=torch.float32)
        s = torch.tensor(self.mean_scores, dtype=torch.float32)
        corr = ((a - a.mean()) * (s - s.mean())).sum() / (
            a.std(unbiased=False) * s.std(unbiased=False) * len(a) + 1e-8
        )
        return float(corr.item()) > 0.5


class SteeringHarness:
    """Adds alpha*v to the residual stream at layer L during generation."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layer: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self._handle = None

    def _install(self, vector: torch.Tensor, alpha: float) -> None:
        if self._handle is not None:
            self.remove()
        v = (alpha * vector).to(
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
        )
        block = _layer_module(self.model, self.layer)

        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                return (out[0] + v,) + out[1:]
            return out + v

        self._handle = block.register_forward_hook(hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(
            out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def compare(
        self,
        vector: torch.Tensor,
        prompts: list[str],
        alphas: list[float],
        judge: Callable[[str], float],
        max_new_tokens: int = 128,
    ) -> SteeringResult:
        """Sweep alpha, generate N responses per alpha, run judge.

        Judge is a callable returning a scalar trait score given completion text.
        In practice: Claude-as-judge with a rubric, or a log-likelihood ratio
        under two prompt templates, etc.
        """
        result = SteeringResult(alphas=list(alphas), mean_scores=[])
        for alpha in alphas:
            self._install(vector, alpha)
            scores, samples = [], []
            for p in prompts:
                text = self.generate(p, max_new_tokens=max_new_tokens)
                scores.append(judge(text))
                samples.append(text)
            self.remove()
            result.per_prompt_scores[alpha] = scores
            result.samples[alpha] = samples
            result.mean_scores.append(sum(scores) / max(len(scores), 1))
        return result
