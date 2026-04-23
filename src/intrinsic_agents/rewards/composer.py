"""Composes per-step trait projections into a scalar reward.

    R_int   = sum_i  w_i * aggregate_t( s_i(t) )
    R_total = alpha * R_int + beta * R_ext

Aggregators:
    mean            — average projection across rollout steps
    discounted_sum  — gamma-discounted sum, favors early alignment (episodic)
    terminal        — projection at the final step only (episodic)
    max             — peak projection during the rollout
    ema             — exponential moving average; streaming-friendly (continuous)
    sliding_window  — mean over the last W steps (continuous)

`discounted_sum` is the episodic default. For the continuous-learning regime
(no terminal state, rolling-buffer updates) use `ema` or `sliding_window` —
both emit a well-defined reward at every step without waiting for the end of
an episode that never arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import yaml

Aggregator = Literal["mean", "discounted_sum", "terminal", "max", "ema", "sliding_window"]


@dataclass
class RewardConfig:
    weights: dict[str, float] = field(default_factory=dict)  # trait -> w_i
    aggregator: Aggregator = "discounted_sum"
    gamma: float = 0.95
    alpha: float = 1.0   # intrinsic coefficient
    beta: float = 0.0    # external coefficient
    # Continuous-learning params (used when aggregator is ema/sliding_window).
    ema_alpha: float = 0.1
    window: int = 64
    # Safety rails for continuous intrinsic-only training.
    kl_coef: float = 0.05              # KL penalty against frozen base model
    reward_clip: float | None = 5.0    # clip R_int to [-c, c] to prevent unbounded exploitation
    entropy_bonus: float = 0.01        # encourages exploration, resists mode collapse

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RewardConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    @classmethod
    def from_preset(cls, preset_name: str, presets_dir: str | Path) -> "RewardConfig":
        path = Path(presets_dir) / f"{preset_name}.yaml"
        return cls.from_yaml(path)


def _aggregate(
    traj: torch.Tensor,
    how: Aggregator,
    gamma: float,
    ema_alpha: float = 0.1,
    window: int = 64,
) -> float:
    if traj.numel() == 0:
        return 0.0
    if how == "mean":
        return float(traj.mean().item())
    if how == "terminal":
        return float(traj[-1].item())
    if how == "max":
        return float(traj.max().item())
    if how == "discounted_sum":
        t = torch.arange(traj.numel(), dtype=traj.dtype)
        discounts = gamma ** t
        return float((traj * discounts).sum().item())
    if how == "sliding_window":
        tail = traj[-window:]
        return float(tail.mean().item())
    if how == "ema":
        # Exponential moving average computed left-to-right. For a stateless
        # call we return the final EMA value; StreamingRewardComposer below
        # maintains the state across calls for true online use.
        v = float(traj[0].item())
        for x in traj[1:]:
            v = ema_alpha * float(x.item()) + (1 - ema_alpha) * v
        return v
    raise ValueError(f"unknown aggregator {how!r}")


class RewardComposer:
    def __init__(self, config: RewardConfig):
        self.config = config

    def intrinsic(self, trait_trajectories: dict[str, torch.Tensor]) -> float:
        total = 0.0
        for trait, w in self.config.weights.items():
            if w == 0.0:
                continue
            traj = trait_trajectories.get(trait)
            if traj is None:
                continue
            total += w * _aggregate(
                traj,
                self.config.aggregator,
                self.config.gamma,
                ema_alpha=self.config.ema_alpha,
                window=self.config.window,
            )
        return total

    def _clip(self, r: float) -> float:
        c = self.config.reward_clip
        if c is None:
            return r
        return max(-c, min(c, r))

    def compose(
        self,
        trait_trajectories: dict[str, torch.Tensor],
        external_reward: float = 0.0,
    ) -> dict[str, float]:
        """Returns a breakdown dict so every component is logged, not hidden."""
        r_int_raw = self.intrinsic(trait_trajectories)
        r_int = self._clip(r_int_raw)
        r_total = self.config.alpha * r_int + self.config.beta * external_reward
        return {
            "R_int_raw": r_int_raw,
            "R_int": r_int,
            "R_ext": external_reward,
            "R_total": r_total,
            "alpha": self.config.alpha,
            "beta": self.config.beta,
        }


class StreamingRewardComposer:
    """Per-step reward for continuous (non-episodic) training.

    Maintains per-trait EMAs so we can emit a well-defined reward *every* step
    without aggregating over an episode that doesn't terminate. Pair with
    `train/online.py`'s rolling-buffer updates.
    """

    def __init__(self, config: RewardConfig):
        self.config = config
        self._ema: dict[str, float] = {}

    def step(
        self,
        trait_projections: dict[str, float],
        external_reward: float = 0.0,
    ) -> dict[str, float]:
        a = self.config.ema_alpha
        per_trait = {}
        r_int = 0.0
        for trait, w in self.config.weights.items():
            s = trait_projections.get(trait, 0.0)
            prev = self._ema.get(trait, s)
            ema = a * s + (1 - a) * prev
            self._ema[trait] = ema
            per_trait[f"ema_{trait}"] = ema
            r_int += w * ema
        # Clipping prevents unbounded exploitation under infinite horizon.
        c = self.config.reward_clip
        if c is not None:
            r_int = max(-c, min(c, r_int))
        r_total = self.config.alpha * r_int + self.config.beta * external_reward
        return {
            "R_int": r_int,
            "R_ext": external_reward,
            "R_total": r_total,
            **per_trait,
        }

    def reset(self) -> None:
        self._ema.clear()
