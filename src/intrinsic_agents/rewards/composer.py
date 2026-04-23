"""Composes per-step trait projections into a scalar reward.

    R_int   = sum_i  w_i * aggregate_t( s_i(t) )
    R_total = alpha * R_int + beta * R_ext

Aggregators:
    mean            — average projection across rollout steps
    discounted_sum  — gamma-discounted sum, favors early alignment
    terminal        — projection at the final step only
    max             — peak projection during the rollout

`discounted_sum` is the default because it matches the thesis framing: the
agent is rewarded for *sustained* movement toward `v*` over its trajectory,
not just a one-shot terminal match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import yaml

Aggregator = Literal["mean", "discounted_sum", "terminal", "max"]


@dataclass
class RewardConfig:
    weights: dict[str, float] = field(default_factory=dict)  # trait -> w_i
    aggregator: Aggregator = "discounted_sum"
    gamma: float = 0.95
    alpha: float = 1.0   # intrinsic coefficient
    beta: float = 0.0    # external coefficient

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RewardConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    @classmethod
    def from_preset(cls, preset_name: str, presets_dir: str | Path) -> "RewardConfig":
        path = Path(presets_dir) / f"{preset_name}.yaml"
        return cls.from_yaml(path)


def _aggregate(traj: torch.Tensor, how: Aggregator, gamma: float) -> float:
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
            total += w * _aggregate(traj, self.config.aggregator, self.config.gamma)
        return total

    def compose(
        self,
        trait_trajectories: dict[str, torch.Tensor],
        external_reward: float = 0.0,
    ) -> dict[str, float]:
        """Returns a breakdown dict so every component is logged, not hidden."""
        r_int = self.intrinsic(trait_trajectories)
        r_total = self.config.alpha * r_int + self.config.beta * external_reward
        return {
            "R_int": r_int,
            "R_ext": external_reward,
            "R_total": r_total,
            "alpha": self.config.alpha,
            "beta": self.config.beta,
        }
