"""Continuous (non-episodic) RL loop with rolling-buffer policy-gradient updates.

Contract
--------
The agent lives inside a scenario that never terminates. Every K turns we do a
policy update on the most recent W turns in a rolling buffer, then keep rolling.
No episode boundary, no full-trajectory aggregation.

Why this exists
---------------
Thesis: reward is a match between envisioned future and actual future. Under
that view there is no "end of task" — the agent is continuously predicting and
continuously being rewarded on how well its projections drift toward `v*`. An
episodic trainer (GRPO-on-fixed-rollouts) can't capture that; it collapses the
whole rollout into one scalar and throws away the temporal structure.

Safety rails (all configurable in RewardConfig)
-----------------------------------------------
- KL anchor to a frozen copy of the base model: prevents the agent from
  drifting into incoherent token spam that happens to max `joy`.
- Reward clipping: caps R_int so a single trait can't dominate forever.
- Entropy bonus: maintains exploration, resists mode collapse.
- Periodic vector re-extraction: the trait vectors were extracted from the
  base model; as the policy drifts, the vectors drift too. Re-extract every
  `refresh_vectors_every` updates to keep them pointed at the right
  subspace — otherwise the agent is optimizing a stale target.

TODO (this file)
----------------
- Plug in a real policy-gradient update (PPO on log-probs of the generated
  tokens, or KTO-style preference updates using the per-step reward as
  implicit preference signal).
- Gradient accumulation across the rolling buffer.
- Checkpoint every N updates; keep last 3 checkpoints for rollback on collapse.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from ..agents.llm_agent import LLMAgent
from ..monitor.logger import JsonlLogger
from ..rewards.composer import RewardConfig, StreamingRewardComposer
from .rollout import TurnRecord, stream_rollout


@dataclass
class OnlineConfig:
    update_every: int = 16          # policy update every K turns
    buffer_size: int = 128          # rolling buffer of most-recent turns
    refresh_vectors_every: int = 1024  # re-extract persona vectors every N updates
    checkpoint_every: int = 256
    max_updates: int | None = None  # None = forever; caller sends SIGINT to stop
    kl_divergence_alarm: float = 10.0  # stop if KL from base > this
    trait_drift_alarm: float = 0.95    # stop if |<h_mean, v_i>| saturates near ceiling


def _reduce_turn(turn: TurnRecord) -> dict[str, float]:
    """Collapse a single turn's per-token trajectory into per-trait scalars."""
    out: dict[str, float] = {}
    for trait, traj in turn.trait_trajectory.items():
        out[trait] = float(traj.mean().item()) if traj.numel() else 0.0
    return out


def run_continuous(
    agents: list[LLMAgent],
    scenario_prompt: str,
    reward_cfg: RewardConfig,
    online_cfg: OnlineConfig,
    logger: JsonlLogger | None = None,
) -> None:
    """Drive a never-ending rollout with periodic rolling-buffer updates.

    The policy-gradient step is currently a TODO stub — this function owns the
    loop shape (stream → per-step reward → rolling buffer → trigger update →
    safety checks) so the missing piece is well-contained.
    """
    composers = {a.name: StreamingRewardComposer(reward_cfg) for a in agents}
    buffers: dict[str, deque[dict]] = {
        a.name: deque(maxlen=online_cfg.buffer_size) for a in agents
    }

    updates = 0
    stream = stream_rollout(agents, scenario_prompt)

    for step, turn in enumerate(stream):
        projections = _reduce_turn(turn)
        reward = composers[turn.agent_id].step(projections)

        buffers[turn.agent_id].append(
            {"turn": turn, "projections": projections, "reward": reward}
        )

        if logger is not None:
            logger.log(
                {
                    "step": step,
                    "update": updates,
                    "agent": turn.agent_id,
                    "response": turn.response,
                    **{f"s_{k}": v for k, v in projections.items()},
                    **reward,
                }
            )

        if (step + 1) % online_cfg.update_every == 0:
            _policy_update(agents, buffers, reward_cfg, online_cfg)
            updates += 1

            if _saturation_check(projections, online_cfg):
                print(f"[online] trait drift alarm at update {updates}; halting")
                return

            if online_cfg.max_updates and updates >= online_cfg.max_updates:
                return

            if updates % online_cfg.refresh_vectors_every == 0:
                _refresh_vectors(agents)

            if updates % online_cfg.checkpoint_every == 0:
                _checkpoint(agents, updates)


def _policy_update(
    agents: list[LLMAgent],
    buffers: dict[str, deque[dict]],
    reward_cfg: RewardConfig,
    online_cfg: OnlineConfig,
) -> None:
    """TODO: per-token log-prob + reward -> PPO-style policy gradient update,
    plus KL penalty against a frozen base copy weighted by `reward_cfg.kl_coef`.
    For now this is a no-op so the rest of the loop is exercisable."""
    _ = (agents, buffers, reward_cfg, online_cfg)


def _refresh_vectors(agents: list[LLMAgent]) -> None:
    """TODO: re-extract persona vectors from the *current* policy and swap the
    probe's bank. Without this, trait vectors drift out of alignment with the
    model's representation geometry as training proceeds."""
    _ = agents


def _saturation_check(projections: dict[str, float], cfg: OnlineConfig) -> bool:
    return any(abs(v) > cfg.trait_drift_alarm for v in projections.values())


def _checkpoint(agents: list[LLMAgent], step: int) -> None:
    """TODO: save LoRA adapters. Keep only the last 3 checkpoints to bound disk."""
    _ = (agents, step)
