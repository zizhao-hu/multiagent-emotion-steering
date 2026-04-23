"""GRPO training loop.

TODO: wire up TRL's GRPOTrainer once the rollout+reward path is verified on a
tiny model. The reward function below is the integration point — it takes a
list of generated completions and must return a list of floats.

For now this module defines the reward-function adapter so downstream code can
import a stable symbol.
"""

from __future__ import annotations

from collections.abc import Callable

from ..rewards.composer import RewardComposer
from .rollout import RolloutRecord, trajectories_for


def make_reward_fn(
    composer: RewardComposer,
    agent_id: str,
) -> Callable[[RolloutRecord], dict[str, float]]:
    """Adapter: rollout record -> reward breakdown dict for a single agent."""

    def reward_fn(rec: RolloutRecord) -> dict[str, float]:
        traj = trajectories_for(rec, agent_id)
        r_ext = rec.external_reward.get(agent_id, 0.0)
        return composer.compose(traj, external_reward=r_ext)

    return reward_fn
