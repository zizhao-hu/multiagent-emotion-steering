"""Multi-agent rollout collector.

Each rollout: turn-based loop where each agent responds in sequence given the
running transcript. The probe attached to each agent yields a per-turn trait
trajectory, which `rewards/composer.py` collapses into R_int.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..agents.llm_agent import LLMAgent


@dataclass
class TurnRecord:
    agent_id: str
    prompt: str
    response: str
    trait_trajectory: dict[str, torch.Tensor]


@dataclass
class RolloutRecord:
    turns: list[TurnRecord] = field(default_factory=list)
    external_reward: dict[str, float] = field(default_factory=dict)  # per-agent


def run_rollout(
    agents: list[LLMAgent],
    scenario_prompt: str,
    max_turns: int,
    max_new_tokens: int = 128,
) -> RolloutRecord:
    transcript = scenario_prompt.rstrip() + "\n"
    rec = RolloutRecord()
    for turn in range(max_turns):
        agent = agents[turn % len(agents)]
        prompt = transcript + f"\n{agent.name}:"
        text, traj = agent.respond(prompt, max_new_tokens=max_new_tokens)
        text = text.strip().split("\n")[0]
        transcript += f"\n{agent.name}: {text}"
        rec.turns.append(
            TurnRecord(agent_id=agent.name, prompt=prompt, response=text, trait_trajectory=traj)
        )
    return rec


def trajectories_for(rec: RolloutRecord, agent_id: str) -> dict[str, torch.Tensor]:
    """Concatenate per-turn trait trajectories for a single agent across a rollout."""
    per_trait: dict[str, list[torch.Tensor]] = {}
    for turn in rec.turns:
        if turn.agent_id != agent_id:
            continue
        for trait, traj in turn.trait_trajectory.items():
            per_trait.setdefault(trait, []).append(traj)
    return {
        trait: torch.cat(chunks) if chunks else torch.zeros(0)
        for trait, chunks in per_trait.items()
    }
