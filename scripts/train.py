"""Entry point for a finetuning run.

TODO: full GRPO loop. This file currently loads the experiment config, builds
the reward composer, instantiates agents with probes attached, runs one rollout
to verify plumbing, and logs the reward breakdown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from intrinsic_agents.agents.llm_agent import LLMAgent
from intrinsic_agents.monitor.logger import JsonlLogger
from intrinsic_agents.rewards.composer import RewardComposer, RewardConfig
from intrinsic_agents.train.grpo import make_reward_fn
from intrinsic_agents.train.rollout import run_rollout

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_reward_config(cfg: dict) -> RewardConfig:
    reward = cfg["reward"]
    if "preset" in reward:
        return RewardConfig.from_preset(
            reward["preset"],
            REPO_ROOT / "src" / "intrinsic_agents" / "rewards" / "presets",
        )
    return RewardConfig(**reward)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=str(REPO_ROOT / "runs"))
    args = p.parse_args()

    cfg = load_config(args.config)
    run_name = Path(args.config).stem
    out_dir = Path(args.out) / run_name

    reward_cfg = build_reward_config(cfg)
    composer = RewardComposer(reward_cfg)

    model_name = cfg["model"]["name"]
    layer = cfg["model"]["layer"]
    agents = [
        LLMAgent.load(
            name=a["id"],
            model_name=model_name,
            cache_dir=str(REPO_ROOT / "vectors" / "cache"),
            layer=layer,
        )
        for a in cfg["agents"]
    ]

    scenario_prompt = cfg["scenario"]["prompt"]
    rec = run_rollout(agents, scenario_prompt, max_turns=cfg["scenario"]["max_turns"])

    with JsonlLogger(out_dir / "rollout.jsonl") as log:
        for agent in agents:
            reward = make_reward_fn(composer, agent.name)(rec)
            log.log({"agent": agent.name, **reward})
            print(f"[{agent.name}] {reward}")


if __name__ == "__main__":
    main()
