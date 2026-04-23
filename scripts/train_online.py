"""Entry point for the continuous-learning regime.

Streams turns forever, updates the policy on a rolling buffer every K turns,
and logs per-step reward components so the long trajectory is inspectable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from intrinsic_agents.agents.llm_agent import LLMAgent
from intrinsic_agents.monitor.logger import JsonlLogger
from intrinsic_agents.rewards.composer import RewardConfig
from intrinsic_agents.train.online import OnlineConfig, run_continuous

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=str(REPO_ROOT / "runs"))
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_name = Path(args.config).parent.name
    out_dir = Path(args.out) / run_name

    reward_cfg = RewardConfig(**cfg["reward"])
    online_cfg = OnlineConfig(**cfg.get("online", {}))

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

    with JsonlLogger(out_dir / "stream.jsonl") as log:
        run_continuous(
            agents=agents,
            scenario_prompt=cfg["scenario"]["prompt"],
            reward_cfg=reward_cfg,
            online_cfg=online_cfg,
            logger=log,
        )


if __name__ == "__main__":
    main()
