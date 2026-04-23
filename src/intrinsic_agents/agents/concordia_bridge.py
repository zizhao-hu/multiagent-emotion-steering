"""Adapter that exposes an LLMAgent as Concordia's LanguageModel interface.

TODO: implement once Concordia integration is prioritized. The surface area is
small — Concordia's `language_model.LanguageModel` needs `sample_text(prompt,
**kwargs) -> str` and `sample_choice(prompt, responses) -> int`. Both can be
implemented by delegating to `LLMAgent.respond`.
"""

from __future__ import annotations

from ..agents.llm_agent import LLMAgent


class ConcordiaLLMBridge:
    def __init__(self, agent: LLMAgent):
        self.agent = agent

    def sample_text(self, prompt: str, max_tokens: int = 256, **_kwargs) -> str:
        text, _traj = self.agent.respond(prompt, max_new_tokens=max_tokens)
        return text

    def sample_choice(self, prompt: str, responses: list[str]) -> int:
        raise NotImplementedError("Plug in scoring-based choice once Concordia is wired up.")
