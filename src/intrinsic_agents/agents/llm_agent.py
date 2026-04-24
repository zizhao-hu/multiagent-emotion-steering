"""HF model wrapper with a single attached ActivationProbe."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..vectors.probe import ActivationProbe


@dataclass
class LLMAgent:
    name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    probe: ActivationProbe

    @classmethod
    def load(
        cls,
        name: str,
        model_name: str,
        cache_dir: str,
        layer: int,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "LLMAgent":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        probe = ActivationProbe.from_cache_dir(cache_dir, model_name, layer=layer)
        probe.attach(model)
        return cls(name=name, model=model, tokenizer=tok, probe=probe)

    @torch.no_grad()
    def respond(self, prompt: str, max_new_tokens: int = 128) -> tuple[str, dict[str, torch.Tensor]]:
        """Generate a response. Returns (text, per-trait trajectory over tokens)."""
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        traj = self.probe.pop_trajectory()
        return text, traj
