"""Runtime probe: one forward hook, one matmul.

Registers a hook on the residual stream at layer L, stacks all cached trait
vectors into a single `[n_traits, hidden_dim]` matrix, and projects every token's
activation onto that matrix in one matmul. Emits a per-step dict
`{trait_name: scalar}` that `rewards/composer.py` consumes at rollout end.
"""

from __future__ import annotations

from pathlib import Path

import torch


def _layer_module(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    """Return the transformer block at index `layer`. Covers common HF layouts."""
    # Llama/Qwen/Mistral: model.model.layers[i]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer]
    # GPT-2 family: model.transformer.h[i]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer]
    raise AttributeError(
        f"Don't know how to index layer {layer} on {type(model).__name__}"
    )


class ActivationProbe:
    """Captures last-token residual-stream projection onto a bank of trait vectors.

    Usage:
        probe = ActivationProbe.from_cache_dir("vectors/cache", model_name, layer=L)
        probe.attach(model)
        ... model(**inputs) ...
        scores = probe.pop()   # {"honesty": 0.37, "joy": -0.12, ...} for the last token
        probe.detach()
    """

    def __init__(self, trait_names: list[str], bank: torch.Tensor, layer: int):
        self.trait_names = trait_names
        # `bank` shape: [n_traits, hidden_dim], each row unit-norm.
        self.bank = bank
        self.layer = layer
        self._handle = None
        self._buffer: list[torch.Tensor] = []

    @classmethod
    def from_cache_dir(
        cls,
        cache_dir: str | Path,
        model_name: str,
        layer: int,
        traits: list[str] | None = None,
    ) -> "ActivationProbe":
        cache_dir = Path(cache_dir)
        safe = model_name.replace("/", "_")
        pattern = f"{safe}_*_layer{layer}.pt"
        files = sorted(cache_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No cached vectors found in {cache_dir} matching {pattern}. "
                "Run scripts/extract_vectors.py first."
            )
        names, vecs = [], []
        for f in files:
            blob = torch.load(f, map_location="cpu", weights_only=False)
            trait = blob["trait"]
            if traits is not None and trait not in traits:
                continue
            names.append(trait)
            vecs.append(blob["vector"])
        bank = torch.stack(vecs, dim=0)
        return cls(trait_names=names, bank=bank, layer=layer)

    def attach(self, model: torch.nn.Module) -> None:
        if self._handle is not None:
            raise RuntimeError("Probe already attached")
        block = _layer_module(model, self.layer)
        bank = self.bank.to(next(model.parameters()).device, dtype=torch.float32)
        self.bank = bank

        def hook(_module, _inp, out):
            # Most HF blocks return Tuple[hidden_states, ...]; some return Tensor.
            h = out[0] if isinstance(out, tuple) else out
            # h: [batch, seq, hidden]. Project every token onto the bank.
            proj = torch.matmul(h.float(), bank.t())  # [batch, seq, n_traits]
            self._buffer.append(proj.detach())

        self._handle = block.register_forward_hook(hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._buffer.clear()

    def pop(self, reduction: str = "last") -> dict[str, float]:
        """Collapse buffered projections into per-trait scalars, then clear.

        reduction:
            "last" — take the last token of the last batch (typical for gen step)
            "mean" — mean across all tokens of all batches
        """
        if not self._buffer:
            return {name: 0.0 for name in self.trait_names}
        if reduction == "last":
            last = self._buffer[-1]  # [batch, seq, n_traits]
            scores = last[:, -1, :].mean(dim=0)
        elif reduction == "mean":
            stacked = torch.cat([b.reshape(-1, b.shape[-1]) for b in self._buffer], dim=0)
            scores = stacked.mean(dim=0)
        else:
            raise ValueError(f"unknown reduction {reduction!r}")
        self._buffer.clear()
        return {name: float(scores[i].item()) for i, name in enumerate(self.trait_names)}

    def pop_trajectory(self) -> dict[str, torch.Tensor]:
        """Per-trait tensor of per-token projections across all buffered forwards."""
        if not self._buffer:
            return {name: torch.zeros(0) for name in self.trait_names}
        stacked = torch.cat([b.reshape(-1, b.shape[-1]) for b in self._buffer], dim=0)
        self._buffer.clear()
        return {name: stacked[:, i].cpu() for i, name in enumerate(self.trait_names)}
