"""Extract persona/emotion vectors from a base LLM via contrastive probing.

Method (Anthropic 2025, Persona Vectors):
    For each (pos, neg) prompt pair, capture the residual-stream activation at
    layer L at the final prompt token. The trait vector is the L2-normalized
    difference of per-class means:

        v = normalize( mean(h_pos) - mean(h_neg) )

    One forward pass per prompt, no generation. Runs on CPU for small models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TraitSpec:
    name: str
    kind: str
    description: str
    pairs: list[dict[str, str]]


def load_traits(path: str | Path) -> dict[str, TraitSpec]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {
        name: TraitSpec(name=name, **spec)
        for name, spec in raw["traits"].items()
    }


def _last_token_hidden(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int,
    device: str,
) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True, use_cache=False)
    # hidden_states[layer] has shape (1, seq_len, hidden_dim); take last token.
    return out.hidden_states[layer][0, -1].float().cpu()


def extract_trait_vector(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    trait: TraitSpec,
    layer: int,
    device: str = "cpu",
) -> torch.Tensor:
    pos_acts, neg_acts = [], []
    for pair in trait.pairs:
        pos_acts.append(_last_token_hidden(model, tokenizer, pair["pos"], layer, device))
        neg_acts.append(_last_token_hidden(model, tokenizer, pair["neg"], layer, device))
    diff = torch.stack(pos_acts).mean(0) - torch.stack(neg_acts).mean(0)
    return diff / (diff.norm() + 1e-8)


def default_layer(model: AutoModelForCausalLM) -> int:
    """Middle layer — empirically where trait signals concentrate."""
    n = getattr(model.config, "num_hidden_layers", None) or len(model.transformer.h)
    return n // 2


def cache_path(cache_dir: str | Path, model_name: str, trait_name: str, layer: int) -> Path:
    safe = model_name.replace("/", "_")
    return Path(cache_dir) / f"{safe}_{trait_name}_layer{layer}.pt"


def extract_all(
    model_name: str,
    traits_yaml: str | Path,
    out_dir: str | Path,
    layer: int | None = None,
    device: str | None = None,
    selected: list[str] | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, Path]:
    """Extract vectors for every trait in the YAML. Returns {trait: cache_path}."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        # bf16 on cuda saves half the VRAM and is fine for difference-of-means;
        # fp32 on cpu since cpu bf16 ops are unsupported on most kernels.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()

    if layer is None:
        layer = default_layer(model)

    traits = load_traits(traits_yaml)
    if selected:
        traits = {k: v for k, v in traits.items() if k in selected}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name, spec in traits.items():
        vec = extract_trait_vector(model, tok, spec, layer=layer, device=device)
        path = cache_path(out_dir, model_name, name, layer)
        torch.save(
            {
                "vector": vec,
                "model": model_name,
                "trait": name,
                "kind": spec.kind,
                "layer": layer,
                "hidden_dim": vec.shape[0],
            },
            path,
        )
        written[name] = path
    return written
