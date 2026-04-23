"""Sanity checks for the vector extraction pipeline.

These tests use a tiny `sshleifer/tiny-gpt2` model so they can run in CI on CPU
in a couple of seconds. They verify only *shape* and *unit-norm* properties —
semantic evaluation (honest-vs-deceptive separation) is deferred to an integration
test on a real model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer

from intrinsic_agents.vectors.extract import (
    default_layer,
    extract_trait_vector,
    load_traits,
)

TINY_MODEL = "sshleifer/tiny-gpt2"
TRAITS_YAML = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "intrinsic_agents"
    / "vectors"
    / "traits.yaml"
)


@pytest.fixture(scope="module")
def tiny():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32)
    model.eval()
    return model, tok


def test_traits_yaml_loads():
    traits = load_traits(TRAITS_YAML)
    assert "honesty" in traits
    assert "joy" in traits
    for spec in traits.values():
        assert len(spec.pairs) >= 4
        for pair in spec.pairs:
            assert "pos" in pair and "neg" in pair


def test_default_layer_is_middle(tiny):
    model, _ = tiny
    L = default_layer(model)
    n = model.config.num_hidden_layers
    assert 0 <= L <= n


def test_extract_trait_vector_shape_and_norm(tiny):
    model, tok = tiny
    traits = load_traits(TRAITS_YAML)
    L = default_layer(model)
    vec = extract_trait_vector(model, tok, traits["honesty"], layer=L, device="cpu")
    assert vec.ndim == 1
    assert vec.shape[0] == model.config.hidden_size
    # unit norm up to fp noise
    assert abs(vec.norm().item() - 1.0) < 1e-4


def test_two_traits_not_collinear(tiny):
    """Different traits should produce distinguishable (non-collinear) vectors."""
    model, tok = tiny
    traits = load_traits(TRAITS_YAML)
    L = default_layer(model)
    v_honesty = extract_trait_vector(model, tok, traits["honesty"], layer=L)
    v_joy = extract_trait_vector(model, tok, traits["joy"], layer=L)
    cos = torch.dot(v_honesty, v_joy).abs().item()
    # Tiny-gpt2 is near-random, so we only require "not essentially parallel".
    assert cos < 0.99
