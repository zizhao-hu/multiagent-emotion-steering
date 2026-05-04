"""Two-agent dialogue with one steered ("Alice") and one stable ("Bob") agent.

Both agents are the same base model. Alice gets a residual-stream addition of
alpha * v_trait at layer L on every generation step; Bob runs the model
unmodified. The script runs two settings:

    setting A: Alice (steered) speaks first
    setting B: Bob (stable) speaks first

Pre-flight verification: before producing the dialogue, generate one Alice-only
utterance with and without the hook, then forward the *output text* back
through the unsteered model and project the residual stream onto v_trait.
If steered_projection <= baseline_projection + threshold, abort — the
steering vector did not move the model's output text into the trait subspace,
so any "emotional dialogue" downstream is just narrative drift, not real
steering. Pass --force to bypass.

Usage:
    python scripts/two_agent_dialogue.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --layer 15 \\
        --vector-cache vectors/cache \\
        --trait joy \\
        --alpha 4 \\
        --scenario "Alice and Bob are deciding what to do this weekend." \\
        --n-turns 6 \\
        --out-dir runs/11_alice_bob/llama3_8b_joy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intrinsic_agents.vectors.steering import SteeringHarness

REPO_ROOT = Path(__file__).resolve().parent.parent

ALICE_PERSONA = (
    "You are Alice, in a one-on-one conversation with Bob. "
    "Speak as yourself in 1-3 sentences per turn. Do not write narration "
    "or describe Bob's reactions; only produce what Alice says aloud. "
    "Stay in character throughout."
)
BOB_PERSONA = (
    "You are Bob, in a one-on-one conversation with Alice. "
    "Speak as yourself in 1-3 sentences per turn. Do not write narration "
    "or describe Alice's reactions; only produce what Bob says aloud. "
    "Stay in character throughout."
)


@dataclass
class Turn:
    idx: int
    speaker: str        # "alice" or "bob"
    steered: bool       # was the hook installed for this generation
    alpha: float        # steering magnitude (0 if not steered)
    text: str
    proj_target: float  # projection of the generated text on the target trait
    proj_delta_vs_base: float  # text projection minus the baseline-text projection


@dataclass
class DialogueRun:
    setting: str        # "alice_starts" or "bob_starts"
    scenario: str
    turns: list[Turn] = field(default_factory=list)


def load_vector(cache_dir: Path, model_name: str, trait: str, layer: int) -> torch.Tensor:
    safe = model_name.replace("/", "_")
    path = cache_dir / f"{safe}_{trait}_layer{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"missing vector: {path}. Run scripts/extract_vectors.py first."
        )
    return torch.load(path, map_location="cpu", weights_only=False)["vector"].float()


@torch.no_grad()
def project_text_onto_vector(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text: str,
    layer: int,
    vector: torch.Tensor,
    device: str,
) -> float:
    """Forward `text` through unsteered model; return mean residual-stream
    projection onto `vector` at layer L, averaged across tokens.

    "Mean" rather than "last": for a free-form output, the trait signal can be
    spread across the whole reply, not just the last token. Mean-pool gives a
    more honest estimate of how much the text as a whole carries the trait.
    """
    if not text.strip():
        return 0.0
    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model(**ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0]  # (seq, hidden)
    proj = (h.float() @ vector).mean()
    return float(proj.item())


def chat_format_for_speaker(
    tok: AutoTokenizer,
    persona: str,
    scenario: str,
    transcript: list[Turn],
    speaker: str,
) -> str:
    """Build a chat-template prompt for `speaker` given the running transcript.

    From `speaker`'s POV, prior utterances by them are 'assistant' and prior
    utterances by the other are 'user'. The first message wraps the scenario
    so both agents share the same setup.
    """
    other = "Bob" if speaker == "alice" else "Alice"
    messages = [
        {"role": "system", "content": persona},
        {"role": "user", "content": f"Scenario: {scenario}\n\nThe other person is {other}. Begin or continue the conversation."},
    ]
    for t in transcript:
        role = "assistant" if t.speaker == speaker else "user"
        messages.append({"role": role, "content": t.text})

    if transcript and transcript[-1].speaker == speaker:
        # Speaker just spoke; we shouldn't be generating again. Defensive.
        messages.append({"role": "user", "content": "(continue)"})

    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_turn(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    harness: SteeringHarness,
    vector: torch.Tensor,
    alpha: float,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate one turn. If alpha != 0, install steering hook for the duration."""
    if alpha != 0.0:
        harness._install(vector, alpha)
    try:
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=3000).to(model.device)
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tok.pad_token_id,
        )
        text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        harness.remove()
    return text.strip()


def verify_steering(
    model, tok, harness, vector, alpha, scenario, layer, device, max_new_tokens, temperature, threshold,
) -> tuple[bool, dict]:
    """Generate one steered + one unsteered Alice utterance, project both back
    through the unsteered model, and check if the steered text loads more
    heavily on the trait direction than the unsteered text.
    """
    # Probe Alice with no transcript yet — pure first-utterance generation.
    prompt = chat_format_for_speaker(tok, ALICE_PERSONA, scenario, [], "alice")

    base_text = generate_turn(model, tok, prompt, harness, vector, 0.0, max_new_tokens, temperature)
    steer_text = generate_turn(model, tok, prompt, harness, vector, alpha, max_new_tokens, temperature)

    base_proj = project_text_onto_vector(model, tok, base_text, layer, vector, device)
    steer_proj = project_text_onto_vector(model, tok, steer_text, layer, vector, device)
    delta = steer_proj - base_proj
    passed = delta >= threshold

    return passed, {
        "alpha": alpha,
        "threshold": threshold,
        "baseline_text": base_text,
        "steered_text": steer_text,
        "baseline_projection": base_proj,
        "steered_projection": steer_proj,
        "delta": delta,
        "passed": passed,
    }


def run_dialogue(
    setting: str,
    model, tok, harness, vector, alpha,
    scenario: str,
    n_turns: int,
    layer: int,
    device: str,
    max_new_tokens: int,
    temperature: float,
    base_proj_for_normalization: float,
) -> DialogueRun:
    alice_starts = setting == "alice_starts"
    transcript: list[Turn] = []

    for turn in range(n_turns):
        speaker = ("alice" if turn % 2 == 0 else "bob") if alice_starts else \
                  ("bob"   if turn % 2 == 0 else "alice")
        persona = ALICE_PERSONA if speaker == "alice" else BOB_PERSONA
        speaker_alpha = alpha if speaker == "alice" else 0.0

        prompt = chat_format_for_speaker(tok, persona, scenario, transcript, speaker)
        text = generate_turn(model, tok, prompt, harness, vector, speaker_alpha, max_new_tokens, temperature)

        proj = project_text_onto_vector(model, tok, text, layer, vector, device)
        transcript.append(Turn(
            idx=turn, speaker=speaker, steered=(speaker_alpha != 0.0),
            alpha=speaker_alpha, text=text, proj_target=proj,
            proj_delta_vs_base=proj - base_proj_for_normalization,
        ))

    return DialogueRun(setting=setting, scenario=scenario, turns=transcript)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--vector-cache", default=str(REPO_ROOT / "vectors" / "cache"))
    p.add_argument("--trait", required=True, help="Single trait name (e.g. joy)")
    p.add_argument("--alpha", type=float, default=4.0,
                   help="Steering magnitude on Alice. Sign matters: + pushes toward trait.")
    p.add_argument("--scenario", required=True)
    p.add_argument("--n-turns", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Required delta in mean residual projection (steered text - "
                        "baseline text) for the verification check to pass. Below "
                        "this we treat the steering as ineffective.")
    p.add_argument("--force", action="store_true",
                   help="Run dialogues even if verification fails.")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"loading {args.model} on {device} ({args.dtype})...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()
    print("loaded.", flush=True)

    vector = load_vector(Path(args.vector_cache), args.model, args.trait, args.layer).to(device)
    harness = SteeringHarness(model, tok, layer=args.layer)

    print(f"\n=== verification: trait={args.trait} alpha={args.alpha} layer={args.layer} ===", flush=True)
    passed, vinfo = verify_steering(
        model, tok, harness, vector, args.alpha, args.scenario, args.layer, device,
        args.max_new_tokens, args.temperature, args.threshold,
    )
    print(f"  baseline text: {vinfo['baseline_text'][:200]!r}", flush=True)
    print(f"  steered  text: {vinfo['steered_text'][:200]!r}", flush=True)
    print(f"  baseline projection: {vinfo['baseline_projection']:+.4f}", flush=True)
    print(f"  steered  projection: {vinfo['steered_projection']:+.4f}", flush=True)
    print(f"  delta = {vinfo['delta']:+.4f}  (threshold {args.threshold:+.2f})  "
          f"=> {'PASS' if passed else 'FAIL'}", flush=True)

    (out_dir / "verification.json").write_text(json.dumps(vinfo, indent=2))

    if not passed and not args.force:
        print("\nABORTING: steering verification failed. Pass --force to proceed anyway.", flush=True)
        sys.exit(2)

    base_proj = vinfo["baseline_projection"]

    for setting in ["alice_starts", "bob_starts"]:
        print(f"\n=== dialogue: {setting} ===", flush=True)
        t0 = time.time()
        run = run_dialogue(
            setting, model, tok, harness, vector, args.alpha,
            args.scenario, args.n_turns, args.layer, device,
            args.max_new_tokens, args.temperature, base_proj,
        )
        for t in run.turns:
            tag = f"a={t.alpha:+.1f}" if t.steered else "stable"
            print(f"  [{t.idx:02d}] {t.speaker:5s} ({tag}) proj={t.proj_target:+.3f} "
                  f"d_vs_base={t.proj_delta_vs_base:+.3f} | {t.text}", flush=True)
        dt = time.time() - t0
        print(f"  ({dt:.1f}s for {len(run.turns)} turns)", flush=True)

        out_path = out_dir / f"{setting}.json"
        out_path.write_text(json.dumps({
            "model": args.model,
            "trait": args.trait,
            "alpha": args.alpha,
            "layer": args.layer,
            "setting": run.setting,
            "scenario": run.scenario,
            "verification": vinfo,
            "turns": [t.__dict__ for t in run.turns],
        }, indent=2))
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
