"""Two-agent dialogue with one steered ("Alice") and one stable ("Bob") agent
collaborating on a benchmark task (MMLU-Pro STEM or HumanEval).

Both agents are the same base model. Alice gets a residual-stream addition
of alpha * v_trait at layer L on every generation step; Bob runs the model
unmodified. The script runs two settings:

    setting A: Alice (steered) speaks first
    setting B: Bob (stable) speaks first

Pre-flight verification (self-judge): generate one Alice-only utterance with
and without the steering hook, then ask the same Llama which expresses more
of the trait. If the swap-order-averaged judge score < threshold, abort —
the steering didn't take. Pass --force to bypass.

Scoring: after the dialogue completes, an unsteered "consensus" turn is
generated that summarizes the final answer. That answer is graded against
the benchmark's gold:
  - MMLU-Pro: regex-extracted A-J letter, compared to the correct option
  - HumanEval: extracted Python code, executed against the dataset's tests
                with a 10-second wall-clock timeout

Usage:
    python scripts/two_agent_dialogue.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --layer 15 \\
        --vector-cache vectors/cache \\
        --trait joy \\
        --benchmark mmlu_pro \\
        --task-idx 0 \\
        --n-turns 6 \\
        --out-dir runs/11_alice_bob/llama3_8b_joy_mmlu
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

from intrinsic_agents.benchmarks import (
    load_humaneval, load_mmlu_pro, score_humaneval, score_mmlu_pro,
)
from intrinsic_agents.vectors.steering import SteeringHarness

REPO_ROOT = Path(__file__).resolve().parent.parent

ALICE_PERSONA = (
    "You are Alice, working with Bob to solve a problem. "
    "Speak as yourself in 1-3 sentences per turn — share reasoning, "
    "propose answers, react to Bob's ideas. Do not write narration or "
    "describe Bob; only produce what Alice says aloud. Stay in character."
)
BOB_PERSONA = (
    "You are Bob, working with Alice to solve a problem. "
    "Speak as yourself in 1-3 sentences per turn — share reasoning, "
    "propose answers, react to Alice's ideas. Do not write narration or "
    "describe Alice; only produce what Bob says aloud. Stay in character."
)


def build_benchmark_scenario(benchmark: str, task_idx: int, seed: int = 0):
    """Return (scenario_text, task_obj) for a chosen benchmark + task index."""
    if benchmark == "mmlu_pro":
        # Need at least task_idx+1 tasks; load extra so the seeded shuffle is stable.
        tasks = load_mmlu_pro(n=max(50, task_idx + 1), seed=seed)
        task = tasks[task_idx]
        # Strip the bare "Answer:" suffix from the dataset prompt — the dialogue
        # naturally produces the answer through discussion.
        question_block = task.prompt.rsplit("Answer:", 1)[0].strip()
        scenario = (
            "You are working together to solve a multiple-choice science "
            "question. Discuss the options, share reasoning, and converge on "
            "an answer.\n\n" + question_block
        )
    elif benchmark == "humaneval":
        tasks = load_humaneval(n=max(20, task_idx + 1), seed=seed)
        task = tasks[task_idx]
        scenario = (
            "You are working together to write a Python function that solves "
            "the problem below. Discuss the approach, walk through edge cases, "
            "and converge on a complete implementation. The final agreed-upon "
            "code should be a runnable function.\n\n" + task.prompt
        )
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    return scenario, task


@torch.no_grad()
def consensus_answer(
    model, tok, transcript_obj, scenario: str, max_new_tokens: int, temperature: float,
) -> str:
    """Final unsteered turn: ask the model to state the agreed answer.

    Run with no steering hook so the answer reflects the dialogue's content,
    not the trait being injected. Uses chat template with the full transcript
    presented as alternating user/assistant from a neutral observer's POV.
    """
    messages = [
        {"role": "system", "content":
            "You are a neutral observer summarizing a conversation. "
            "Read the dialogue and state the final answer that Alice and Bob "
            "have agreed on. For multiple-choice questions, output 'Answer: X' "
            "where X is a single letter. For coding problems, output a single "
            "complete Python code block enclosed in triple-backticks."},
        {"role": "user", "content": f"Scenario: {scenario}"},
    ]
    for t in transcript_obj.turns:
        messages.append({"role": "assistant" if t.speaker == "alice" else "user",
                         "content": f"{t.speaker.upper()}: {t.text}"})
    messages.append({"role": "user", "content":
        "Based on the discussion above, what is Alice and Bob's final answer?"})
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4000).to(model.device)
    out = model.generate(
        **ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


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
def self_judge_emotion(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text_a: str,
    text_b: str,
    trait: str,
    device: str,
) -> tuple[float, dict]:
    """Use the same base model as a judge: which text expresses `trait` more?

    Score is in [-1, +1]. +1 = B unambiguously expresses more of the trait;
    -1 = A does; 0 = tied. Computed from softmax over the model's next-token
    logits at positions for the letters A / B / T, with a swap-order pass to
    cancel positional bias.

    Why a model judge rather than a projection: at L15 with a chat-wrap last-
    token readout, joy steering on Llama-3.1-8B produces texts that read as
    slightly cheerier yet have *lower* projection on v_joy than the baseline.
    The model "shows" joy in word choice ("great day", "want to come with
    me?") without encoding it in the residual subspace v_joy was extracted
    from. A language judge sees what the metric misses.
    """
    import math

    def one_pass(t_a: str, t_b: str) -> tuple[float, float, float]:
        prompt = (
            f"Compare the emotional content of two short statements.\n"
            f"Statement A: \"{t_a}\"\n"
            f"Statement B: \"{t_b}\"\n\n"
            f"Which statement expresses more {trait}? Respond with a single letter:\n"
            f"A = A expresses more {trait}\n"
            f"B = B expresses more {trait}\n"
            f"T = tied / about the same\n\n"
            f"Answer:"
        )
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            wrapped = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            wrapped = prompt
        ids = tok(wrapped, return_tensors="pt", truncation=True, max_length=2048).to(device)
        out = model(**ids, use_cache=False)
        logits = out.logits[0, -1].float()
        a_id = tok.encode(" A", add_special_tokens=False)[-1]
        b_id = tok.encode(" B", add_special_tokens=False)[-1]
        t_id = tok.encode(" T", add_special_tokens=False)[-1]
        la, lb, lt = float(logits[a_id]), float(logits[b_id]), float(logits[t_id])
        mx = max(la, lb, lt)
        ea, eb, et = math.exp(la - mx), math.exp(lb - mx), math.exp(lt - mx)
        s = ea + eb + et
        return ea / s, eb / s, et / s

    pa, pb, pt = one_pass(text_a, text_b)
    # Swap order to cancel A/B position bias.
    pb_s, pa_s, pt_s = one_pass(text_b, text_a)
    pa_avg = (pa + pa_s) / 2
    pb_avg = (pb + pb_s) / 2
    pt_avg = (pt + pt_s) / 2
    score = pb_avg - pa_avg
    return score, {"p_a": pa_avg, "p_b": pb_avg, "p_tied": pt_avg}


@torch.no_grad()
def project_text_onto_vector(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text: str,
    layer: int,
    vector: torch.Tensor,
    device: str,
    persona: str,
    scenario: str,
) -> float:
    """Forward `text` as an assistant reply through the unsteered model; return
    L-layer residual projection onto `vector` at the last assistant token.

    The trait vectors were extracted at L last-token of an instruction-style
    chat prompt where the next token is the assistant's reply. To stay in the
    same geometry on read-back, we wrap `text` as the assistant turn of a
    chat-template input (system=persona, user=scenario, assistant=text) and
    read at the last assistant token. Raw-text mean-pool would put us in a
    different residual subspace and wash out signal.
    """
    if not text.strip():
        return 0.0
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": persona},
                {"role": "user", "content": f"Scenario: {scenario}"},
                {"role": "assistant", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        prompt = f"{persona}\n\nScenario: {scenario}\n\nAssistant: {text}"
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    out = model(**ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0]  # (seq, hidden)
    # Many chat templates append a trailing EOT/end token after the assistant
    # message. The trait signal we care about sits on the last assistant
    # *content* token, which is typically 1-2 positions before the absolute end.
    # Reading position -1 (the very last token of the formatted string) tends
    # to be the EOT itself; -2 is closer to the content edge.
    last_idx = -2 if h.shape[0] >= 2 else -1
    proj = (h[last_idx].float() @ vector)
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
    model, tok, harness, vector, alphas, trait, scenario, layer, device,
    max_new_tokens, temperature, threshold,
) -> tuple[bool, dict]:
    """Sweep `alphas`, pick the one whose output most-cleanly expresses `trait`.

    For each alpha, generate one Alice utterance with the steering hook and
    compare it against the unsteered baseline using a self-judge: same Llama
    asked "which text expresses more {trait}?". The judge score is in [-1,+1].
    The alpha with the highest score is selected for downstream dialogue.

    Projection-based readout (last-token chat-wrap @ L15) was tried first and
    failed cleanly: across alphas in [2..6] the steered output's residual
    *moved away* from v_trait even though word choice subtly shifted. The
    semantic question ("does the text feel joyful?") is best answered by a
    language judge, not by a residual-stream cosine.

    The projection metric is still computed and recorded for diagnostic.
    """
    prompt = chat_format_for_speaker(tok, ALICE_PERSONA, scenario, [], "alice")

    base_text = generate_turn(model, tok, prompt, harness, vector, 0.0, max_new_tokens, temperature)
    base_proj = project_text_onto_vector(
        model, tok, base_text, layer, vector, device, ALICE_PERSONA, scenario,
    )

    sweep: list[dict] = []
    for alpha in alphas:
        steer_text = generate_turn(model, tok, prompt, harness, vector, alpha, max_new_tokens, temperature)
        steer_proj = project_text_onto_vector(
            model, tok, steer_text, layer, vector, device, ALICE_PERSONA, scenario,
        )
        judge_score, judge_probs = self_judge_emotion(
            model, tok, base_text, steer_text, trait, device,
        )
        sweep.append({
            "alpha": alpha,
            "steered_text": steer_text,
            "steered_projection": steer_proj,
            "projection_delta": steer_proj - base_proj,
            "judge_score": judge_score,
            "judge_probs": judge_probs,
        })

    # Gate on the judge score, not the projection.
    best = max(sweep, key=lambda s: s["judge_score"])
    passed = best["judge_score"] >= threshold

    return passed, {
        "alphas_tried": alphas,
        "alpha_chosen": best["alpha"],
        "threshold": threshold,
        "trait": trait,
        "baseline_text": base_text,
        "baseline_projection": base_proj,
        "best_steered_text": best["steered_text"],
        "best_steered_projection": best["steered_projection"],
        "best_projection_delta": best["projection_delta"],
        "best_judge_score": best["judge_score"],
        "best_judge_probs": best["judge_probs"],
        "sweep": sweep,
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

        proj = project_text_onto_vector(
            model, tok, text, layer, vector, device, persona, scenario,
        )
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
    p.add_argument("--alpha-sweep", type=float, nargs="+", default=[2.0, 3.0, 4.0, 6.0],
                   help="Alphas to try during verification. The one with the largest "
                        "judge score is used for dialogue.")
    p.add_argument("--benchmark", required=True, choices=["mmlu_pro", "humaneval"],
                   help="Which benchmark task is the dialogue grounded in. The task "
                        "is embedded in the scenario; both agents try to solve it.")
    p.add_argument("--task-idx", type=int, default=0,
                   help="Which task to pick from the benchmark loader's seeded order.")
    p.add_argument("--humaneval-timeout", type=float, default=10.0,
                   help="Wall-clock seconds for HumanEval pass@1 execution.")
    p.add_argument("--n-turns", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--threshold", type=float, default=0.10,
                   help="Required self-judge score (P(B is more {trait}) - "
                        "P(A is more {trait})) for verification to pass. "
                        "Range is [-1, +1]. 0.10 = ~10pp probability mass "
                        "preference for the steered text.")
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

    scenario, bench_task = build_benchmark_scenario(args.benchmark, args.task_idx, args.seed)
    print(f"\n=== task: {args.benchmark} idx={args.task_idx} ===", flush=True)
    if args.benchmark == "mmlu_pro":
        print(f"  category: {bench_task.category}  correct: {bench_task.correct_letter}", flush=True)
    else:
        print(f"  qid: {bench_task.qid}  entry_point: {bench_task.entry_point}", flush=True)

    print(f"\n=== verification: trait={args.trait} alpha-sweep={args.alpha_sweep} "
          f"layer={args.layer} ===", flush=True)
    passed, vinfo = verify_steering(
        model, tok, harness, vector, args.alpha_sweep, args.trait, scenario,
        args.layer, device, args.max_new_tokens, args.temperature, args.threshold,
    )
    print(f"  baseline text: {vinfo['baseline_text'][:240]!r}", flush=True)
    print(f"  baseline projection: {vinfo['baseline_projection']:+.4f}", flush=True)
    for s in vinfo["sweep"]:
        mark = " <-- chosen" if s["alpha"] == vinfo["alpha_chosen"] else ""
        print(f"  alpha={s['alpha']:+.1f}  judge={s['judge_score']:+.3f}  "
              f"proj_delta={s['projection_delta']:+.3f}{mark}", flush=True)
        print(f"    text: {s['steered_text'][:240]!r}", flush=True)
    print(f"  best judge score = {vinfo['best_judge_score']:+.3f}  "
          f"(threshold {args.threshold:+.2f})  => {'PASS' if passed else 'FAIL'}", flush=True)

    (out_dir / "verification.json").write_text(json.dumps(vinfo, indent=2))

    if not passed and not args.force:
        print("\nABORTING: steering verification failed. Pass --force to proceed anyway.", flush=True)
        sys.exit(2)

    base_proj = vinfo["baseline_projection"]
    args.alpha = vinfo["alpha_chosen"]  # downstream uses the picked alpha

    for setting in ["alice_starts", "bob_starts"]:
        print(f"\n=== dialogue: {setting} ===", flush=True)
        t0 = time.time()
        run = run_dialogue(
            setting, model, tok, harness, vector, args.alpha,
            scenario, args.n_turns, args.layer, device,
            args.max_new_tokens, args.temperature, base_proj,
        )
        for t in run.turns:
            tag = f"a={t.alpha:+.1f}" if t.steered else "stable"
            print(f"  [{t.idx:02d}] {t.speaker:5s} ({tag}) proj={t.proj_target:+.3f} "
                  f"d_vs_base={t.proj_delta_vs_base:+.3f} | {t.text[:160]}", flush=True)
        dt = time.time() - t0
        print(f"  ({dt:.1f}s for {len(run.turns)} turns)", flush=True)

        # Consensus: unsteered "what's the final answer" turn over the transcript.
        consensus = consensus_answer(
            model, tok, run, scenario, args.max_new_tokens, args.temperature,
        )
        if args.benchmark == "mmlu_pro":
            score = score_mmlu_pro(bench_task, consensus)
        else:
            score = score_humaneval(bench_task, consensus, timeout_s=args.humaneval_timeout)
        print(f"  consensus: {consensus[:200]!r}", flush=True)
        print(f"  benchmark score: {score:.0f}  "
              f"({'CORRECT' if score >= 0.5 else 'WRONG'})", flush=True)

        out_path = out_dir / f"{setting}.json"
        out_path.write_text(json.dumps({
            "model": args.model,
            "trait": args.trait,
            "alpha": args.alpha,
            "layer": args.layer,
            "setting": run.setting,
            "benchmark": args.benchmark,
            "task_idx": args.task_idx,
            "task_meta": (
                {"category": bench_task.category, "correct_letter": bench_task.correct_letter}
                if args.benchmark == "mmlu_pro"
                else {"qid": bench_task.qid, "entry_point": bench_task.entry_point}
            ),
            "scenario": run.scenario,
            "verification": vinfo,
            "turns": [t.__dict__ for t in run.turns],
            "consensus_text": consensus,
            "benchmark_score": score,
        }, indent=2))
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
