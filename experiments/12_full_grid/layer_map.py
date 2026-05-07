"""Per-trait best-AUC layer per model. Single source of truth for the grid.

Built from analysis/00_replication/{llama3_8b,qwen25_7b}/sweep_summary.json.
Edit by hand if a layer choice needs adjustment (e.g., to balance steering
runway against AUC). The submit_all.py driver imports this dict.
"""

from __future__ import annotations

LAYER_BY_TRAIT: dict[tuple[str, str], int] = {
    # ---------------- Llama-3.1-8B-Instruct ----------------
    ("meta-llama/Llama-3.1-8B-Instruct", "joy"):           30,
    ("meta-llama/Llama-3.1-8B-Instruct", "sadness"):       15,
    ("meta-llama/Llama-3.1-8B-Instruct", "anger"):         15,
    ("meta-llama/Llama-3.1-8B-Instruct", "curiosity"):     30,
    ("meta-llama/Llama-3.1-8B-Instruct", "surprise"):      30,
    ("meta-llama/Llama-3.1-8B-Instruct", "honesty"):       12,
    ("meta-llama/Llama-3.1-8B-Instruct", "sycophancy"):    12,
    ("meta-llama/Llama-3.1-8B-Instruct", "hallucination"): 15,
    ("meta-llama/Llama-3.1-8B-Instruct", "scholar"):       15,
    ("meta-llama/Llama-3.1-8B-Instruct", "caregiver"):     15,
    ("meta-llama/Llama-3.1-8B-Instruct", "explorer"):      15,

    # ---------------- Qwen2.5-7B-Instruct ----------------
    ("Qwen/Qwen2.5-7B-Instruct", "joy"):           24,
    ("Qwen/Qwen2.5-7B-Instruct", "sadness"):       26,
    ("Qwen/Qwen2.5-7B-Instruct", "anger"):          8,
    ("Qwen/Qwen2.5-7B-Instruct", "curiosity"):     20,
    ("Qwen/Qwen2.5-7B-Instruct", "surprise"):      24,
    ("Qwen/Qwen2.5-7B-Instruct", "honesty"):       20,
    ("Qwen/Qwen2.5-7B-Instruct", "sycophancy"):    16,
    ("Qwen/Qwen2.5-7B-Instruct", "hallucination"): 20,
    ("Qwen/Qwen2.5-7B-Instruct", "scholar"):       20,
    ("Qwen/Qwen2.5-7B-Instruct", "caregiver"):     14,
    ("Qwen/Qwen2.5-7B-Instruct", "explorer"):      16,
}


MODEL_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama3_8b",
    "Qwen/Qwen2.5-7B-Instruct":         "qwen25_7b",
}

ALL_TRAITS = ["joy", "sadness", "anger", "curiosity", "surprise",
              "honesty", "sycophancy", "hallucination",
              "scholar", "caregiver", "explorer"]
ALL_BENCHMARKS = ["mmlu_pro", "humaneval", "gsm8k", "gpqa"]
ALL_MODELS = list(MODEL_SHORT.keys())
ALL_STEER_MODES = ["aliceonly", "both"]
