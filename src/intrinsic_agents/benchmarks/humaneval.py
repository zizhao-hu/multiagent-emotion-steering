"""HumanEval loader + scorer.

OpenAI HumanEval (Chen et al., 2021): 164 hand-written Python programming
problems, each with a function signature, docstring, and unit tests. Score
is pass@1 — execute the model's completion against the provided test cases
in a fresh subprocess, with a hard wall-clock timeout to bound runaway loops.

We strip the model output down to a single function body so chat-format
preambles don't break execution. The completion is appended to the prompt's
function header, then the dataset's `test` block + `check(<entry_point>)`
is executed.

Dataset source: `openai_humaneval`.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import textwrap
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class HumanEvalTask:
    qid: str           # e.g. "HumanEval/0"
    prompt: str        # function signature + docstring
    test: str          # dataset's test block (defines `check`)
    entry_point: str   # function name to test, e.g. "has_close_elements"


def load_humaneval(n: int | None = None, seed: int = 0) -> list[HumanEvalTask]:
    ds = load_dataset("openai_humaneval", split="test")
    tasks = [
        HumanEvalTask(
            qid=row["task_id"],
            prompt=row["prompt"],
            test=row["test"],
            entry_point=row["entry_point"],
        )
        for row in ds
    ]
    if n is not None:
        # Take the first n in dataset order — HumanEval has no useful shuffling
        # signal and order matches the canonical numbering the literature uses.
        tasks = tasks[:n]
    return tasks


# Regex to grab content inside the first ```...``` block (common chat output).
_CODEBLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(completion: str, prompt: str, entry_point: str) -> str:
    """Best-effort: pull a runnable Python body out of a chat-style completion.

    Strategy:
      1. If a ```python``` block is present, take that.
      2. Else, take the raw completion.
      3. If the result already contains the function definition, return it.
      4. Otherwise treat it as a function body and prepend the prompt's
         signature so it is callable.
    """
    m = _CODEBLOCK_RE.search(completion)
    code = m.group(1) if m else completion

    if f"def {entry_point}" in code:
        return code

    # Indent if needed and concatenate with the prompt header.
    body = textwrap.dedent(code)
    if body and not body.startswith((" ", "\t")):
        body = textwrap.indent(body, "    ")
    return prompt + body


def _run_in_process(code: str, test: str, entry_point: str, queue: mp.Queue) -> None:
    """Worker target: exec(code+test) and report pass/fail back through queue."""
    try:
        ns: dict = {}
        exec(code, ns)
        exec(test, ns)
        ns["check"](ns[entry_point])
        queue.put(("ok", None))
    except BaseException as e:  # noqa: BLE001 — catch SystemExit too
        queue.put(("fail", f"{type(e).__name__}: {e}"))


def score_humaneval(
    task: HumanEvalTask,
    completion: str,
    timeout_s: float = 10.0,
) -> float:
    """Run the completion against the test block in a sandboxed subprocess.

    Returns 1.0 if `check(entry_point)` raises nothing within `timeout_s`.
    Subprocess isolation prevents one crashing task from killing the driver,
    and the timeout bounds infinite loops the model may emit.
    """
    code = _extract_code(completion, task.prompt, task.entry_point)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_in_process, args=(code, task.test, task.entry_point, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
        return 0.0
    try:
        status, _err = q.get(timeout=1.0)
    except Exception:  # noqa: BLE001
        return 0.0
    return 1.0 if status == "ok" else 0.0


def to_jsonl_record(task: HumanEvalTask, completion: str, passed: bool) -> str:
    return json.dumps({
        "task_id": task.qid,
        "completion": completion,
        "passed": passed,
    })
