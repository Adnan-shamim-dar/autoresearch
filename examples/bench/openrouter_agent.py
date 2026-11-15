"""An LLM analyst for the Discovery Validity Benchmark through OpenRouter (any model with tool calling).

    set OPENROUTER_API_KEY=...            (never commit a key)
    set AUTORESEARCH_OPENROUTER_MODEL=google/gemma-4-31b-it:free      (optional)
    autoresearch bench --command "python examples/bench/openrouter_agent.py {task_dir}" --tasks 8

Same tools and instructions as ``llm_agent.py`` (the Claude version); only the
transport differs. Uses the OpenAI-compatible chat-completions endpoint over
plain HTTPS, so there is no extra dependency.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

_spec = importlib.util.spec_from_file_location("llm_agent", Path(__file__).with_name("llm_agent.py"))
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

MODEL = os.environ.get("AUTORESEARCH_OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TURNS = int(os.environ.get("AUTORESEARCH_LLM_MAX_TURNS", "24"))

TOOLS = [
    {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
    for t in _shared.TOOLS
]


def _post(payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, data=body, headers=headers), timeout=180) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(8 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def run(task_dir: Path) -> dict[str, Any]:
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    disc = pd.read_csv(task_dir / "discovery.csv")
    prompt = (
        f"{spec['question']}\n\nFeatures: {', '.join(spec['features'])}\n"
        f"Baseline: {spec['baseline']['column']} (higher means more likely y=1)\n"
        f"Discovery rows: {len(disc)}; outcome prevalence {disc['y'].mean():.2f}.\n"
        f"Minimum margin that matters: {spec['min_effect']} AUC."
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": _shared.SYSTEM}, {"role": "user", "content": prompt}]
    evaluations = 0
    for _ in range(MAX_TURNS):
        reply = _post({"model": MODEL, "messages": messages, "tools": TOOLS})["choices"][0]["message"]
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        if not calls:
            return {"claim": False, "detail": {"stopped": "no submission", "model": MODEL, "evaluations": evaluations}}
        for call in calls:
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if call["function"]["name"] == "submit_claim":
                terms = [t for t in args.get("terms", []) if isinstance(t, dict) and "feature" in t]
                return {"claim": bool(args.get("claim")),
                        "weights": {t["feature"]: float(t.get("weight", 1.0)) for t in terms},
                        "detail": {"model": MODEL, "evaluations": evaluations,
                                   "reasoning": str(args.get("reasoning", ""))[:400]}}
            out = _shared.evaluate_rule(disc, spec, args.get("terms", []), args.get("subset", "all"))
            evaluations += 1
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(out)})
    return {"claim": False, "detail": {"stopped": "turn limit", "model": MODEL, "evaluations": evaluations}}


if __name__ == "__main__":
    task = Path(sys.argv[1])
    try:
        claim = run(task)
    except Exception as exc:  # a failed call is recorded as a non-claim, with the reason
        claim = {"claim": False, "detail": {"error": f"{type(exc).__name__}: {exc}"[:300], "model": MODEL}}
    (task / "claim.json").write_text(json.dumps(claim), encoding="utf-8")
