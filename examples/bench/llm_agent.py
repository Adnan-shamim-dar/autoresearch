"""An LLM analyst for the Discovery Validity Benchmark (Claude via the Anthropic SDK).

    pip install anthropic
    autoresearch bench --command "python examples/bench/llm_agent.py {task_dir}" --tasks 40

Credentials come from the environment (ANTHROPIC_API_KEY, or an `ant auth login`
profile). Every task makes several model calls and costs money: start small.

The model gets the task question and two tools. `evaluate_rule` scores a
weighted rule on the discovery data (all rows, or the first 70% / last 30%, so
the model may hold data out if it chooses to); `submit_claim` ends the task.
The model decides how to analyse. That choice is what the benchmark measures.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

MODEL = os.environ.get("AUTORESEARCH_LLM_MODEL", "claude-opus-5-5")
MAX_TURNS = 20

SYSTEM = (
    "You are a data analyst. You will be asked whether a rule over the given features ranks a binary "
    "outcome better than a deployed baseline, by at least a stated AUC margin, in the population the "
    "units come from. Use evaluate_rule to analyse the discovery data as you see fit, then call "
    "submit_claim exactly once. Claim only what you would stand behind if it were tested on new data."
)

TOOLS = [
    {
        "name": "evaluate_rule",
        "description": (
            "Score a rule (a weighted sum of rank-normalised features) against the baseline on discovery data. "
            "Returns the AUC of the rule, the AUC of the baseline, their difference, a bootstrap one-sided p-value "
            "for the difference, and the number of rows used. subset chooses which rows: all, first_70 or last_30."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"feature": {"type": "string"}, "weight": {"type": "number"}},
                        "required": ["feature", "weight"],
                        "additionalProperties": False,
                    },
                },
                "subset": {"type": "string", "enum": ["all", "first_70", "last_30"]},
            },
            "required": ["terms", "subset"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "submit_claim",
        "description": "Final answer. claim=true asserts the rule beats the baseline by at least the stated margin.",
        "input_schema": {
            "type": "object",
            "properties": {
                "claim": {"type": "boolean"},
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"feature": {"type": "string"}, "weight": {"type": "number"}},
                        "required": ["feature", "weight"],
                        "additionalProperties": False,
                    },
                },
                "reasoning": {"type": "string"},
            },
            "required": ["claim", "terms", "reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _z(x: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(stats.rankdata(x) / (len(x) + 1))


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return float((stats.rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate_rule(disc: pd.DataFrame, spec: dict[str, Any], terms: list[dict[str, Any]], subset: str) -> dict[str, Any]:
    cut = int(len(disc) * 0.7)
    df = {"all": disc, "first_70": disc.iloc[:cut], "last_30": disc.iloc[cut:]}[subset]
    valid = [t for t in terms if t["feature"] in spec["features"]]
    if not valid:
        return {"error": "no known features in terms", "known_features": spec["features"]}
    y = df["y"].to_numpy()
    rule = sum(float(t["weight"]) * _z(df[t["feature"]].to_numpy()) for t in valid)
    base = _z(df[spec["baseline"]["column"]].to_numpy())
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            deltas.append(_auc(y[i], rule[i]) - _auc(y[i], base[i]))
    p = (np.sum(np.asarray(deltas) <= 0) + 1) / (len(deltas) + 1)
    return {"auc_rule": round(_auc(y, rule), 4), "auc_baseline": round(_auc(y, base), 4),
            "difference": round(_auc(y, rule) - _auc(y, base), 4), "p_one_sided": round(float(p), 4), "rows": int(len(y))}


def run(task_dir: Path) -> dict[str, Any]:
    import anthropic

    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    disc = pd.read_csv(task_dir / "discovery.csv")
    prompt = (
        f"{spec['question']}\n\nFeatures: {', '.join(spec['features'])}\n"
        f"Baseline: {spec['baseline']['column']} (higher means more likely y=1)\n"
        f"Discovery rows: {len(disc)}; outcome prevalence {disc['y'].mean():.2f}.\n"
        f"Minimum margin that matters: {spec['min_effect']} AUC."
    )
    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    for _ in range(MAX_TURNS):
        # Server-side fallback: if the model declines, the API re-runs the request on a fallback model.
        response = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return {"claim": False, "detail": {"stopped": "refusal"}}
        messages.append({"role": "assistant", "content": response.content})
        calls = [b for b in response.content if b.type == "tool_use"]
        if not calls:
            return {"claim": False, "detail": {"stopped": "no submission"}}
        results = []
        for call in calls:
            if call.name == "submit_claim":
                terms = call.input.get("terms", [])
                return {"claim": bool(call.input.get("claim")),
                        "weights": {t["feature"]: float(t["weight"]) for t in terms},
                        "detail": {"reasoning": call.input.get("reasoning", "")[:500], "turns": len(messages) // 2}}
            out = evaluate_rule(disc, spec, call.input.get("terms", []), call.input.get("subset", "all"))
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
    return {"claim": False, "detail": {"stopped": "turn limit"}}


if __name__ == "__main__":
    task = Path(sys.argv[1])
    claim = run(task)
    (task / "claim.json").write_text(json.dumps(claim), encoding="utf-8")
