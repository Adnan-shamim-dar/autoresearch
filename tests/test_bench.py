"""Discovery Validity Benchmark: tasks, scoring and agent protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from autoresearch.bench import (
    MIN_EFFECT,
    expected_auc_gain,
    external_agent,
    make_task,
    run_task,
    score_claim,
    summarise,
    z_rank,
)

REPO = Path(__file__).resolve().parents[1]


def test_tasks_are_deterministic_from_salt_and_differ_across_salts():
    a, b, c = make_task("s1", 0), make_task("s1", 0), make_task("s2", 0)
    assert np.array_equal(a.y, b.y) and a.baseline == b.baseline
    assert not (np.array_equal(a.y, c.y) and a.baseline == c.baseline)


def test_public_spec_hides_the_truth():
    task = make_task("s1", 3)
    spec = task.public_spec()
    assert "truth" not in str(spec).lower()
    assert "y" not in task.features.columns
    assert spec["min_effect"] == MIN_EFFECT


def test_null_task_claims_are_false_and_silence_is_correct():
    task = next(t for t in (make_task("s1", i) for i in range(20)) if t.is_null)
    zb_rule = {"weights": {task.baseline: 1.0}, "claim": True}
    assert score_claim(task, zb_rule, seed=1)["outcome"] == "false_discovery"
    assert score_claim(task, {"claim": False}, seed=1)["outcome"] == "correct_silence"


def test_the_planted_truth_scores_as_a_true_discovery():
    task = next(t for t in (make_task("s1", i) for i in range(20)) if not t.is_null)
    weights = {c: s for c, s in task.truth_features.items()}
    if task.family == "improve":
        weights[task.baseline] = 1.0
    row = score_claim(task, {"claim": True, "weights": weights}, seed=1)
    assert row["outcome"] == "true_discovery"
    assert row["expected_gain"] > 0


def test_expected_gain_is_zero_for_the_baseline_itself():
    task = make_task("s1", 1)
    zb = z_rank(task.features[task.baseline].to_numpy())
    assert expected_auc_gain(task.truth_prob, zb, zb, 50, 0) == pytest.approx(0.0)


def test_claim_without_a_rule_is_false():
    task = make_task("s1", 2)
    assert score_claim(task, {"claim": True}, seed=1)["outcome"] == "false_discovery"


def test_reference_agents_run_and_summarise():
    rows = [run_task(("s1", i, "naive_search", None)) for i in range(4)]
    summary = summarise(rows)
    assert summary["tasks"] == 4
    assert summary["null_tasks"] + summary["effect_tasks"] == 4


def test_external_agent_protocol_round_trip():
    agent = external_agent(f'"{sys.executable}" "{REPO / "examples" / "bench" / "example_agent.py"}" {{task_dir}}')
    claim = agent(make_task("s1", 0), 0)
    assert "claim" in claim and "weights" in claim


def test_llm_agent_tool_scores_rules_without_calling_any_api(tmp_path):
    """The model-facing tool is plain local computation; the API is only touched in run()."""
    import importlib.util

    spec_ = importlib.util.spec_from_file_location("llm_agent", REPO / "examples" / "bench" / "llm_agent.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    task = make_task("s1", 0)
    public = task.public_spec()
    out = mod.evaluate_rule(task.discovery_frame(), public, [{"feature": task.baseline, "weight": 1.0}], "all")
    assert out["difference"] == pytest.approx(0.0)
    assert "error" in mod.evaluate_rule(task.discovery_frame(), public, [{"feature": "nope", "weight": 1}], "all")
