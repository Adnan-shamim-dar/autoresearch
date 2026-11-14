"""Spec §5: the proposer's feedback channel carries categories, never numbers."""

from __future__ import annotations

import pytest

from autoresearch.feedback_leak import run_experiment
from autoresearch.proposer import sanitize_feedback


def test_sanitize_strips_every_number():
    raw = [
        {"id": "c1", "status": "kept", "reason_code": "kept_top_fraction", "score": 0.08, "p": 0.01, "margin": 0.3},
        {"id": "c2", "status": "rejected", "reason_code": "rejected_below_cut", "mean_delta": -0.02},
    ]
    clean = sanitize_feedback(raw)
    assert clean == [
        {"id": "c1", "status": "kept", "reason_code": "kept_top_fraction"},
        {"id": "c2", "status": "rejected", "reason_code": "rejected_below_cut"},
    ]


@pytest.mark.parametrize("code", ["delta=0.08", "rejected (p 0.06)", "Kept", "kept_0_08"])
def test_reason_codes_cannot_smuggle_numbers_or_text(code):
    (clean,) = sanitize_feedback([{"id": "c", "status": "kept", "reason_code": code}])
    assert clean["reason_code"] == "kept_unspecified"


def test_unknown_status_becomes_rejected():
    (clean,) = sanitize_feedback([{"id": "c", "status": "promising", "reason_code": "x"}])
    assert clean["status"] == "rejected"


@pytest.mark.integration
def test_leaking_scores_raises_false_positives_and_the_guard_prevents_it():
    """Spec §5 acceptance test, shown failing when the guard is broken.

    Same proposer, same pure-null data. Leaky channel = guard removed.
    """
    result = run_experiment(replicates=20)
    leaky = result["summary"]["leaky"]
    guarded = result["summary"]["guarded"]
    assert guarded["fpr"] <= 0.10
    assert leaky["fpr"] >= guarded["fpr"] + 0.15
    assert leaky["mean_delta"] > guarded["mean_delta"]
