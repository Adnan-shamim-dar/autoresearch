"""OC-table summarisation and publishing (no calibration run required)."""

from __future__ import annotations

import json
import math

import pytest

from autoresearch.calibration import publish_oc_table, summarize_replicates


def _rep(effect, *, affirmed=False, confirmed=False, fp_alpha=False, tier="exploratory", ok=True):
    return {
        "ok": ok,
        "target_effect": effect,
        "true_effect": effect,
        "delta": effect,
        "p_confirmation": 0.5,
        "p_selection": 0.5,
        "detected_affirmed": affirmed,
        "detected_confirmed": confirmed,
        "false_positive_alpha": fp_alpha,
        "tier": tier,
    }


def test_detect_is_promotion_rate_and_confirmed_is_stricter():
    rows = [
        _rep(0.1, affirmed=True, confirmed=True),
        _rep(0.1, affirmed=True, confirmed=False),
        _rep(0.1, affirmed=False),
        _rep(0.1, affirmed=False),
    ]
    out = summarize_replicates(rows, (0.1,))
    (row,) = out["rows"]
    assert row["n"] == 4
    assert row["detect"] == pytest.approx(0.5)
    assert row["detect_affirmed"] == row["detect"]
    assert row["detect_confirmed"] == pytest.approx(0.25)
    assert row["detect_ci_low"] < row["detect"] < row["detect_ci_high"]


def test_fpr_comes_from_null_rows_only():
    rows = [
        _rep(0.0, fp_alpha=True),
        _rep(0.0),
        _rep(0.0),
        _rep(0.0, affirmed=True),
        _rep(0.1, affirmed=True, fp_alpha=True),
    ]
    out = summarize_replicates(rows, (0.0, 0.1))
    assert out["fpr_alpha"] == pytest.approx(0.25)
    assert out["fpr_decision"] == pytest.approx(0.25)
    for row in out["rows"]:
        assert row["fpr"] == pytest.approx(0.25)


def test_failed_replicates_and_missing_effects_are_skipped():
    rows = [_rep(0.1, affirmed=True), _rep(0.1, affirmed=True, ok=False)]
    out = summarize_replicates(rows, (0.05, 0.1))
    assert [r["effect"] for r in out["rows"]] == [0.1]
    assert out["rows"][0]["n"] == 1
    assert math.isnan(out["fpr_alpha"])


def test_power_refusals_are_counted():
    rows = [_rep(0.03, tier="power_refusal"), _rep(0.03, tier="power_refusal"), _rep(0.03)]
    (row,) = summarize_replicates(rows, (0.03,))["rows"]
    assert row["power_refusals"] == 2


def test_publish_rebuilds_rows_and_markdown(tmp_path):
    cal = tmp_path / "calibration"
    cal.mkdir()
    payload = {
        "generated_at": "2026-01-01T00:00:00Z",
        "domain": "test",
        "parameters": {"effects": [0.0, 0.1], "reps_per_effect": 2, "candidates": 4},
        "rows": [{"stale": True}],
        "replicates": [_rep(0.0), _rep(0.0), _rep(0.1, affirmed=True), _rep(0.1, affirmed=True, confirmed=True)],
        "method": "unit test",
        "limitations": ["synthetic"],
    }
    (cal / "oc_table.json").write_text(json.dumps(payload), encoding="utf-8")

    out = publish_oc_table(tmp_path)

    assert [r["effect"] for r in out["rows"]] == [0.0, 0.1]
    on_disk = json.loads((cal / "oc_table.json").read_text(encoding="utf-8"))
    assert on_disk["rows"] == out["rows"]
    assert "summarized_at" in on_disk
    md = (cal / "oc_table.md").read_text(encoding="utf-8")
    assert "| 0.100 | 2 |" in md
    assert "- synthetic" in md


def test_published_table_in_repo_is_consistent():
    """The checked-in OC table must re-summarise to the rows it publishes."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "calibration" / "oc_table.json"
    if not path.exists():
        pytest.skip("no published OC table")
    payload = json.loads(path.read_text(encoding="utf-8"))
    effects = tuple(float(e) for e in payload["parameters"]["effects"])
    fresh = summarize_replicates(payload["replicates"], effects)["rows"]
    assert len(fresh) == len(payload["rows"])
    for a, b in zip(fresh, payload["rows"]):
        assert a["detect"] == pytest.approx(b["detect"])
        assert a["fpr"] == pytest.approx(b["fpr"])
