"""Real-data (plasmode) calibration: planting, splitting, and one replicate."""

from __future__ import annotations

import pytest

from autoresearch.calibration_real import contract_dict, plant, run_one_real, split
from autoresearch.wdbc import load_frame


@pytest.mark.parametrize("target", [0.0, 0.06, 0.15])
def test_plant_realises_the_requested_effect_on_real_features(target):
    frame = load_frame()
    planted, realised = plant(frame, target, seed=3)
    assert realised == pytest.approx(target, abs=0.005)
    assert 0.3 < planted["malignant"].mean() < 0.45
    # Measurements are untouched: only the outcome is re-drawn.
    assert planted.drop(columns="malignant").equals(frame.drop(columns="malignant"))


def test_split_is_stratified_and_disjoint():
    planted, _ = plant(load_frame(), 0.06, seed=3)
    disc, conf = split(planted, seed=3)
    assert len(disc) + len(conf) == len(planted)
    assert set(disc["tumour_id"]).isdisjoint(conf["tumour_id"])
    assert abs(disc["malignant"].mean() - conf["malignant"].mean()) < 0.02


def test_contract_drops_sign_commitments_and_prices_the_bank():
    raw = contract_dict(confirmation_n=228, k_candidates=32)
    assert raw["power"]["k_candidates"] == 32
    assert not raw["commitments"]


@pytest.mark.integration
def test_null_replicate_on_real_features_is_not_promoted_and_scan_passes():
    """No-group data has no group hash, so the leakage scan must not report overlap."""
    row = run_one_real({"seed": 12, "target_effect": 0.0, "candidates": 16, "permutations": 19})
    assert row["ok"]
    assert not row["detected_affirmed"]
    assert row["adversary_results"]["leakage_scan"] == "pass"
