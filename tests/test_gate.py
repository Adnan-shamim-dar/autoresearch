from __future__ import annotations

import pytest

from autoresearch.gate import assign_tier
from autoresearch.power import assess_spec


def _power():
    return assess_spec(
        n_rows=3873,
        n_groups=160,
        prevalence=0.2,
        baseline_auc=0.70,
        effect=0.04,
        k_candidates=100,
    )


def _confirmation(delta: float, ci_low: float, ci_high: float, p: float, stability: float = 0.9):
    return {
        "n": 3873,
        "n_groups": 160,
        "delta": delta,
        "point_delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_one_sided": p,
        "stability_confirmation": stability,
        "se": max(1e-6, (ci_high - ci_low) / (2 * 1.96)),
    }


def _attacks(ok: bool = True):
    result = "pass" if ok else "fail"
    return [
        {"attack": "leakage_scan", "result": result, "critical": True},
        {"attack": "pseudo_group", "result": result, "critical": True},
        {"attack": "baseline_advocate", "result": result, "critical": False},
    ]


def _tier(contract, confirmation, p_sel=0.01, stability_folds=1.0, attacks_ok=True, table=None):
    return assign_tier(
        contract=contract,
        candidate={"id": "c-test"},
        confirmation=confirmation,
        selection_null={"p_value": p_sel, "n_permutations": 99, "effective_independent": 20},
        discovery_stability={"fold_fraction_positive": stability_folds},
        adversary_results=_attacks(attacks_ok),
        power=_power(),
        alpha_run=0.05,
        calibration_table=table,
        budget_info=None,
    )


def test_gate_can_say_yes_at_a_large_effect(contract):
    decision = _tier(contract, _confirmation(0.12, 0.10, 0.14, 0.001))
    assert decision.tier == "confirmed"
    assert decision.verdict == "affirmed"
    assert decision.claim.startswith("Confirmed:")


def test_supported_when_stability_is_below_the_confirmed_bar(contract):
    decision = _tier(contract, _confirmation(0.12, 0.10, 0.14, 0.001, stability=0.75))
    assert decision.tier == "supported"
    assert decision.verdict == "affirmed"


def test_provisional_when_effect_is_below_the_decision_threshold(contract):
    decision = _tier(contract, _confirmation(0.03, 0.01, 0.05, 0.01))
    assert decision.tier == "provisional"
    assert decision.verdict == "not_affirmed"


def test_exploratory_when_selection_null_does_not_clear_alpha(contract):
    decision = _tier(contract, _confirmation(0.12, 0.10, 0.14, 0.001), p_sel=0.20)
    assert decision.tier == "exploratory"
    assert decision.verdict == "not_affirmed"


def test_artifact_risk_when_ci_includes_zero(contract):
    decision = _tier(contract, _confirmation(0.02, -0.01, 0.05, 0.20))
    assert decision.tier == "artifact_risk"
    assert decision.verdict == "not_affirmed"


def test_artifact_risk_when_a_critical_attack_fails(contract):
    decision = _tier(contract, _confirmation(0.12, 0.10, 0.14, 0.001), attacks_ok=False)
    assert decision.tier == "artifact_risk"
    assert decision.verdict == "not_affirmed"


def test_negative_claim_is_confirmed_when_upper_bound_rules_out_the_effect(contract):
    decision = _tier(contract, _confirmation(-0.01, -0.03, 0.02, 0.8))
    assert decision.tier == "confirmed"
    assert decision.verdict == "refused"
    assert decision.claim_type == "negative"


def test_gate_reports_promoted_detection_rate_from_legacy_table(contract):
    table = [
        {"effect": 0.0, "detect": 0.0, "detect_affirmed": 0.0, "fpr": 0.02},
        {"effect": 0.10, "detect": 0.0, "detect_affirmed": 0.85, "fpr": 0.02},
        {"effect": 0.15, "detect": 0.85, "detect_affirmed": 1.0, "fpr": 0.02},
    ]
    decision = _tier(contract, _confirmation(0.12, 0.10, 0.14, 0.001), table=table)
    assert decision.gate_oc["source"] == "calibration_interpolated"
    assert 0.85 < decision.gate_oc["detect"] < 1.0


def test_gate_uses_nearest_calibration_row_outside_the_grid(contract):
    table = [
        {"effect": 0.0, "detect": 0.0, "fpr": 0.02},
        {"effect": 0.15, "detect": 1.0, "fpr": 0.02},
    ]
    decision = _tier(contract, _confirmation(0.20, 0.17, 0.23, 0.001), table=table)
    assert decision.gate_oc["source"] == "calibration_nearest"
    assert decision.gate_oc["detect"] == 1.0


def test_gate_oc_falls_back_to_power_approximation_without_a_table(contract):
    decision = _tier(contract, _confirmation(0.12, 0.10, 0.14, 0.001))
    assert decision.gate_oc["source"] == "power_approximation"
    assert 0.0 <= decision.gate_oc["detect"] <= 1.0
