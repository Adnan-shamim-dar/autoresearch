from __future__ import annotations

import math

import numpy as np
from scipy import stats

from autoresearch.power import assess, assess_continuous, assess_spec


def test_not_answerable_at_realistic_effect_and_prescription_recovers_it():
    # 800 confirmation rows give a plain MDE of about 0.087, above the 0.04
    # effect that matters. This is the acceptance case in the build spec.
    assessment = assess_spec(
        n_rows=800,
        n_groups=160,
        prevalence=0.20,
        baseline_auc=0.70,
        effect=0.04,
        k_candidates=100,
    )
    assert assessment.verdict == "NOT_ANSWERABLE"
    assert 0.08 <= assessment.mde_plain <= 0.095
    assert assessment.mde_search > assessment.mde_plain
    assert assessment.required_n_search is not None
    assert assessment.required_n_search > 800
    assert "Collect about" in assessment.prescription

    # Simulate under the oracle's own sampling model: at the prescribed n the
    # search-priced test should recover a true 0.04 effect at roughly target power.
    n = int(assessment.required_n_search)
    se_at_n = assessment.se_delta * math.sqrt(800.0 / n)
    crit = stats.norm.ppf(1 - 0.05 / 100)
    rng = np.random.default_rng(2026)
    reps = 400
    detected = 0
    for _ in range(reps):
        estimate = rng.normal(0.04, se_at_n)
        detected += int(estimate / se_at_n > crit)
    power = detected / reps
    assert 0.72 <= power <= 0.88, power
    assert assessment.required_n_plain is not None
    assert assessment.required_n_plain < n


def test_answerable_when_effect_is_large():
    assessment = assess_spec(
        n_rows=4000,
        n_groups=400,
        prevalence=0.2,
        baseline_auc=0.7,
        effect=0.15,
        k_candidates=10,
    )
    assert assessment.verdict == "ANSWERABLE"
    assert assessment.max_k_answerable >= 10
    assert assessment.designs


def test_assess_on_real_frame_reports_curves_and_designs(contract, small_frame):
    assessment = assess(
        contract,
        small_frame,
        confirmation_n=600,
        confirmation_groups=60,
        k_candidates=16,
        bootstrap=40,
        seed=3,
    )
    assert assessment.verdict in {"ANSWERABLE", "ANSWERABLE_IF_K_LE", "NOT_ANSWERABLE"}
    assert assessment.mde_search >= assessment.mde_plain > 0
    assert assessment.prescription
    ids = {d["id"] for d in assessment.designs}
    assert {"single_candidate_auc", "best_of_k", "direction_aggregation", "partial_pooling", "sequential"} <= ids
    assert len(assessment.mde_curve["n"]) == len(assessment.mde_curve["series"]["K=1"])
    assert len(assessment.power_curve["effect"]) == len(assessment.power_curve["series"]["K=1"])
    assert assessment.diagnostics["method"] == "group_bootstrap_prespecified_reference"
    assert "label_use" in assessment.diagnostics


def test_continuous_power_refuses_two_observations():
    assessment = assess_continuous(n_per_group=1, sd=0.12, effect=0.10)
    assert assessment["verdict"] == "NOT_ANSWERABLE"
    assert assessment["required_n_per_group"] >= 2
    assert "Collect about" in assessment["prescription"]


def test_operating_point_dial_prices_each_alpha_in_decision_units():
    """Spec §12.4: a dial, each setting's cost in the decision's own units."""
    from autoresearch.power import operating_points

    ops = operating_points(effect=0.04, se=0.012, k=50, target_power=0.8)
    alphas = [o["alpha"] for o in ops]
    assert alphas == sorted(alphas) and 0.05 in alphas
    # Looser alpha: more wrong actions on noise, fewer missed real effects.
    wrong = [o["wrong_actions_per_100_null_decisions"] for o in ops]
    missed = [o["missed_real_effects_per_100"] for o in ops]
    assert wrong == sorted(wrong)
    assert missed == sorted(missed, reverse=True)
