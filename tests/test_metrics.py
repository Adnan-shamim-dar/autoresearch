from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from autoresearch import metrics


def _toy(n_groups: int = 8, per_group: int = 12, seed: int = 3):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per_group)
    y = rng.integers(0, 2, size=n_groups * per_group)
    s = rng.normal(size=n_groups * per_group)
    return y, groups, s


def test_auc_agrees_with_reference_implementation():
    rng = np.random.default_rng(11)
    for _ in range(20):
        y = rng.integers(0, 2, size=257)
        s = rng.normal(size=257)
        assert abs(metrics.auc(y, s) - roc_auc_score(y, s)) < 1e-9


def test_auc_handles_ties_like_sklearn():
    y = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    s = np.array([0.5, 0.5, 0.7, 0.7, 0.7, 0.2, 0.9, 0.1])
    assert abs(metrics.auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_auc_ignores_nan_scores():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, np.nan, 0.4, 0.1])
    assert abs(metrics.auc(y, s) - roc_auc_score(y[[0, 2, 3]], s[[0, 2, 3]])) < 1e-12


def test_bootstrap_is_deterministic_and_clustered():
    y, groups, s = _toy()
    s_base = np.zeros(len(y))
    a = metrics.group_bootstrap_deltas(y, groups, s, s_base, n_boot=80, seed=5)
    b = metrics.group_bootstrap_deltas(y, groups, s, s_base, n_boot=80, seed=5)
    assert a.shape == b.shape
    assert np.array_equal(a, b)


def test_per_group_delta_is_zero_when_scores_identical():
    y, groups, s = _toy()
    deltas = metrics.per_group_delta(y, groups, s, s.copy())
    assert set(deltas.values()) == {0.0}


def test_top_k_enrichment_rewards_a_perfect_ranking():
    groups = np.array(["a"] * 10 + ["b"] * 10)
    y = np.array([1, 1, 1] + [0] * 7 + [1, 1, 1] + [0] * 7)
    s = np.array([3.0, 2.0, 1.0] + [0.0] * 7 + [3.0, 2.0, 1.0] + [0.0] * 7)
    out = metrics.top_k_per_group(y, groups, s, k=3)
    assert out["top_k_hit_rate"] == 1.0
    assert out["top_k_enrichment"] > 1.0


def test_mean_pct_rank_is_one_for_a_perfect_ranking():
    groups = np.array(["a"] * 5 + ["b"] * 5)
    y = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    s = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    # Min-max percentile within group: best positive 1.0, second-best 0.75.
    assert metrics.mean_pct_rank(y, groups, s) == 0.875


def test_bootstrap_summary_reports_ci_and_p_value():
    rng = np.random.default_rng(2)
    deltas = rng.normal(0.05, 0.02, size=500)
    out = metrics.bootstrap_summary(deltas)
    assert out["ci_low"] < 0.05 < out["ci_high"]
    assert 0.0 <= out["p_one_sided"] <= 1.0
    assert out["n_boot"] == 500
