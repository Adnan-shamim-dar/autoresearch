"""Spec §9: the automated red team, including the pseudo-group control."""

from __future__ import annotations

import numpy as np
import pytest

from autoresearch import adversary, metrics
from autoresearch.adversary import context_heterogeneity, pseudo_group_control, within_group_permuter


def test_within_group_permuter_keeps_each_group_multiset():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 30, 600).astype(str)
    values = rng.normal(size=600)
    out = within_group_permuter(groups)(values, np.random.default_rng(1))
    assert not np.array_equal(out, values)
    for g in np.unique(groups):
        assert sorted(out[groups == g]) == sorted(values[groups == g])


def _clustered(seed: int = 0, n_groups: int = 120, per_group: int = 20):
    """Homogeneous data whose improvement varies strongly by patient.

    For half the patients the candidate is sharp, for the rest it is useless,
    so the uncertainty of any delta is dominated by which patients are sampled.
    Only a test that resamples patients sees that.
    """
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"p{i:03d}" for i in range(n_groups)], per_group)
    sharp = np.repeat(rng.random(n_groups) < 0.5, per_group)
    latent = rng.normal(size=len(groups))
    y = (latent + rng.normal(scale=0.7, size=len(groups)) > 0.6).astype(int)
    s_base = latent + rng.normal(scale=1.2, size=len(groups))
    s_cand = np.where(sharp, latent + rng.normal(scale=0.2, size=len(groups)), rng.normal(size=len(groups)))
    return y, groups, s_cand, s_base


class _Contract:
    contexts: list[str] = []


def test_heterogeneity_detects_a_real_context_difference():
    y, groups, s_cand, s_base = _clustered(seed=1)
    labels = np.where(np.arange(len(y)) < len(y) // 2, "A", "B")
    s_cand = np.where(labels == "A", s_base + 0.8 * (s_cand - s_base) + 0.5 * y, s_cand)
    rng = np.random.default_rng(7)
    s_mod = s_cand.copy()
    b = labels == "B"
    s_mod[b] = s_base[b] + rng.normal(scale=0.05, size=b.sum())  # no real improvement in context B
    res = context_heterogeneity(y, groups, s_mod, s_base, labels, n_boot=200, seed=2)
    assert res["p_value"] < 0.05
    assert res["levels"]["A"]["delta"] > res["levels"]["B"]["delta"]


def test_pseudo_group_control_passes_honest_machinery():
    y, groups, s_cand, s_base = _clustered(seed=3, per_group=60)
    out = pseudo_group_control(_Contract(), None, y, groups, s_cand, s_base, n_splits=40, seed=4)
    assert out["attack"] == "pseudo_group"
    assert out["result"] == "pass"
    assert out["false_alarm_rate"] <= 0.10


def test_pseudo_group_control_catches_machinery_that_ignores_clustering(monkeypatch):
    """Shown failing when the guard is broken.

    Guard: the context test bootstraps patients, never units. Break it so the
    bootstrap resamples units; SEs shrink, random halves look different, and
    the pseudo-group control must flag the machinery.
    """
    def unit_bootstrap(y, groups, s_cand, s_base, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            out.append(metrics.auc(y[idx], s_cand[idx]) - metrics.auc(y[idx], s_base[idx]))
        return np.asarray(out)

    y, groups, s_cand, s_base = _clustered(seed=3, per_group=60)
    monkeypatch.setattr(adversary.metrics, "group_bootstrap_deltas", unit_bootstrap)
    out = pseudo_group_control(_Contract(), None, y, groups, s_cand, s_base, n_splits=40, seed=4)
    assert out["result"] == "fail"
    assert out["false_alarm_rate"] > 0.2


@pytest.mark.parametrize("n_groups", [6])
def test_pseudo_group_skips_when_there_is_nothing_to_split(n_groups):
    y, groups, s_cand, s_base = _clustered(seed=0, n_groups=n_groups)
    out = pseudo_group_control(_Contract(), None, y, groups, s_cand, s_base)
    assert out["result"] == "skip"
