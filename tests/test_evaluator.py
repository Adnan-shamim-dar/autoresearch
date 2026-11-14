from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from autoresearch.budget import compute_budget
from autoresearch.contract import Contract
from autoresearch.demo import generate_immunotherapy_data, make_contract_dict
from autoresearch.evaluator import (
    assign_folds,
    effective_independent,
    evaluate_candidates,
    permute_within_groups,
)
from autoresearch.proposer import Proposer
from autoresearch.scoring import FeatureMatrix

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _prepared(n_patients: int = 20, seed: int = 11):
    df = generate_immunotherapy_data(seed=seed, n_patients=n_patients)
    contract = Contract(
        make_contract_dict(effect=0.04, k_candidates=16, confirmation_n=len(df))
    )
    contract.validate_static()
    contract.attach_data(df)
    budget = compute_budget(contract, df)
    fm = FeatureMatrix(df, contract, fit=True)
    proposer = Proposer(contract=contract, data=df, fm=fm, budget=budget, seed=seed)
    candidates = [c for c in proposer.seed_candidates(n=16) if c is not None]
    return df, contract, budget, fm, candidates


def test_folds_are_group_disjoint_and_cover_every_row(small_frame):
    y = small_frame["immunogenic"].to_numpy()
    groups = small_frame["patient_id"].astype(str).to_numpy()
    folds = assign_folds(groups, y, n_splits=5, seed=4)
    assert len(folds) == 5
    all_idx = np.concatenate(folds)
    assert sorted(all_idx.tolist()) == list(range(len(small_frame)))
    seen: set[str] = set()
    for test in folds:
        g = set(groups[test])
        assert not (g & seen), "a group leaked across folds"
        seen |= g


def test_permutation_preserves_group_counts(small_frame):
    y = small_frame["immunogenic"].to_numpy()
    groups = small_frame["patient_id"].astype(str).to_numpy()
    rng = np.random.default_rng(0)
    perm = permute_within_groups(y, groups, rng)
    for g in np.unique(groups):
        idx = groups == g
        assert int(perm[idx].sum()) == int(y[idx].sum())
    assert not np.array_equal(perm, y)


def test_effective_independent_collapses_correlated_columns():
    rng = np.random.default_rng(2)
    base = rng.normal(size=200)
    x = np.column_stack([base, base + rng.normal(0, 1e-4, 200), rng.normal(size=200)])
    k, clusters = effective_independent(x, ["a", "b", "c"])
    assert k == 2
    assert ["a", "b"] in clusters


def test_evaluation_is_deterministic_and_feedback_is_categorical():
    df, contract, budget, fm, candidates = _prepared()
    baseline = df["baseline_score"].to_numpy()
    first = evaluate_candidates(df, contract, candidates, fm, baseline, budget, n_splits=5, seed=5)
    second = evaluate_candidates(df, contract, candidates, fm, baseline, budget, n_splits=5, seed=5)
    assert first.selected_id is not None
    assert first.selected_id == second.selected_id
    assert first.selected_result == second.selected_result
    assert first.effective_independent == second.effective_independent

    for result in first.results.values():
        feedback = result.get("feedback")
        assert feedback is not None
        assert set(feedback) == {"status", "reason_code"}
        assert "mean_delta" not in feedback
        assert feedback["status"] in {"kept", "rejected"}


def test_planted_signal_is_selected_over_baseline():
    df, contract, budget, fm, candidates = _prepared(n_patients=24, seed=13)
    baseline = df["baseline_score"].to_numpy()
    result = evaluate_candidates(df, contract, candidates, fm, baseline, budget, n_splits=5, seed=13)
    assert result.selected_result is not None
    assert result.selected_result["mean_delta"] > 0.05
    assert result.selected_result["stability"] >= 0.6


def test_two_processes_produce_the_same_serialized_result():
    code = """
import hashlib, json, sys
sys.path.insert(0, r'{src}')
from autoresearch.budget import compute_budget
from autoresearch.contract import Contract
from autoresearch.demo import generate_immunotherapy_data, make_contract_dict
from autoresearch.evaluator import evaluate_candidates
from autoresearch.proposer import Proposer
from autoresearch.scoring import FeatureMatrix
from autoresearch.utils import canonical_json

df = generate_immunotherapy_data(seed=19, n_patients=14)
c = Contract(make_contract_dict(effect=0.04, k_candidates=12, confirmation_n=len(df)))
c.validate_static(); c.attach_data(df)
b = compute_budget(c, df)
fm = FeatureMatrix(df, c, fit=True)
p = Proposer(contract=c, data=df, fm=fm, budget=b, seed=19)
cands = [x for x in p.seed_candidates(n=12) if x is not None]
ev = evaluate_candidates(df, c, cands, fm, df['baseline_score'].to_numpy(), b, n_splits=4, seed=19)
print(hashlib.sha256(canonical_json(ev.to_dict()).encode('utf-8')).hexdigest())
""".format(src=str(SRC))
    hashes = []
    for hashseed in ("0", "12345"):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        env["PYTHONHASHSEED"] = hashseed
        out = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
        hashes.append(out.strip())
    assert hashes[0] == hashes[1]


def _null_for(df, contract, budget, fm, candidates, n_perm=19, seed=5):
    from autoresearch.evaluator import selection_aware_null

    baseline = df["baseline_score"].to_numpy()
    ev = evaluate_candidates(df, contract, candidates, fm, baseline, budget, n_splits=5, seed=seed)
    ids = [c.id for c in candidates if c.id not in ev.duplicate_map]
    comp = {c.id: c.complexity for c in candidates}
    y = df[contract.outcome_column].to_numpy().astype(int)
    groups = df[contract.group_column].astype(str).to_numpy()
    folds = assign_folds(groups, y, n_splits=5, seed=seed)
    return selection_aware_null(ev.score_matrix, baseline, y, groups, folds, ids, comp, n_perm, ("t", seed))


def test_selection_null_prices_a_real_signal():
    df, contract, budget, fm, candidates = _prepared(n_patients=60, seed=13)
    null = _null_for(df, contract, budget, fm, candidates)
    assert null["computed_on"].startswith("discovery")
    assert null["n_permutations"] == 19
    assert null["observed_max_stat"] > null["max_stat_p95"]
    assert null["p_value"] == 1 / 20


def test_selection_null_does_not_reward_noise():
    """Pure noise: the observed search maximum looks like a draw from the null."""
    df, contract, budget, fm, candidates = _prepared(n_patients=60, seed=13)
    rng = np.random.default_rng(99)
    groups = df[contract.group_column].astype(str).to_numpy()
    y = df[contract.outcome_column].to_numpy().astype(int)
    noisy = df.copy()
    noisy[contract.outcome_column] = permute_within_groups(y, groups, rng)
    null = _null_for(noisy, contract, budget, fm, candidates)
    # The best of K on noise is still positive; only the priced p tells the truth.
    assert null["observed_max_stat"] > 0
    assert null["p_value"] > 0.05
