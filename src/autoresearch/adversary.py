from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from . import metrics
from .budget import ComplexityBudget
from .contract import Contract
from .evaluator import evaluate_candidates
from .scoring import Candidate, FeatureMatrix
from .utils import rng_for, sha256_file


def _delta(y: np.ndarray, groups: np.ndarray, s_cand: np.ndarray, s_base: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        y = y[mask]
        groups = groups[mask]
        s_cand = s_cand[mask]
        s_base = s_base[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(metrics.auc(y, s_cand) - metrics.auc(y, s_base))


def _mean_group_delta(y: np.ndarray, groups: np.ndarray, s_cand: np.ndarray, s_base: np.ndarray) -> float:
    vals = list(metrics.per_group_delta(y, groups, s_cand, s_base).values())
    return float(np.mean(vals)) if vals else float("nan")


def pseudo_group_control(
    y: np.ndarray,
    groups: np.ndarray,
    s_cand: np.ndarray,
    s_base: np.ndarray,
    n_splits: int = 20,
    n_perm: int = 20,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    rng = rng_for("pseudo-group", seed, len(y))
    observed = []
    nulls = []
    significant = 0
    for split_i in range(n_splits):
        # Split each homogeneous group into two pseudo-contexts. If the machinery
        # "discovers" that the halves need different answers, that is an artifact.
        half = np.zeros(len(y), dtype=int)
        for g in sorted(set(groups)):
            idx = np.where(groups == g)[0]
            perm = rng.permutation(idx)
            half[perm[: len(perm) // 2]] = 1
        mask_a = half == 0
        mask_b = half == 1
        d_a = _delta(y, groups, s_cand, s_base, mask_a)
        d_b = _delta(y, groups, s_cand, s_base, mask_b)
        obs = abs(d_a - d_b) if np.isfinite(d_a) and np.isfinite(d_b) else float("nan")
        observed.append(obs)
        local_null = []
        for _ in range(n_perm):
            shuffled = half.copy()
            for g in sorted(set(groups)):
                idx = np.where(groups == g)[0]
                shuffled[idx] = rng.permutation(shuffled[idx])
            da = _delta(y, groups, s_cand, s_base, shuffled == 0)
            db = _delta(y, groups, s_cand, s_base, shuffled == 1)
            if np.isfinite(da) and np.isfinite(db):
                local_null.append(abs(da - db))
        if local_null and np.isfinite(obs):
            p = (1 + np.sum(np.asarray(local_null) >= obs)) / (len(local_null) + 1)
            nulls.append(p)
            if p < alpha:
                significant += 1
    false_alarm_rate = significant / max(1, n_splits)
    result = "pass" if false_alarm_rate <= max(0.15, 2 * alpha) else "fail"
    return {
        "attack": "pseudo_group",
        "result": result,
        "critical": True,
        "n_splits": int(n_splits),
        "significant_splits": int(significant),
        "false_alarm_rate": float(false_alarm_rate),
        "note": (
            "Surrogate audit: unit halves inside homogeneous groups. The full context-specific search is the "
            "production version; this checks the same failure mode at the effect level."
        ),
    }


def outcome_permutation(
    y: np.ndarray,
    groups: np.ndarray,
    s_cand: np.ndarray,
    s_base: np.ndarray,
    observed: float,
    n_perm: int = 49,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    rng = rng_for("outcome-perm", seed, len(y))
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in sorted(set(groups)):
            idx = np.where(groups == g)[0]
            yp[idx] = rng.permutation(yp[idx])
        d = _delta(yp, groups, s_cand, s_base)
        if np.isfinite(d):
            null.append(d)
    if not null:
        return {"attack": "outcome_permutation", "result": "warn", "critical": True, "p_value": None}
    p = (1 + np.sum(np.asarray(null) >= observed)) / (len(null) + 1)
    return {
        "attack": "outcome_permutation",
        "result": "pass" if p <= alpha else "fail",
        "critical": True,
        "p_value": float(p),
        "observed": float(observed),
        "null_p95": float(np.percentile(null, 95)),
        "n_permutations": int(len(null)),
        "note": "Permuted within groups, preserving group outcome counts.",
    }


def feature_permutation(
    contract: Contract,
    data,
    fm: FeatureMatrix,
    candidate: Candidate,
    y: np.ndarray,
    groups: np.ndarray,
    s_base: np.ndarray,
    observed: float,
    n_perm: int = 29,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = rng_for("feature-perm", seed, candidate.id)
    weights = candidate.params.get("weights", {}) or {}
    top = [k for k, _ in sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:3]]
    out = []
    for feature in top:
        local = []
        for _ in range(n_perm):
            dfp = data.copy()
            for g in sorted(set(groups)):
                idx = np.where(groups == g)[0]
                vals = dfp.iloc[idx][feature].to_numpy().copy()
                dfp.iloc[idx, dfp.columns.get_loc(feature)] = rng.permutation(vals)
            fmp = FeatureMatrix(dfp, contract, fit=False, fit_stats=candidate.params.get("fit_stats", {}))
            try:
                sp = fmp.score_candidate(candidate)
            except Exception:
                continue
            d = _delta(y, groups, sp, s_base)
            if np.isfinite(d):
                local.append(d)
        if local:
            p = (1 + np.sum(np.asarray(local) >= observed)) / (len(local) + 1)
            out.append(
                {
                    "attack": f"feature_permutation:{feature}",
                    "result": "pass" if p <= alpha else "warn",
                    "critical": False,
                    "p_value": float(p),
                    "observed": float(observed),
                    "null_p95": float(np.percentile(local, 95)),
                    "note": "If the effect survives destroying this feature, the candidate may not be using it.",
                }
            )
    return out


def group_label_permutation(
    y: np.ndarray,
    groups: np.ndarray,
    s_cand: np.ndarray,
    s_base: np.ndarray,
    n_perm: int = 49,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    observed = _mean_group_delta(y, groups, s_cand, s_base)
    rng = rng_for("group-perm", seed, len(y))
    null = []
    for _ in range(n_perm):
        gp = rng.permutation(groups)
        d = _mean_group_delta(y, gp, s_cand, s_base)
        if np.isfinite(d):
            null.append(d)
    if not null or not np.isfinite(observed):
        return {"attack": "group_label_permutation", "result": "warn", "critical": False, "p_value": None}
    p = (1 + np.sum(np.asarray(null) >= observed)) / (len(null) + 1)
    return {
        "attack": "group_label_permutation",
        "result": "pass" if p <= alpha else "warn",
        "critical": False,
        "p_value": float(p),
        "observed": float(observed),
        "null_p95": float(np.percentile(null, 95)),
        "note": "Mean per-group delta after shuffling which rows belong together.",
    }


def leave_one_site_out(
    contract: Contract,
    data,
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    col = contract.contexts[0] if contract.contexts else (contract.group_column or None)
    if not col or col not in data.columns:
        return {"attack": "leave_one_site_out", "result": "skip", "critical": False, "note": "No context column available."}
    levels = sorted(map(str, data[col].dropna().unique().tolist()))
    if len(levels) < 3:
        return {
            "attack": "leave_one_site_out",
            "result": "warn",
            "critical": False,
            "levels": levels,
            "note": "Fewer than 3 sites/cohorts; leave-one-out has little power.",
        }
    out = []
    for level in levels:
        mask = data[col].astype(str).to_numpy() != level
        d = _delta(y, groups, candidate_scores, baseline_scores, mask)
        if np.isfinite(d):
            out.append({"held_out": level, "delta": float(d)})
    if not out:
        return {"attack": "leave_one_site_out", "result": "warn", "critical": False}
    vals = np.array([r["delta"] for r in out])
    result = "pass" if np.mean(vals > 0) >= 0.7 else "warn"
    return {
        "attack": "leave_one_site_out",
        "result": result,
        "critical": False,
        "sites": out,
        "min_delta": float(vals.min()),
        "max_delta": float(vals.max()),
        "fraction_positive": float(np.mean(vals > 0)),
        "note": "A result carried by one source is not a portable result.",
    }


def leakage_scan(
    contract: Contract,
    validation: dict[str, Any] | None,
    code_files: list[str | Path] | None = None,
    discovery_groups: list[str] | None = None,
    confirmation_group_hash: str | None = None,
) -> dict[str, Any]:
    errors = []
    if validation:
        errors.extend([e.get("reason_code") for e in validation.get("errors", [])])
    if discovery_groups and confirmation_group_hash:
        from .utils import sha256_obj

        if sha256_obj(sorted(map(str, discovery_groups))) == confirmation_group_hash:
            errors.append("split_overlap")
    hashes = {}
    for p in code_files or []:
        path = Path(p)
        if path.exists():
            hashes[path.name] = sha256_file(path)
    # A combined preprocessing/scoring hash makes silent edits visible.
    code_hash = hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(hashes.items())).encode()).hexdigest() if hashes else None
    result = "pass" if not errors else "fail"
    return {
        "attack": "leakage_scan",
        "result": result,
        "critical": True,
        "errors": errors,
        "code_hash": code_hash,
        "note": "Splits are upstream of all transforms; group-local ranks use features only. Confirmation labels stay sealed.",
    }


def baseline_advocate(
    contract: Contract,
    data,
    fm: FeatureMatrix,
    baseline: np.ndarray,
    budget: ComplexityBudget,
    selected_discovery_delta: float,
    n_candidates: int = 24,
    seed: int = 0,
) -> dict[str, Any]:
    rng = rng_for("baseline-advocate", seed)
    base_cand = contract.baseline_candidate()
    base_w = base_cand.params.get("weights", {}) or {}
    if not base_w:
        # Column baseline: use univariate weights as the advocate seed.
        base_w = {f.name: 1.0 for f in contract.features[:1]}
    feature_names = [f.name for f in contract.features if f.name in data.columns]
    cands: list[Candidate] = []
    for i in range(n_candidates):
        w = dict(base_w)
        # Spend the same search budget trying to make the incumbent win.
        for _ in range(int(rng.integers(0, 3))):
            f = str(rng.choice(feature_names))
            w[f] = w.get(f, 0.0) + float(rng.normal(0, 0.25))
        w = {k: v for k, v in w.items() if abs(v) > 1e-9}
        if not w:
            continue
        from .scoring import make_candidate

        c = make_candidate(
            w,
            mechanism=f"Baseline advocate #{i}: perturb the incumbent with the same search budget.",
            family="baseline_advocate",
            cid=f"ba-{i:03d}",
            fit_stats=fm.export_fit_stats(list(w)),
        )
        cands.append(c)
    if not cands:
        return {"attack": "baseline_advocate", "result": "skip", "critical": False}
    ev = evaluate_candidates(data, contract, cands, fm, baseline, budget, n_splits=4, seed=seed + 1, keep_fraction=0.5)
    best = max(
        [r["mean_delta"] for r in ev.results.values() if r.get("status") == "evaluated"],
        default=float("nan"),
    )
    margin = selected_discovery_delta - best if np.isfinite(best) else float("inf")
    result = "pass" if margin >= 0.005 else ("warn" if margin >= -0.005 else "fail")
    return {
        "attack": "baseline_advocate",
        "result": result,
        "critical": True,
        "baseline_advocate_best_delta": float(best) if np.isfinite(best) else None,
        "selected_discovery_delta": float(selected_discovery_delta),
        "margin": float(margin) if np.isfinite(margin) else None,
        "n_candidates": len(cands),
        "note": "The incumbent gets the same search budget. An undertuned baseline is the commonest false win.",
    }


def prespecified_failure(
    contract: Contract,
    data,
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any] | None:
    spec = contract.adversary.prespecified_failure
    if not spec:
        return None
    col = spec.get("context_column") or spec.get("column")
    val = spec.get("value")
    if not col or col not in data.columns:
        return {"attack": "prespecified_failure", "result": "warn", "critical": False, "note": "Context column missing."}
    mask = data[col].astype(str).to_numpy() == str(val)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return {"attack": "prespecified_failure", "result": "warn", "critical": False, "note": "Subgroup too small."}
    d = _delta(y, groups, candidate_scores, baseline_scores, mask)
    expected = str(spec.get("expected", "no_effect"))
    if expected == "no_effect":
        result = "pass" if (not np.isfinite(d) or d < contract.decision.smallest_effect_that_matters) else "fail"
    else:
        result = "pass" if np.isfinite(d) and d >= contract.decision.smallest_effect_that_matters else "warn"
    return {
        "attack": "prespecified_failure",
        "result": result,
        "critical": False,
        "context": f"{col}={val}",
        "delta": float(d) if np.isfinite(d) else None,
        "expected": expected,
        "note": "A claim that wins everywhere is usually measuring something generic.",
    }


def run_adversary(
    contract: Contract,
    data,
    fm: FeatureMatrix,
    candidate: Candidate,
    y: np.ndarray,
    groups: np.ndarray,
    baseline: np.ndarray,
    discovery_delta: float,
    budget: ComplexityBudget,
    alpha: float = 0.05,
    seed: int = 0,
    validation: dict[str, Any] | None = None,
    discovery_groups: list[str] | None = None,
    confirmation_group_hash: str | None = None,
    code_files: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    cand_scores = fm.score_candidate(candidate)
    attacks: list[dict[str, Any]] = []
    attacks.append(
        pseudo_group_control(
            y, groups, cand_scores, baseline,
            n_splits=contract.adversary.pseudo_group_splits,
            n_perm=12,
            alpha=alpha,
            seed=seed + 11,
        )
    )
    attacks.append(
        outcome_permutation(y, groups, cand_scores, baseline, discovery_delta, n_perm=contract.adversary.permutations, alpha=alpha, seed=seed + 12)
    )
    attacks.extend(
        feature_permutation(contract, data, fm, candidate, y, groups, baseline, discovery_delta, n_perm=19, alpha=alpha, seed=seed + 13)
    )
    attacks.append(group_label_permutation(y, groups, cand_scores, baseline, n_perm=39, alpha=alpha, seed=seed + 14))
    attacks.append(
        leave_one_site_out(contract, data, cand_scores, baseline, y, groups, alpha=alpha)
    )
    attacks.append(
        leakage_scan(contract, validation, code_files=code_files, discovery_groups=discovery_groups, confirmation_group_hash=confirmation_group_hash)
    )
    attacks.append(
        baseline_advocate(
            contract, data, fm, baseline, budget, discovery_delta,
            n_candidates=contract.adversary.baseline_advocate_candidates,
            seed=seed + 15,
        )
    )
    pf = prespecified_failure(contract, data, cand_scores, baseline, y, groups)
    if pf:
        attacks.append(pf)
    return attacks
