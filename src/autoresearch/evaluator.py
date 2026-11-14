from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

from . import metrics
from .budget import ComplexityBudget
from .contract import Contract
from .scoring import Candidate, FeatureMatrix, candidate_complexity
from .utils import rng_for


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def assign_folds(
    groups: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> list[np.ndarray]:
    groups = np.asarray(groups).astype(str)
    y = np.asarray(y).astype(int)
    n_groups = len(set(groups))
    if n_groups == 1:
        # No declared clustering: units are independent; stratify on the outcome.
        from sklearn.model_selection import StratifiedKFold

        k = int(max(2, min(n_splits, np.bincount(y).min() if len(np.unique(y)) > 1 else 2)))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [test for _, test in skf.split(np.zeros(len(y)), y)]
    n_splits = int(max(2, min(n_splits, max(2, n_groups // 2))))
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [test for _, test in sgkf.split(np.zeros(len(y)), y, groups)]
        if all(len(f) > 0 for f in folds):
            return folds
    except Exception:
        pass

    # Deterministic fallback: sort groups by positive count, round-robin.
    order = []
    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        order.append((int(y[idx].sum()), len(idx), g))
    order.sort(key=lambda t: (-t[0], -t[1], t[2]))
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for i, (_, _, g) in enumerate(order):
        buckets[i % n_splits].append(g)
    folds = []
    for b in buckets:
        idx = np.where(np.isin(groups, b))[0]
        if len(idx):
            folds.append(idx)
    return folds


def permute_within_groups(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(y).copy()
    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        out[idx] = rng.permutation(out[idx])
    return out


# ---------------------------------------------------------------------------
# Fast rank/AUC machinery
# ---------------------------------------------------------------------------

def _fold_ranks(score_matrix: np.ndarray) -> np.ndarray:
    return stats.rankdata(score_matrix, axis=0)


def _delta_from_ranks(
    ranks: np.ndarray,
    y: np.ndarray,
    baseline_ranks: np.ndarray,
) -> np.ndarray:
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return np.full(ranks.shape[1], np.nan)
    pos = y.astype(bool)
    auc_c = (ranks[pos].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    auc_b = (baseline_ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return auc_c - auc_b


def _fold_weight(y: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    return float(max(1, n1 * n0))


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------

@dataclass
class SearchEvaluation:
    candidates: list[Candidate]
    results: dict[str, dict[str, Any]]
    fold_rows: list[dict[str, Any]]
    selected_id: str | None
    selected_candidate: Candidate | None
    selected_result: dict[str, Any] | None
    effective_independent: int
    clusters: list[list[str]]
    duplicate_map: dict[str, str]
    score_matrix: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "results": self.results,
            "fold_rows": self.fold_rows,
            "selected_id": self.selected_id,
            "selected": self.selected_candidate.to_dict() if self.selected_candidate else None,
            "selected_result": self.selected_result,
            "effective_independent": self.effective_independent,
            "clusters": self.clusters,
            "duplicate_map": self.duplicate_map,
        }


def effective_independent(
    score_matrix: np.ndarray,
    ids: list[str],
    threshold: float = 0.98,
) -> tuple[int, list[list[str]]]:
    if score_matrix.shape[1] <= 1:
        return int(score_matrix.shape[1]), [[i] for i in ids]
    x = np.asarray(score_matrix, dtype=float)
    x = x - np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = 1.0
    x = x / sd
    # Pairwise Pearson correlation; NaNs -> 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    parent = list(range(len(ids)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if abs(corr[i, j]) >= threshold:
                union(i, j)
    clusters: dict[int, list[str]] = {}
    for i, cid in enumerate(ids):
        clusters.setdefault(find(i), []).append(cid)
    out = [sorted(v) for v in clusters.values()]
    out.sort(key=lambda v: (-len(v), v[0]))
    return len(out), out


def evaluate_candidates(
    data,
    contract: Contract,
    candidates: list[Candidate],
    fm: FeatureMatrix,
    baseline: np.ndarray,
    budget: ComplexityBudget,
    n_splits: int = 5,
    seed: int = 0,
    keep_fraction: float = 0.4,
    dedupe_threshold: float = 0.9995,
) -> SearchEvaluation:
    y = data[contract.outcome_column].to_numpy().astype(int)
    groups = data[contract.group_column].astype(str).to_numpy() if contract.group_column else np.array(["__all__"] * len(data))
    folds = assign_folds(groups, y, n_splits=n_splits, seed=seed)

    # Filter by complexity budget first. Rejected candidates are still reported.
    valid: list[Candidate] = []
    invalid: dict[str, str] = {}
    seen_sig: dict[str, str] = {}
    scores_all: dict[str, np.ndarray] = {}
    for c in candidates:
        ok, why = budget.check(c)
        if not ok:
            invalid[c.id] = why
            continue
        try:
            s = fm.score_candidate(c)
        except Exception:
            invalid[c.id] = "rejected_unscoreable"
            continue
        if not np.all(np.isfinite(s)):
            invalid[c.id] = "rejected_non_finite"
            continue
        sig = ",".join(f"{k}:{round(v,3)}" for k, v in sorted((c.params.get("weights") or {}).items()))
        if sig in seen_sig:
            invalid[c.id] = "rejected_duplicate"
            continue
        seen_sig[sig] = c.id
        valid.append(c)
        scores_all[c.id] = s

    if not valid:
        return SearchEvaluation(
            candidates=candidates,
            results={c.id: {"status": "rejected", "reason_code": invalid.get(c.id, "rejected_invalid")} for c in candidates},
            fold_rows=[],
            selected_id=None,
            selected_candidate=None,
            selected_result=None,
            effective_independent=0,
            clusters=[],
            duplicate_map=invalid,
        )

    score_matrix = np.column_stack([scores_all[c.id] for c in valid])
    # Near-duplicate score vectors are not independent tests.
    dup_map: dict[str, str] = {}
    if score_matrix.shape[1] > 1:
        x = score_matrix - score_matrix.mean(axis=0)
        sd = score_matrix.std(axis=0)
        sd[sd == 0] = 1.0
        x = x / sd
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(x, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        for i in range(len(valid)):
            if valid[i].id in dup_map:
                continue
            for j in range(i + 1, len(valid)):
                if valid[j].id in dup_map:
                    continue
                if abs(corr[i, j]) >= dedupe_threshold:
                    dup_map[valid[j].id] = valid[i].id
    # Remove duplicates from the effective set but keep results as rejected.
    valid_eff = [c for c in valid if c.id not in dup_map]
    idx_eff = [valid.index(c) for c in valid_eff]
    score_matrix_eff = score_matrix[:, idx_eff]

    fold_rows: list[dict[str, Any]] = []
    # Initialize per-candidate fold deltas.
    fold_deltas: dict[str, list[float]] = {c.id: [] for c in valid_eff}
    fold_weights: dict[str, list[float]] = {c.id: [] for c in valid_eff}
    for fold_i, test_idx in enumerate(folds):
        yy = y[test_idx]
        if len(np.unique(yy)) < 2:
            continue
        ranks = _fold_ranks(score_matrix_eff[test_idx])
        base_ranks = stats.rankdata(baseline[test_idx])
        deltas = _delta_from_ranks(ranks, yy, base_ranks)
        w = _fold_weight(yy)
        for j, c in enumerate(valid_eff):
            d = float(deltas[j])
            fold_deltas[c.id].append(d)
            fold_weights[c.id].append(w)
            fold_rows.append(
                {
                    "fold": fold_i,
                    "candidate_id": c.id,
                    "n_test": int(len(test_idx)),
                    "n_positive": int(yy.sum()),
                    "delta": d,
                    "weight": w,
                }
            )

    results: dict[str, dict[str, Any]] = {}
    for c in candidates:
        if c.id in invalid:
            results[c.id] = {"status": "rejected", "reason_code": invalid[c.id], "complexity": candidate_complexity(c)}
            continue
        if c.id in dup_map:
            results[c.id] = {
                "status": "rejected",
                "reason_code": "rejected_duplicate",
                "duplicate_of": dup_map[c.id],
                "complexity": candidate_complexity(c),
            }
            continue
        ds = np.asarray(fold_deltas.get(c.id, []), dtype=float)
        ws = np.asarray(fold_weights.get(c.id, []), dtype=float)
        good = ~np.isnan(ds)
        if good.sum() == 0:
            results[c.id] = {"status": "rejected", "reason_code": "rejected_uncomputable", "complexity": candidate_complexity(c)}
            continue
        ds = ds[good]
        ws = ws[good]
        mean_delta = float(np.average(ds, weights=ws)) if ws.sum() > 0 else float(np.mean(ds))
        stability = float(np.mean(ds > 0))
        results[c.id] = {
            "status": "evaluated",
            "mean_delta": mean_delta,
            "stability": stability,
            "n_folds": int(len(ds)),
            "fold_deltas": [float(v) for v in ds],
            "complexity": candidate_complexity(c),
        }

    # Categorical feedback only: rank internally, never expose a score.
    evaluated = [(cid, r) for cid, r in results.items() if r.get("status") == "evaluated"]
    evaluated.sort(key=lambda kv: (-kv[1]["mean_delta"], kv[1]["complexity"], kv[0]))
    keep_n = max(1, int(np.ceil(len(evaluated) * keep_fraction)))
    kept = set()
    for i, (cid, r) in enumerate(evaluated):
        if i < keep_n:
            r["feedback"] = {"status": "kept", "reason_code": "kept_top_fraction"}
            kept.add(cid)
        else:
            r["feedback"] = {"status": "rejected", "reason_code": "rejected_below_cut"}
    for cid, r in results.items():
        if "feedback" not in r:
            r["feedback"] = {"status": "rejected", "reason_code": r.get("reason_code", "rejected_invalid")}

    # Exactly one winner, deterministic tie-break.
    selected = None
    selected_result = None
    if evaluated:
        best = evaluated[0][0]
        by_id = {c.id: c for c in valid_eff}
        selected = by_id.get(best)
        selected_result = results.get(best)

    eff_k, clusters = effective_independent(score_matrix_eff, [c.id for c in valid_eff])
    duplicate_map = {**invalid, **dup_map}
    return SearchEvaluation(
        candidates=candidates,
        results=results,
        fold_rows=fold_rows,
        selected_id=selected.id if selected else None,
        selected_candidate=selected,
        selected_result=selected_result,
        effective_independent=eff_k,
        clusters=clusters,
        duplicate_map=duplicate_map,
        score_matrix=score_matrix_eff,
    )


def select_on_permuted_outcomes(
    score_matrix: np.ndarray,
    baseline: np.ndarray,
    y_perm: np.ndarray,
    folds: list[np.ndarray],
    candidate_ids: list[str],
    complexities: dict[str, int],
) -> tuple[str | None, float, list[float]]:
    fold_deltas: list[np.ndarray] = []
    weights: list[float] = []
    for test_idx in folds:
        yy = y_perm[test_idx]
        if len(np.unique(yy)) < 2:
            continue
        ranks = _fold_ranks(score_matrix[test_idx])
        base_ranks = stats.rankdata(baseline[test_idx])
        d = _delta_from_ranks(ranks, yy, base_ranks)
        fold_deltas.append(d)
        weights.append(_fold_weight(yy))
    if not fold_deltas:
        return None, float("nan"), []
    dmat = np.vstack(fold_deltas)
    w = np.asarray(weights)
    means = []
    for j in range(dmat.shape[1]):
        good = ~np.isnan(dmat[:, j])
        means.append(float(np.average(dmat[good, j], weights=w[good])) if good.any() else float("nan"))
    means_arr = np.asarray(means)
    if np.all(np.isnan(means_arr)):
        return None, float("nan"), []
    order = sorted(
        range(len(candidate_ids)),
        key=lambda j: (-(means_arr[j] if np.isfinite(means_arr[j]) else -1e9), complexities.get(candidate_ids[j], 99), candidate_ids[j]),
    )
    best = order[0]
    return candidate_ids[best], float(means_arr[best]), [float(v) for v in dmat[:, best]]


def interpretability_bill(
    data,
    contract: Contract,
    rule_scores: np.ndarray,
    baseline: np.ndarray,
    folds: list[np.ndarray],
    seed: int = 0,
) -> dict[str, Any]:
    """What the readable rule costs against an unconstrained model (spec §12.5).

    Both are scored on the same group-aware folds against the same baseline.
    The unconstrained model sees every contract feature with no complexity cap.
    A positive gap means readability is costing accuracy.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    y = data[contract.outcome_column].to_numpy().astype(int)
    names = [f.name for f in contract.features if f.name in data.columns]
    X = data[names].apply(lambda col: col.astype(float)).to_numpy()
    all_idx = np.arange(len(y))
    rule_d, model_d, weights = [], [], []
    for test_idx in folds:
        yy = y[test_idx]
        if len(np.unique(yy)) < 2:
            continue
        train_idx = np.setdiff1d(all_idx, test_idx)
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict_proba(X[test_idx])[:, 1]
        base_auc = metrics.auc(yy, baseline[test_idx])
        rule_d.append(metrics.auc(yy, rule_scores[test_idx]) - base_auc)
        model_d.append(metrics.auc(yy, pred) - base_auc)
        weights.append(_fold_weight(yy))
    if not weights:
        return {"rule_cv_delta": None, "unconstrained_cv_delta": None, "gap_vs_unconstrained": None}
    rule_cv = float(np.average(rule_d, weights=weights))
    model_cv = float(np.average(model_d, weights=weights))
    return {
        "rule_cv_delta": rule_cv,
        "unconstrained_cv_delta": model_cv,
        "gap_vs_unconstrained": model_cv - rule_cv,
        "unconstrained_model": f"HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05) on {len(names)} features",
    }


def selection_aware_null(
    score_matrix: np.ndarray,
    baseline: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: list[np.ndarray],
    candidate_ids: list[str],
    complexities: dict[str, int],
    n_permutations: int,
    rng_parts: tuple[Any, ...],
) -> dict[str, Any]:
    """Price the search: the best score the whole search reaches on noise.

    Spec §6: re-run the search on outcomes permuted within the contract's
    groups, record the best score achieved, repeat; the distribution of those
    maxima is the bar. Everything here runs on search (discovery) data, so it
    costs no sealed labels: the confirmation set still sees exactly one
    candidate. The candidate pool is held fixed across permutations (a
    surrogate for re-running the adaptive proposer, spec §12.1); the maximum is
    taken over every evaluated candidate, so the multiplicity of the whole pool
    is priced.
    """
    _cid, observed, _ = select_on_permuted_outcomes(score_matrix, baseline, y, folds, candidate_ids, complexities)
    maxima: list[float] = []
    for b in range(int(n_permutations)):
        rng = rng_for(*rng_parts, b)
        y_perm = permute_within_groups(y, groups, rng)
        cid, best, _ = select_on_permuted_outcomes(score_matrix, baseline, y_perm, folds, candidate_ids, complexities)
        if cid is not None and np.isfinite(best):
            maxima.append(float(best))
    summary: dict[str, Any] = {
        "statistic": "max cross-validated paired delta over the candidate pool",
        "computed_on": "discovery data only; no sealed labels",
        "n_candidates": int(len(candidate_ids)),
        "observed_max_stat": float(observed) if np.isfinite(observed) else None,
        "n_permutations": int(len(maxima)),
    }
    if not maxima or not np.isfinite(observed):
        summary.update({"p_value": None, "max_stat_p95": None, "null_mean": None, "null_min": None, "null_max": None, "null_max_stats": []})
        return summary
    arr = np.asarray(maxima, dtype=float)
    summary.update(
        {
            "p_value": float((1 + np.sum(arr >= observed)) / (len(arr) + 1)),
            "max_stat_p95": float(np.percentile(arr, 95)),
            "null_mean": float(np.mean(arr)),
            "null_min": float(np.min(arr)),
            "null_max": float(np.max(arr)),
            "null_max_stats": [float(v) for v in arr],
        }
    )
    return summary


def leave_one_group_out_delta(
    y: np.ndarray,
    groups: np.ndarray,
    s_cand: np.ndarray,
    s_base: np.ndarray,
) -> dict[str, Any]:
    full = metrics.auc(y, s_cand) - metrics.auc(y, s_base)
    deltas: dict[str, float] = {}
    for g in sorted(set(groups)):
        idx = np.where(groups != g)[0]
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        d = metrics.auc(yy, s_cand[idx]) - metrics.auc(yy, s_base[idx])
        deltas[str(g)] = float(d)
    if not deltas:
        return {"full_delta": float(full), "fraction_positive": float("nan"), "min_delta": None, "n_groups": 0}
    vals = np.array(list(deltas.values()))
    return {
        "full_delta": float(full),
        "fraction_positive": float(np.mean(vals > 0)),
        "min_delta": float(vals.min()),
        "max_delta": float(vals.max()),
        "n_groups": int(len(vals)),
        "deltas": deltas,
    }
