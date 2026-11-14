from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np
from scipy import stats

from .utils import rng_for


# ---------------------------------------------------------------------------
# Core rank statistics
# ---------------------------------------------------------------------------

def _clean(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    mask = ~np.isnan(s)
    return y[mask].astype(int), s[mask]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve with average ranks for ties."""
    y, s = _clean(y, s)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = stats.rankdata(s)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_from_ranks(ranks: np.ndarray, positive_mask: np.ndarray) -> float:
    n1 = int(positive_mask.sum())
    n0 = int(len(positive_mask) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[positive_mask].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def spearman(y: np.ndarray, s: np.ndarray) -> float:
    y, s = _clean(y, s)
    if len(np.unique(y)) < 2 or len(np.unique(s)) < 2:
        return float("nan")
    rho, _ = stats.spearmanr(s, y)
    return float(rho)


# ---------------------------------------------------------------------------
# Group structure
# ---------------------------------------------------------------------------

def group_indices(groups: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        out.setdefault(str(g), []).append(i)
    return {k: np.asarray(v, dtype=int) for k, v in sorted(out.items())}


def evaluable_groups(y: np.ndarray, groups: np.ndarray) -> list[str]:
    out = []
    for g, idx in group_indices(groups).items():
        yy = y[idx]
        if len(np.unique(yy)) == 2:
            out.append(g)
    return out


def per_group_delta(y: np.ndarray, groups: np.ndarray, s_cand: np.ndarray, s_base: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for g, idx in group_indices(groups).items():
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        a = auc(yy, s_cand[idx])
        b = auc(yy, s_base[idx])
        if not (np.isnan(a) or np.isnan(b)):
            out[g] = float(a - b)
    return out


def group_bootstrap_deltas(
    y: np.ndarray,
    groups: np.ndarray,
    s_cand: np.ndarray,
    s_base: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Bootstrap resampling *groups*, never units. Paired by construction."""
    gi = group_indices(groups)
    keys = list(gi.keys())
    if len(keys) < 2:
        return np.array([], dtype=float)
    rng = rng_for("group_bootstrap", seed, len(keys), n_boot)
    deltas: list[float] = []
    # Cap retries so pathological data cannot loop forever.
    for _ in range(n_boot):
        for _attempt in range(20):
            pick = rng.integers(0, len(keys), size=len(keys))
            idx = np.concatenate([gi[keys[i]] for i in pick])
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            a = auc(yy, s_cand[idx])
            b = auc(yy, s_base[idx])
            if not (np.isnan(a) or np.isnan(b)):
                deltas.append(float(a - b))
                break
    return np.asarray(deltas, dtype=float)


def bootstrap_summary(deltas: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    if len(deltas) == 0:
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "se": float("nan"),
            "p_one_sided": float("nan"),
            "p_two_sided": float("nan"),
            "n_boot": 0,
        }
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_one = float((1 + np.sum(deltas <= 0)) / (len(deltas) + 1))
    p_two = float(min(1.0, 2 * min(p_one, 1 - p_one + 1 / (len(deltas) + 1))))
    return {
        "delta": float(np.mean(deltas)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "se": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "n_boot": int(len(deltas)),
    }


# ---------------------------------------------------------------------------
# Secondary metrics
# ---------------------------------------------------------------------------

def top_k_per_group(
    y: np.ndarray,
    groups: np.ndarray,
    s: np.ndarray,
    k: int = 3,
) -> dict[str, float]:
    """Hit rate in each group's top-k, and enrichment over the group base rate.

    This is the endpoint-substitution metric: it is robust where a global mean
    AUC is noisy, and it is usually what the decision-maker actually wanted.
    """
    hits = 0
    picked = 0
    base_hits = 0.0
    base_n = 0
    for g, idx in group_indices(groups).items():
        yy = y[idx]
        ss = s[idx]
        if len(yy) == 0:
            continue
        order = np.argsort(-ss, kind="mergesort")[: min(k, len(ss))]
        hits += int(yy[order].sum())
        picked += len(order)
        base_hits += float(yy.sum()) / len(yy) * len(order)
        base_n += len(order)
    if picked == 0:
        return {"top_k_hit_rate": float("nan"), "top_k_enrichment": float("nan")}
    hit_rate = hits / picked
    base_rate = base_hits / base_n if base_n else float("nan")
    return {
        "top_k_hit_rate": float(hit_rate),
        "top_k_enrichment": float(hit_rate / base_rate) if base_rate and base_rate > 0 else float("nan"),
    }


def mean_pct_rank(y: np.ndarray, groups: np.ndarray, s: np.ndarray) -> float:
    vals: list[float] = []
    for _, idx in group_indices(groups).items():
        yy = y[idx]
        if yy.sum() == 0:
            continue
        ss = s[idx]
        ranks = stats.rankdata(ss)
        # percentile in (0, 1], higher means the positive was ranked higher
        pct = (ranks - 1) / max(1, len(ss) - 1)
        vals.extend(pct[yy == 1].tolist())
    return float(np.mean(vals)) if vals else float("nan")


def frac_improved(per_group: dict[str, float]) -> float:
    if not per_group:
        return float("nan")
    return float(np.mean([v > 0 for v in per_group.values()]))


def compute_metrics(
    y: np.ndarray,
    groups: np.ndarray,
    s_cand: np.ndarray,
    s_base: np.ndarray,
    top_k: int = 3,
) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    a_c = auc(y, s_cand)
    a_b = auc(y, s_base)
    pg = per_group_delta(y, groups, s_cand, s_base)
    tk = top_k_per_group(y, groups, s_cand, top_k)
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "n_groups": int(len(group_indices(groups))),
        "n_groups_evaluable": int(len(pg)),
        "auc_candidate": a_c,
        "auc_baseline": a_b,
        "delta_auc": float(a_c - a_b) if not (np.isnan(a_c) or np.isnan(a_b)) else float("nan"),
        "top_k": int(top_k),
        **tk,
        "frac_improved": frac_improved(pg),
        "mean_pct_rank": mean_pct_rank(y, groups, s_cand),
        "rank_coef": spearman(y, s_cand),
    }


# ---------------------------------------------------------------------------
# Fold aggregation
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    candidate_id: str
    mean_delta: float
    weighted_delta: float
    fold_deltas: list[float]
    fold_weights: list[float]
    stability: float
    n_folds: int
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def aggregate_folds(fold_rows: list[dict[str, Any]], candidate_id: str) -> CandidateResult:
    weights = []
    deltas = []
    metrics = []
    for r in fold_rows:
        m = r["metrics"]
        w = max(1.0, m["n_positive"] * m["n_negative"])
        weights.append(w)
        deltas.append(m["delta_auc"])
        metrics.append(m)
    good = [(d, w) for d, w in zip(deltas, weights) if not np.isnan(d)]
    if good:
        ds = np.array([d for d, _ in good], dtype=float)
        ws = np.array([w for _, w in good], dtype=float)
        mean = float(np.average(ds, weights=ws))
        stability = float(np.mean(ds > 0))
    else:
        ds = np.array([], dtype=float)
        ws = np.array([], dtype=float)
        mean = float("nan")
        stability = 0.0
    return CandidateResult(
        candidate_id=candidate_id,
        mean_delta=mean,
        weighted_delta=mean,
        fold_deltas=[float(d) if not np.isnan(d) else None for d in ds],
        fold_weights=[float(w) for w in ws],
        stability=stability,
        n_folds=len(ds),
        metrics={
            "mean_auc_candidate": float(np.mean([m["auc_candidate"] for m in metrics if not np.isnan(m["auc_candidate"])]))
            if metrics else float("nan"),
            "mean_top_k_enrichment": float(
                np.mean(
                    [
                        m["top_k_enrichment"]
                        for m in metrics
                        if not np.isnan(m.get("top_k_enrichment", float("nan")))
                    ]
                )
            )
            if metrics
            else float("nan"),
        },
    )
