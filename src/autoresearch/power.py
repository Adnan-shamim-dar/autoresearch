from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics
from .budget import compute_budget
from .contract import Contract
from .scoring import FeatureMatrix, Candidate, make_candidate, baseline_scores
from .utils import clamp


def _z(p: float) -> float:
    return float(stats.norm.ppf(p))


def _power_normal(effect: float, se: float, alpha: float, k: int = 1) -> float:
    if not np.isfinite(se) or se <= 0 or not np.isfinite(effect):
        return float("nan")
    crit = _z(1 - alpha / max(1, k))
    return float(clamp(stats.norm.cdf(effect / se - crit), 0.0, 1.0))


def _required_n(effect: float, se_at_n: float, n_at_se: float, alpha: float, k: int, power: float) -> float | None:
    if not np.isfinite(effect) or effect <= 0 or not np.isfinite(se_at_n) or se_at_n <= 0:
        return None
    mde = (_z(1 - alpha / max(1, k)) + _z(power)) * se_at_n
    return float(max(4, math.ceil(n_at_se * (mde / effect) ** 2)))


def _hanley_se(auc: float, n1: int, n0: int) -> float:
    auc = clamp(float(auc), 0.001, 0.999)
    q1 = auc / (2 - auc)
    q0 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n0 - 1) * (q0 - auc * auc)) / max(1, n1 * n0)
    return float(math.sqrt(max(1e-12, var)))


def _reference_candidate(data: pd.DataFrame, contract: Contract) -> Candidate | None:
    """A pre-specified reference candidate, fixed before outcomes are examined.

    Power should not depend on how strong the observed signal happens to be. The
    reference is an equal-weight, direction-aware combination of every declared
    feature. It uses no labels; transform statistics are feature-only. This makes
    the variance estimate design-based and stable under a true null, where a
    label-fitted reference would be noise and would inflate the variance.
    """
    fm = FeatureMatrix(data, contract, fit=True)
    weights = {f.name: 1.0 for f in contract.features if f.name in data.columns}
    if not weights:
        return None
    return make_candidate(
        weights=weights,
        mechanism="Pre-specified reference candidate for variance estimation: equal-weight, direction-aware combination of all declared features.",
        family="reference",
        cid="power-reference",
        fit_stats=fm.export_fit_stats(list(weights)),
    )


def _estimate_se_delta(data: pd.DataFrame, contract: Contract, bootstrap: int = 300, seed: int = 0) -> tuple[float, dict[str, Any]]:
    y = data[contract.outcome_column].to_numpy().astype(int)
    groups = data[contract.group_column].astype(str).to_numpy() if contract.group_column else np.array(["__all__"] * len(data))
    base = baseline_scores(data, contract)
    ref = _reference_candidate(data, contract)
    if ref is not None:
        fm = FeatureMatrix(data, contract, fit=True)
        ref_scores = fm.score_candidate(ref)
        deltas = metrics.group_bootstrap_deltas(y, groups, ref_scores, base, n_boot=bootstrap, seed=seed)
        if len(deltas) >= 10:
            se = float(np.std(deltas, ddof=1))
            return max(1e-6, se), {
                "method": "group_bootstrap_prespecified_reference",
                "reference_candidate": ref.to_dict(),
                "delta_mean": float(np.mean(deltas)),
                "n_boot": int(len(deltas)),
                "label_use": "none: reference weights and transform statistics are outcome-independent",
            }
    a = metrics.auc(y, base)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    se_auc = _hanley_se(a, n1, n0)
    return math.sqrt(2) * se_auc, {"method": "hanley_fallback", "se_auc": se_auc}


@dataclass
class Assessment:
    verdict: str
    effect_that_matters: float
    alpha: float
    target_power: float
    mde_plain: float
    mde_search: float
    max_k_answerable: int
    k_candidates: int
    confirmation_n: int
    confirmation_groups: int
    confirmation_prevalence: float
    baseline_auc: float
    se_delta: float
    required_n_plain: int | None
    required_n_search: int | None
    required_groups_search: int | None
    prescription: str
    designs: list[dict[str, Any]]
    mde_curve: dict[str, Any]
    power_curve: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    operating_points: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def operating_points(
    effect: float,
    se: float,
    k: int,
    target_power: float,
    alphas: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.10),
) -> list[dict[str, Any]]:
    """The alpha dial (spec §12.4), each setting priced in the decision's units.

    For every alpha: the search-priced MDE, the chance of detecting the effect
    that matters, and per 100 decisions where nothing works, how many times the
    decision's action would be taken on noise.
    """
    rows = []
    for a in alphas:
        rows.append(
            {
                "alpha": float(a),
                "mde_search": float((_z(1 - a / max(1, k)) + _z(target_power)) * se),
                "power_at_effect": float(_power_normal(effect, se, a, k)),
                "wrong_actions_per_100_null_decisions": float(100 * a),
                "missed_real_effects_per_100": float(100 * (1 - _power_normal(effect, se, a, k))),
            }
        )
    return rows


def _designs(
    effect: float,
    se: float,
    n_conf: int,
    groups: int,
    prevalence: float,
    alpha: float,
    target_power: float,
    k: int,
    top_k: int,
    median_group_size: float,
) -> list[dict[str, Any]]:
    mde_plain = (_z(1 - alpha) + _z(target_power)) * se
    mde_search = (_z(1 - alpha / max(1, k)) + _z(target_power)) * se
    req_plain = _required_n(effect, se, n_conf, alpha, 1, target_power)
    req_search = _required_n(effect, se, n_conf, alpha, k, target_power)
    power_plain = _power_normal(effect, se, alpha, 1)
    power_search = _power_normal(effect, se, alpha, k)
    designs: list[dict[str, Any]] = []

    designs.append(
        {
            "id": "single_candidate_auc",
            "name": "Pre-chosen candidate, paired AUC",
            "power_at_current_n": power_plain,
            "required_n": req_plain,
            "assumptions": ["One candidate fixed before confirmation", "Group-aware paired bootstrap"],
            "establishes": "Whether this one candidate beats the deployed baseline by the declared effect.",
            "does_not_establish": "That the candidate would have been found by a search; that other candidates are worse.",
            "method": "normal approximation to paired AUC difference",
        }
    )
    designs.append(
        {
            "id": "best_of_k",
            "name": f"Search then confirm one of K≈{k}",
            "power_at_current_n": power_search,
            "required_n": req_search,
            "assumptions": ["K independent candidates", "Selection rule fixed in advance", "Max-statistic correction"],
            "establishes": "That the winner of the declared search beats baseline after paying the search price.",
            "does_not_establish": "Independence if candidates are correlated; use effective independent K.",
            "method": "max-statistic / Bonferroni normal approximation",
        }
    )

    # Direction aggregation: K pre-committed directional predictions, one sign test.
    p_alt = clamp(0.5 + effect / 2.0, 0.5, 0.99)
    k_dir = max(4, min(5000, k))
    crit = int(stats.binom.ppf(1 - alpha, k_dir, 0.5)) if k_dir > 0 else 0
    power_dir = float(1 - stats.binom.cdf(crit - 1, k_dir, p_alt)) if k_dir else float("nan")
    req_dir = None
    for kk in range(4, 5001):
        cc = int(stats.binom.ppf(1 - alpha, kk, 0.5))
        if 1 - stats.binom.cdf(cc - 1, kk, p_alt) >= target_power:
            req_dir = kk
            break
    designs.append(
        {
            "id": "direction_aggregation",
            "name": "K directional predictions, one sign test",
            "power_at_current_n": power_dir,
            "required_n": req_dir,
            "units": "predictions rather than samples",
            "assumptions": [
                "K predictions committed before seeing outcomes",
                "Each has probability p=0.5+(effect/2) of being directionally right",
                "Predictions are exchangeable and not cherry-picked",
            ],
            "establishes": "A powered test built out of individually unpowered parts; a sign claim, not a magnitude.",
            "does_not_establish": "Effect size, or that any single prediction is individually significant.",
            "method": "exact binomial sign test",
        }
    )

    # Top-k enrichment / endpoint substitution.
    frac_top = clamp(top_k / max(1.0, median_group_size), 0.01, 1.0)
    n_top = max(5, int(round(n_conf * frac_top)))
    p0 = clamp(prevalence, 0.01, 0.99)
    p1 = clamp(p0 * (1.0 + max(0.0, effect)), p0, 0.99)
    if p1 > p0:
        crit_top = int(stats.binom.ppf(1 - alpha, n_top, p0))
        power_top = float(1 - stats.binom.cdf(crit_top - 1, n_top, p1))
        num = _z(1 - alpha) * math.sqrt(p0 * (1 - p0)) + _z(target_power) * math.sqrt(p1 * (1 - p1))
        req_top_n = math.ceil((num / (p1 - p0)) ** 2) if p1 > p0 else None
        req_top_rows = int(math.ceil(req_top_n / frac_top)) if req_top_n else None
    else:
        power_top = float("nan")
        req_top_rows = None
    designs.append(
        {
            "id": "endpoint_substitution_top_k",
            "name": f"Top-{top_k} hit rate per group (endpoint substitution)",
            "power_at_current_n": power_top,
            "required_n": req_top_rows,
            "units": "top-k slots",
            "assumptions": ["The decision is about enrichment at the top of each patient's list", "Hits are exchangeable within group"],
            "establishes": "Whether the candidate puts validated hits into the top-k more often than base rate.",
            "does_not_establish": "Global ranking quality, or performance outside the top-k.",
            "method": "binomial enrichment test",
        }
    )

    # Partial pooling: groups become the unit of analysis.
    design_effect = max(1.0, n_conf / max(1.0, groups * max(1.0, median_group_size)))
    between = max(0.05, effect)
    power_pool = float(clamp(stats.norm.cdf(between * math.sqrt(max(1, groups - 3)) - _z(1 - alpha)), 0, 1))
    req_groups_pool = int(math.ceil(((_z(1 - alpha) + _z(target_power)) / between) ** 2 + 3))
    designs.append(
        {
            "id": "partial_pooling",
            "name": "Partial pooling across groups",
            "power_at_current_n": power_pool,
            "required_n": None,
            "required_groups": req_groups_pool,
            "units": "groups",
            "assumptions": [
                "Groups are exchangeable after stratification",
                "The between-group spread parameter, not each group mean, is the question",
                "A hierarchical model is pre-specified",
            ],
            "establishes": "Whether effects vary across groups, borrowing strength from small groups; one estimand, no multiplicity.",
            "does_not_establish": "A group-specific answer for any particular group.",
            "method": "normal approximation to between-group spread",
        }
    )

    # Sequential design.
    designs.append(
        {
            "id": "sequential",
            "name": "Sequential design with a declared stopping rule",
            "power_at_current_n": float(power_plain * 0.98) if np.isfinite(power_plain) else float("nan"),
            "required_n": int(math.ceil(req_plain * 1.1)) if req_plain else None,
            "assumptions": ["Stopping boundaries declared before the first look", "Alpha spent over looks, not re-used"],
            "establishes": "The same claim with a legitimate early-stop option, at a modest sample-size cost.",
            "does_not_establish": "Anything if the analyst peeks without spending alpha.",
            "method": "O'Brien-Fleming-style inflation approximation",
        }
    )

    designs.append(
        {
            "id": "recover_discarded_information",
            "name": "Recover discarded information",
            "power_at_current_n": None,
            "required_n": None,
            "assumptions": ["Ordinal outcomes were collapsed to binary", "Repeated measures were averaged", "Censoring was dropped"],
            "establishes": "Free information the original design threw away can raise power without new samples.",
            "does_not_establish": "That the recovered variable is measurement-valid; check the source instrument.",
            "method": "design audit",
        }
    )
    designs.append(
        {
            "id": "stratify_match",
            "name": "Stratify or match on the confound",
            "power_at_current_n": None,
            "required_n": None,
            "assumptions": ["The confound is measured", "Strata have overlapping support", "The stratification is pre-specified"],
            "establishes": "Kills a confound that otherwise makes two explanations indistinguishable.",
            "does_not_establish": "A causal effect by itself; it only removes one alternative.",
            "method": "design audit",
        }
    )

    # Rank by required sample size where available, then by power.
    def rank_key(d: dict[str, Any]):
        req = d.get("required_n")
        return (0 if req is not None else 1, req if req is not None else 10**12, -(d.get("power_at_current_n") or 0))

    designs.sort(key=rank_key)
    return designs


def assess(
    contract: Contract,
    data: pd.DataFrame,
    confirmation_n: int | None = None,
    confirmation_groups: int | None = None,
    confirmation_prevalence: float | None = None,
    k_candidates: int | None = None,
    alpha: float | None = None,
    target_power: float | None = None,
    effect: float | None = None,
    bootstrap: int = 300,
    seed: int = 0,
) -> Assessment:
    y = data[contract.outcome_column].to_numpy().astype(int)
    n_rows = len(data)
    groups_arr = data[contract.group_column].astype(str).to_numpy() if contract.group_column else np.array(["__all__"] * n_rows)
    n_groups = int(len(set(groups_arr)))
    prevalence = float(confirmation_prevalence) if confirmation_prevalence is not None else float(y.mean())
    n_conf = int(confirmation_n or contract.power.confirmation_n or max(20, int(n_rows * 0.4)))
    g_conf = int(confirmation_groups or contract.power.confirmation_groups or max(2, int(n_groups * 0.4)))
    alpha = float(alpha if alpha is not None else (contract.power.alpha or contract.decision.alpha))
    target_power = float(target_power if target_power is not None else (contract.power.target_power or contract.decision.target_power))
    k = int(k_candidates if k_candidates is not None else contract.power.k_candidates)

    base_auc = metrics.auc(y, baseline_scores(data, contract))
    se_disc, se_diag = _estimate_se_delta(data, contract, bootstrap=bootstrap, seed=seed)
    budget = compute_budget(contract, data)
    deff = budget.design_effect
    pairs_disc = (y.sum() * (len(y) - y.sum())) / max(1, len(y))
    pairs_conf = (prevalence * (1 - prevalence) * n_conf) / deff
    scale = math.sqrt(max(1e-9, (pairs_disc / deff) / max(1e-9, pairs_conf)))
    se_conf = max(1e-6, se_disc * scale)

    mde_plain = (_z(1 - alpha) + _z(target_power)) * se_conf
    mde_search = (_z(1 - alpha / max(1, k)) + _z(target_power)) * se_conf
    effect = float(effect) if effect is not None else contract.decision.smallest_effect_that_matters
    max_k = 0
    if effect > 0 and se_conf > 0:
        z_eff = effect / se_conf - _z(target_power)
        if z_eff > 0:
            # sf, not 1 - cdf: the tail underflows to exactly zero for a
            # comfortably powered effect and would wrongly report K=0.
            tail = float(stats.norm.sf(z_eff))
            if tail > 0:
                max_k = int(min(alpha / tail, 1_000_000))
    if effect >= mde_search:
        verdict = "ANSWERABLE"
    elif effect >= mde_plain:
        verdict = "ANSWERABLE_IF_K_LE"
    else:
        verdict = "NOT_ANSWERABLE"

    req_plain = _required_n(effect, se_conf, n_conf, alpha, 1, target_power)
    req_search = _required_n(effect, se_conf, n_conf, alpha, k, target_power)
    req_groups = int(math.ceil(g_conf * (req_search / n_conf))) if req_search else None

    median_group_size = float(np.median([np.sum(groups_arr == g) for g in set(groups_arr)])) if n_groups else 1.0
    designs = _designs(effect, se_conf, n_conf, g_conf, prevalence, alpha, target_power, k, contract.top_k, median_group_size)

    if verdict == "NOT_ANSWERABLE":
        prescription = (
            f"The question as posed is underpowered. At {n_conf} confirmation rows / {g_conf} groups, "
            f"the minimum detectable effect is {mde_plain:.3f} before search and {mde_search:.3f} after pricing "
            f"K={k} candidates. The declared effect that matters is {effect:.3f}. "
            + (f"Collect about {req_search} rows / {req_groups} groups, or adopt a design in the advisor that buys power without new samples." if req_search else "No sample-size prescription is finite; change the estimand.")
        )
    elif verdict == "ANSWERABLE_IF_K_LE":
        prescription = (
            f"The pre-chosen-candidate test is powered (MDE {mde_plain:.3f}), but a search over K={k} is not "
            f"(MDE {mde_search:.3f}). Restrict the search to K≤{max_k} mechanistically distinct candidates, "
            f"or collect about {req_search} rows for the full search."
        )
    else:
        prescription = (
            f"Answerable: the declared effect {effect:.3f} exceeds the search-priced MDE {mde_search:.3f} "
            f"at {n_conf} rows / {g_conf} groups. Register the selection rule and spend confirmation once."
        )

    n_range = np.unique(np.round(np.linspace(max(10, n_conf * 0.25), max(20, n_conf * 4), 28)).astype(int))
    mde_curve = {
        "n": [int(v) for v in n_range],
        "series": {
            f"K={kk}": [float((_z(1 - alpha / max(1, kk)) + _z(target_power)) * se_conf * math.sqrt(n_conf / max(1, int(v)))) for v in n_range]
            for kk in (1, 10, 100, 1000)
        },
        "effect_that_matters": effect,
    }
    effect_grid = [round(float(v), 3) for v in np.linspace(0.0, max(0.25, effect * 3), 26)]
    power_curve = {
        "effect": effect_grid,
        "series": {
            f"K={kk}": [_power_normal(e, se_conf, alpha, kk) for e in effect_grid]
            for kk in (1, 10, 100, 1000)
        },
        "target_power": target_power,
    }

    warnings: list[str] = []
    if se_diag.get("method") not in {"group_bootstrap_delta", "group_bootstrap_prespecified_reference"}:
        warnings.append("Delta SE fell back to a Hanley-McNeil approximation; clustered variance may be understated.")
    if n_conf < 30:
        warnings.append("Fewer than 30 confirmation rows; the normal approximation is fragile.")
    if g_conf < 4:
        warnings.append("Fewer than 4 confirmation groups; group-aware inference is fragile.")
    if contract.confirmation_meta and contract.confirmation_meta.get("prevalence") is not None:
        p_meta = float(contract.confirmation_meta["prevalence"])
        if abs(p_meta - prevalence) > 0.15:
            warnings.append("Discovery and confirmation prevalence differ by more than 15 percentage points; transport is questionable.")

    return Assessment(
        verdict=verdict,
        effect_that_matters=effect,
        alpha=alpha,
        target_power=target_power,
        mde_plain=float(mde_plain),
        mde_search=float(mde_search),
        max_k_answerable=int(max_k),
        k_candidates=int(k),
        confirmation_n=int(n_conf),
        confirmation_groups=int(g_conf),
        confirmation_prevalence=float(prevalence),
        baseline_auc=float(base_auc),
        se_delta=float(se_conf),
        required_n_plain=int(req_plain) if req_plain else None,
        required_n_search=int(req_search) if req_search else None,
        required_groups_search=int(req_groups) if req_groups else None,
        prescription=prescription,
        designs=designs,
        mde_curve=mde_curve,
        power_curve=power_curve,
        diagnostics={
            **se_diag,
            "budget": budget.to_dict(),
            "se_discovery": se_disc,
            "design_effect": deff,
            "pairs_discovery": pairs_disc,
            "pairs_confirmation": pairs_conf,
            "prevalence_source": "confirmation_meta" if confirmation_prevalence is not None else "discovery",
        },
        warnings=warnings,
        operating_points=operating_points(effect, se_conf, k, target_power),
    )


def assess_spec(
    n_rows: int,
    n_groups: int,
    prevalence: float,
    baseline_auc: float,
    effect: float,
    k_candidates: int = 1,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> Assessment:
    """Power assessment when labels are unavailable, using a Hanley-McNeil SE."""
    n1 = max(2, int(round(n_rows * prevalence)))
    n0 = max(2, n_rows - n1)
    se_auc = _hanley_se(baseline_auc, n1, n0)
    se = math.sqrt(2) * se_auc
    mde_plain = (_z(1 - alpha) + _z(target_power)) * se
    mde_search = (_z(1 - alpha / max(1, k_candidates)) + _z(target_power)) * se
    z_eff = effect / se - _z(target_power)
    max_k = 0
    if z_eff > 0:
        tail = float(stats.norm.sf(z_eff))
        if tail > 0:
            max_k = int(min(alpha / tail, 1_000_000))
    if effect >= mde_search:
        verdict = "ANSWERABLE"
    elif effect >= mde_plain:
        verdict = "ANSWERABLE_IF_K_LE"
    else:
        verdict = "NOT_ANSWERABLE"
    req_plain = _required_n(effect, se, n_rows, alpha, 1, target_power)
    req_search = _required_n(effect, se, n_rows, alpha, k_candidates, target_power)
    req_groups = int(math.ceil(n_groups * (req_search / n_rows))) if req_search else None
    designs = _designs(effect, se, n_rows, n_groups, prevalence, alpha, target_power, k_candidates, 3, max(1, n_rows / max(1, n_groups)))
    prescription = (
        f"NOT_ANSWERABLE at n={n_rows}, groups={n_groups}: MDE {mde_plain:.3f} (plain) / {mde_search:.3f} "
        f"(search K={k_candidates}) versus effect that matters {effect:.3f}. "
        + (f"Collect about {req_search} rows across {req_groups} groups." if req_search else "Change the estimand.")
    ) if verdict == "NOT_ANSWERABLE" else (
        f"{verdict}: MDE plain {mde_plain:.3f}, search-priced {mde_search:.3f}, effect {effect:.3f}."
    )
    return Assessment(
        verdict=verdict,
        effect_that_matters=effect,
        alpha=alpha,
        target_power=target_power,
        mde_plain=float(mde_plain),
        mde_search=float(mde_search),
        max_k_answerable=int(max_k),
        k_candidates=int(k_candidates),
        confirmation_n=int(n_rows),
        confirmation_groups=int(n_groups),
        confirmation_prevalence=float(prevalence),
        baseline_auc=float(baseline_auc),
        se_delta=float(se),
        required_n_plain=int(req_plain) if req_plain else None,
        required_n_search=int(req_search) if req_search else None,
        required_groups_search=int(req_groups) if req_groups else None,
        prescription=prescription,
        designs=designs,
        mde_curve={"n": [], "series": {}, "effect_that_matters": effect},
        power_curve={"effect": [], "series": {}, "target_power": target_power},
        diagnostics={"method": "hanley_mcneil_spec", "se_auc": se_auc},
        warnings=["No row-level data; SE is an analytic approximation, not a group bootstrap."],
    )


def assess_continuous(
    n_per_group: int,
    sd: float,
    effect: float,
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> dict[str, Any]:
    """Two-sample means power for economics-style continuous outcomes."""
    n = max(2, int(n_per_group))
    se = sd * math.sqrt(2.0 / n)
    mde = (_z(1 - alpha) + _z(target_power)) * se
    power = float(clamp(stats.norm.cdf(effect / se - _z(1 - alpha)), 0, 1))
    req_n = int(math.ceil(2 * ((_z(1 - alpha) + _z(target_power)) * sd / max(1e-9, effect)) ** 2))
    return {
        "verdict": "ANSWERABLE" if effect >= mde else "NOT_ANSWERABLE",
        "n_per_group": n,
        "sd": sd,
        "effect_that_matters": effect,
        "mde": float(mde),
        "power": power,
        "required_n_per_group": max(2, req_n),
        "required_n_total": 2 * max(2, req_n),
        "prescription": (
            f"With {n} observations per arm and SD {sd:.2f}, the minimum detectable difference is {mde:.2f} "
            f"at {int(target_power*100)}% power; the declared effect is {effect:.2f}. "
            f"Collect about {max(2, req_n)} per arm."
        ),
        "method": "two-sample normal approximation",
    }
