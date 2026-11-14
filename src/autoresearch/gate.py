from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
from scipy import stats

from .contract import Contract
from .power import Assessment
from .utils import clamp


def _z(p: float) -> float:
    return float(stats.norm.ppf(p))


def _detect_value(row: dict[str, Any]) -> float | None:
    """Promotion rate from an OC-table row.

    Tables written before the promotion/strict split stored the confirmed-only
    rate in ``detect`` and the promoted rate in ``detect_affirmed``. Prefer the
    promoted rate so the certificate reports the instrument's actual yes rate.
    """
    for key in ("detect_affirmed", "detect"):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def _interp_oc(table: list[dict[str, Any]] | None, effect: float) -> dict[str, Any] | None:
    if not table:
        return None
    rows = sorted(
        [r for r in table if _detect_value(r) is not None],
        key=lambda r: float(r.get("effect", r.get("target_effect", 0))),
    )
    if not rows:
        return None
    xs = [float(r.get("effect", r.get("target_effect", 0))) for r in rows]
    if effect <= xs[0]:
        r = rows[0]
    elif effect >= xs[-1]:
        r = rows[-1]
    else:
        r = rows[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= effect <= xs[i + 1]:
                t = (effect - xs[i]) / max(1e-9, xs[i + 1] - xs[i])
                det = float(_detect_value(rows[i])) * (1 - t) + float(_detect_value(rows[i + 1])) * t
                fprs = [float(r["fpr"]) for r in rows if r.get("fpr") is not None]
                return {
                    "effect": float(effect),
                    "detect": float(det),
                    "fpr": float(np.mean(fprs)) if fprs else None,
                    "source": "calibration_interpolated",
                }
    return {
        "effect": float(effect),
        "detect": float(_detect_value(r)),
        "fpr": float(r["fpr"]) if r.get("fpr") is not None else None,
        "source": "calibration_nearest",
    }


@dataclass
class GateDecision:
    tier: str
    verdict: str
    claim_type: str
    claim: str
    checks: list[dict[str, Any]]
    gate_oc: dict[str, Any]
    power_at_observed: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assign_tier(
    contract: Contract,
    candidate: dict[str, Any] | None,
    confirmation: dict[str, Any],
    selection_null: dict[str, Any],
    discovery_stability: dict[str, Any],
    adversary_results: list[dict[str, Any]],
    power: Assessment,
    alpha_run: float,
    calibration_table: list[dict[str, Any]] | None = None,
    budget_info: dict[str, Any] | None = None,
    determinism_ok: bool = True,
) -> GateDecision:
    delta = float(confirmation.get("delta", float("nan")))
    ci_low = float(confirmation.get("ci_low", float("nan")))
    ci_high = float(confirmation.get("ci_high", float("nan")))
    p_conf = float(confirmation.get("p_one_sided", float("nan")))
    stability = float(confirmation.get("stability_confirmation", float("nan")))
    p_sel = float(selection_null.get("p_value", float("nan")))
    effect = contract.decision.smallest_effect_that_matters
    se = float(confirmation.get("se", float("nan")))
    if not np.isfinite(se) or se <= 0:
        if np.isfinite(ci_low) and np.isfinite(ci_high):
            se = max(1e-6, (ci_high - ci_low) / (2 * 1.96))
        else:
            se = float("nan")
    power_obs = float(stats.norm.cdf(delta / se - _z(1 - alpha_run))) if np.isfinite(se) and se > 0 else float("nan")

    critical_attacks = [a for a in adversary_results if a.get("critical", False)]
    attack_failures = [a for a in critical_attacks if a.get("result") == "fail"]
    attack_warnings = [a for a in adversary_results if a.get("result") == "warn"]
    leakage_fail = any(a.get("attack") == "leakage_scan" and a.get("result") == "fail" for a in adversary_results)
    pseudo_fail = any(a.get("attack") == "pseudo_group" and a.get("result") == "fail" for a in adversary_results)

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None, critical: bool = True) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, "critical": critical})

    add("candidate_selected", candidate is not None, None if candidate is None else candidate.get("id"))
    add("confirmation_delta_positive", delta > 0, delta)
    add("confirmation_ci_excludes_zero", ci_low > 0, {"ci_low": ci_low, "ci_high": ci_high})
    add("confirmation_p_le_alpha", p_conf <= alpha_run, {"p_one_sided": p_conf, "alpha_run": alpha_run})
    add("selection_null_p_le_alpha", p_sel <= alpha_run, {"p_value": p_sel, "n_permutations": selection_null.get("n_permutations")})
    add("stability_confirmation", stability >= 0.7 if np.isfinite(stability) else False, stability)
    add("discovery_fold_stability", float(discovery_stability.get("fold_fraction_positive", 0)) >= 0.6, discovery_stability.get("fold_fraction_positive"))
    add("effect_meets_decision_threshold", delta >= effect, {"delta": delta, "effect_that_matters": effect})
    add("adversary_critical_attacks", len(attack_failures) == 0, [a.get("attack") for a in attack_failures])
    add("leakage_scan", not leakage_fail, "no label-derived feature or split overlap")
    add("pseudo_group_control", not pseudo_fail, "no context artifact detected")
    add("determinism", determinism_ok, "same-certificate hash across processes")

    # Gate operating characteristics at the observed effect.
    oc = _interp_oc(calibration_table, max(0.0, delta))
    gate_oc = oc or {
        "effect": float(max(0.0, delta)),
        "detect": float(power_obs) if np.isfinite(power_obs) else None,
        "fpr": float(alpha_run),
        "source": "power_approximation",
    }

    hard_fail = (
        candidate is None
        or leakage_fail
        or pseudo_fail
        or not determinism_ok
        or any(c["name"] in ("confirmation_delta_positive", "confirmation_ci_excludes_zero") and not c["ok"] for c in checks if c["critical"])
    )
    # A negative claim is confirmed when the CI rules out the decision effect.
    if candidate is not None and delta <= 0 and np.isfinite(ci_high) and ci_high < effect:
        return GateDecision(
            tier="confirmed",
            verdict="refused",
            claim_type="negative",
            claim=(
                f"No tested candidate improved {contract.metric_raw.get('primary')} over {contract.baseline.name} "
                f"by the pre-committed {effect:.3f}; the upper 95% bound was {ci_high:.3f} "
                f"on {confirmation.get('n')} held-out units across {confirmation.get('n_groups')} groups."
            ),
            checks=checks,
            gate_oc=gate_oc,
            power_at_observed=power_obs,
            reasons=["upper_ci_below_decision_threshold"],
        )

    if hard_fail:
        return GateDecision(
            tier="artifact_risk",
            verdict="not_affirmed",
            claim_type="negative_insufficient",
            claim=(
                f"The candidate did not clear the certification gate. Point estimate {delta:.3f} "
                f"(95% CI {ci_low:.3f} to {ci_high:.3f}); treat this as an artifact-risk result, not a discovery."
            ),
            checks=checks,
            gate_oc=gate_oc,
            power_at_observed=power_obs,
            reasons=[c["name"] for c in checks if c["critical"] and not c["ok"]],
        )

    if delta > 0 and (p_sel > alpha_run or p_conf > alpha_run or ci_low <= 0):
        return GateDecision(
            tier="exploratory",
            verdict="not_affirmed",
            claim_type="negative_insufficient",
            claim=(
                f"Directional signal only: {delta:.3f} with 95% CI {ci_low:.3f} to {ci_high:.3f}; "
                f"selection-priced p={p_sel:.3f}, confirmation p={p_conf:.3f}, alpha={alpha_run:.3f}. "
                "Not promotable under the registered rule."
            ),
            checks=checks,
            gate_oc=gate_oc,
            power_at_observed=power_obs,
            reasons=["not_significant_after_search"],
        )

    if delta < effect or stability < 0.7 or attack_failures:
        return GateDecision(
            tier="provisional",
            verdict="not_affirmed",
            claim_type="negative_insufficient",
            claim=(
                f"Statistically detectable but below the decision threshold or not robust: delta={delta:.3f} "
                f"versus effect-that-matters={effect:.3f}; stability={stability:.2f}. "
                "Do not reallocate on this alone."
            ),
            checks=checks,
            gate_oc=gate_oc,
            power_at_observed=power_obs,
            reasons=["below_decision_threshold_or_unstable"],
        )

    if stability >= 0.8 and not attack_failures and delta >= effect and (not np.isfinite(power_obs) or power_obs >= contract.decision.target_power):
        tier = "confirmed"
        prefix = "Confirmed:"
    else:
        tier = "supported"
        prefix = "Supported:"

    claim = (
        f"{prefix} on {confirmation.get('n')} held-out units across {confirmation.get('n_groups')} groups, "
        f"the selected rule improved {contract.metric_raw.get('primary')} over {contract.baseline.name} "
        f"by {delta:.3f} (95% CI {ci_low:.3f} to {ci_high:.3f}), exceeding the pre-committed smallest effect "
        f"{effect:.3f}. The search over {selection_null.get('effective_independent', 'K')} effective independent "
        f"hypotheses was priced by a selection-aware null (p={p_sel:.3f})."
    )
    return GateDecision(
        tier=tier,
        verdict="affirmed",
        claim_type="affirmative",
        claim=claim,
        checks=checks,
        gate_oc=gate_oc,
        power_at_observed=power_obs,
        reasons=[],
    )
