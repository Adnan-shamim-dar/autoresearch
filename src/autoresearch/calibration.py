from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics
from .budget import compute_budget
from .contract import Contract, ContractError
from .demo import FEATURES, generate_immunotherapy_data, make_contract_dict, _sigmoid, _z
from .evaluator import (
    assign_folds,
    effective_independent,
    evaluate_candidates,
    permute_within_groups,
    select_on_permuted_outcomes,
    selection_aware_null,
)
from .adversary import run_adversary
from .engine import CODE_FILES
from .gate import assign_tier
from .power import assess
from .proposer import Proposer
from .scoring import Candidate, FeatureMatrix
from .utils import atomic_write_json, ensure_dir, now_iso, rng_for, sha256_obj

# The planted-signal fixture uses the same feature vocabulary as the demo, but
# the signal scale is tuned per replicate so that the *realised* oracle delta
# matches the requested row of the OC table. Synthetic-only calibration is a
# demonstration of the instrument, not evidence about real biology.
TRUTH_WEIGHTS = {
    "binding_rank": 0.60,
    "expression": 1.00,
    "stability": 0.80,
    "foreignness": 0.75,
    "presentation": 0.50,
    "self_similarity": -0.50,
    "processing": 0.25,
    "publicness": 0.15,
}

CALIBRATION_LIMITATIONS = [
    "Synthetic substrate: this table demonstrates the gate's operating characteristics; it is not validated instrument performance on real data.",
    "Synthetic-only calibration is not believed (spec §14): the simulation is written by the method's author. A real-data table is still owed.",
    "The full adversary runs in every replicate; 'Detected' includes its kills. The adversary's own false-alarm rate is reported below.",
    "Candidates come from the operator-family seed bank (no adaptive generations) to keep replicates affordable.",
    "Effects are planted in a known latent score; the search sees only the contract features, never the planted score.",
    "A refusal caused by the power oracle is counted as a non-detection, because that is what the pipeline would actually emit.",
]


def _truth_score(df: pd.DataFrame) -> np.ndarray:
    raw = {
        "binding_rank": -np.log10(df["binding_rank"].to_numpy()),
        "expression": np.log1p(df["expression"].to_numpy()),
        "stability": np.log1p(df["stability"].to_numpy()),
        "foreignness": df["foreignness"].to_numpy(),
        "presentation": df["presentation"].to_numpy(),
        "self_similarity": df["self_similarity"].to_numpy(),
        "processing": df["processing"].to_numpy(),
        "publicness": df["publicness"].to_numpy(),
    }
    return sum(w * _z(raw[k]) for k, w in TRUTH_WEIGHTS.items())


def _intercept_for_prevalence(signal: np.ndarray, prevalence: float) -> float:
    lo, hi = -14.0, 14.0
    for _ in range(80):
        mid = (lo + hi) / 2
        p = _sigmoid(mid + signal)
        if p.mean() < prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _plant_outcomes(
    df: pd.DataFrame,
    target_effect: float,
    seed: int,
    prevalence: float = 0.19,
) -> pd.DataFrame:
    """Re-draw outcomes with a latent signal tuned to a realised oracle delta.

    Common random numbers make the draw monotone in the signal scale, so a
    bisection on the scale lands near the requested effect without regenerating
    features.
    """
    rng = rng_for("calibration-plant", seed)
    truth = _truth_score(df)
    baseline = df["baseline_score"].to_numpy()
    patient_shift = df["patient_id"].map(
        {pid: rng.normal(0, 0.18) for pid in df["patient_id"].unique()}
    ).to_numpy()
    uniforms = rng.random(len(df))

    def draw(scale: float) -> tuple[np.ndarray, float]:
        signal = scale * (truth + patient_shift)
        intercept = _intercept_for_prevalence(signal, prevalence)
        p = _sigmoid(intercept + signal)
        y = (uniforms < p).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            return y, float("nan")
        delta = metrics.auc(y, truth) - metrics.auc(y, baseline)
        return y, float(delta)

    if target_effect <= 0:
        y, realized = draw(0.0)
    else:
        lo, hi = 0.0, 4.0
        y_hi, d_hi = draw(hi)
        if np.isfinite(d_hi) and d_hi < target_effect:
            y, realized = y_hi, d_hi
        else:
            best_y, best_d = draw(0.0)
            for _ in range(22):
                mid = (lo + hi) / 2
                y_mid, d_mid = draw(mid)
                if not np.isfinite(d_mid):
                    hi = mid
                    continue
                if abs(d_mid - target_effect) < abs(best_d - target_effect):
                    best_y, best_d = y_mid, d_mid
                if d_mid < target_effect:
                    lo = mid
                else:
                    hi = mid
            y, realized = best_y, best_d
    out = df.copy()
    out["immunogenic"] = np.asarray(y, dtype=int)
    return out


def _bank(
    contract: Contract,
    discovery: pd.DataFrame,
    n_candidates: int,
    seed: int,
) -> tuple[FeatureMatrix, list[Candidate], np.ndarray]:
    budget = compute_budget(contract, discovery)
    fm = FeatureMatrix(discovery, contract, fit=True)
    proposer = Proposer(
        contract=contract,
        data=discovery,
        fm=fm,
        budget=budget,
        seed=seed,
    )
    raw = proposer.seed_candidates(n=n_candidates)
    valid: list[Candidate] = []
    scores: list[np.ndarray] = []
    for c in raw:
        if c is None:
            continue
        ok, _ = budget.check(c)
        if not ok:
            continue
        try:
            s = fm.score_candidate(c)
        except Exception:
            continue
        if not np.all(np.isfinite(s)):
            continue
        valid.append(c)
        scores.append(s)
    if not scores:
        raise RuntimeError("Calibration candidate bank is empty")
    return fm, valid, np.column_stack(scores)


def _groups(df: pd.DataFrame, contract: Contract) -> np.ndarray:
    if contract.group_column and contract.group_column in df.columns:
        return df[contract.group_column].astype(str).to_numpy()
    return np.array(["__all__"] * len(df))


def _confirmation(
    contract: Contract,
    confirmation: pd.DataFrame,
    candidate: Candidate,
    seed: int,
    bootstrap: int = 600,
) -> dict[str, Any]:
    fm = FeatureMatrix(
        confirmation,
        contract,
        fit=False,
        fit_stats=candidate.params.get("fit_stats", {}),
    )
    s_cand = fm.score_candidate(candidate)
    s_base = fm.score_candidate(contract.baseline_candidate())
    y = confirmation[contract.outcome_column].to_numpy().astype(int)
    groups = _groups(confirmation, contract)
    m = metrics.compute_metrics(y, groups, s_cand, s_base, top_k=contract.top_k)
    boot = metrics.group_bootstrap_deltas(y, groups, s_cand, s_base, n_boot=bootstrap, seed=seed)
    summ = metrics.bootstrap_summary(boot)
    per_group = metrics.per_group_delta(y, groups, s_cand, s_base)
    vals = np.asarray(list(per_group.values()), dtype=float)
    delta = summ["delta"] if np.isfinite(summ["delta"]) else m["delta_auc"]
    return {
        "n": int(len(y)),
        "n_groups": int(len(set(groups))),
        "delta": float(delta),
        "point_delta": float(m["delta_auc"]),
        "ci_low": float(summ["ci_low"]),
        "ci_high": float(summ["ci_high"]),
        "se": float(summ["se"]),
        "p_one_sided": float(summ["p_one_sided"]),
        "stability_confirmation": metrics.stability_fraction(y, groups, s_cand, s_base, boot),
        "auc_candidate": float(m["auc_candidate"]),
        "auc_baseline": float(m["auc_baseline"]),
        "frac_improved": float(m["frac_improved"]),
        "top_k_enrichment": float(m["top_k_enrichment"]),
    }


def _pass_attacks() -> list[dict[str, Any]]:
    """Every attack held at pass: the gate's verdict with the adversary removed."""
    return [
        {"attack": "leakage_scan", "result": "pass", "critical": True},
        {"attack": "pseudo_group", "result": "pass", "critical": True},
        {"attack": "outcome_permutation", "result": "pass", "critical": True},
        {"attack": "baseline_advocate", "result": "pass", "critical": True},
    ]


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    seed = int(payload["seed"])
    target_effect = float(payload["target_effect"])
    n_patients = int(payload.get("patients", 420))
    n_confirmation_groups = int(payload.get("confirmation_groups", 140))
    n_candidates = int(payload["candidates"])

    features = generate_immunotherapy_data(seed=seed, n_patients=n_patients)
    full = _plant_outcomes(features, target_effect=target_effect, seed=seed + 17)
    pids = sorted(full["patient_id"].unique())
    perm = rng_for("calibration-split", seed, n_patients).permutation(pids)
    conf_pids = set(perm[:n_confirmation_groups])
    discovery = full[~full["patient_id"].isin(conf_pids)].reset_index(drop=True)
    confirmation = full[full["patient_id"].isin(conf_pids)].reset_index(drop=True)

    raw = make_contract_dict(
        effect=0.04,
        k_candidates=n_candidates,
        confirmation_n=len(confirmation),
        confirmation_groups=int(confirmation["patient_id"].nunique()),
    )
    true_effect = float(
        metrics.auc(full["immunogenic"].to_numpy(), _truth_score(full))
        - metrics.auc(full["immunogenic"].to_numpy(), full["baseline_score"])
    )
    return evaluate_replicate(
        payload,
        raw,
        discovery,
        confirmation,
        discovery["baseline_score"].to_numpy(),
        true_effect,
    )


def evaluate_replicate(
    payload: dict[str, Any],
    raw_contract: dict[str, Any],
    discovery: pd.DataFrame,
    confirmation: pd.DataFrame,
    baseline_discovery: np.ndarray | None,
    true_effect: float,
) -> dict[str, Any]:
    """One OC-table replicate on prepared data: the engine's pipeline end to end.

    Domain-independent: the substrate (synthetic or real-with-planted-outcome)
    is built by the caller. ``baseline_discovery`` defaults to the contract's
    baseline candidate scored on discovery, as in the engine.
    """
    seed = int(payload["seed"])
    target_effect = float(payload["target_effect"])
    n_candidates = int(payload["candidates"])
    n_permutations = int(payload["permutations"])
    n_splits = int(payload.get("splits", 5))

    contract = Contract(raw_contract)
    contract.validate_static()
    try:
        contract.attach_data(discovery, role="discovery")
    except ContractError as exc:
        return {"ok": False, "error": exc.to_dict(), "seed": seed, "target_effect": target_effect}

    budget = compute_budget(contract, discovery)
    fm, candidates, score_matrix = _bank(contract, discovery, n_candidates, seed)
    if baseline_discovery is None:
        baseline_discovery = fm.score_candidate(contract.baseline_candidate())
    comp = {c.id: c.complexity for c in candidates}

    ev = evaluate_candidates(
        discovery,
        contract,
        candidates,
        fm,
        baseline_discovery,
        budget,
        n_splits=n_splits,
        seed=seed,
    )
    if ev.selected_candidate is None or ev.score_matrix is None:
        return {"ok": False, "error": {"reason_code": "no_candidate"}, "seed": seed, "target_effect": target_effect}
    selected = ev.selected_candidate
    selected_result = ev.selected_result
    eval_ids = [c.id for c in candidates if c.id not in ev.duplicate_map]
    if ev.score_matrix is None or not eval_ids:
        return {"ok": False, "error": {"reason_code": "no_scorable_candidate"}, "seed": seed, "target_effect": target_effect}
    score_matrix_eff = ev.score_matrix
    comp_eff = {cid: comp[cid] for cid in eval_ids}
    eff_k, _clusters = effective_independent(score_matrix_eff, eval_ids)

    observed = _confirmation(contract, confirmation, selected, seed=seed + 101, bootstrap=700)

    y = discovery[contract.outcome_column].to_numpy().astype(int)
    groups = _groups(discovery, contract)
    folds = assign_folds(groups, y, n_splits=n_splits, seed=seed)
    # Same selection-aware null as the engine: discovery data only.
    null_summary = selection_aware_null(
        score_matrix_eff,
        baseline_discovery,
        y,
        groups,
        folds,
        eval_ids,
        comp_eff,
        n_permutations=n_permutations,
        rng_parts=("calibration-null", seed),
    )
    p_sel = float(null_summary["p_value"]) if null_summary["p_value"] is not None else float("nan")

    power = assess(
        contract,
        discovery,
        confirmation_n=len(confirmation),
        confirmation_groups=int(len(set(_groups(confirmation, contract)))) if contract.group_column else len(confirmation),
        confirmation_prevalence=float(confirmation[contract.outcome_column].mean()),
        k_candidates=len(candidates),
        alpha=0.05,
        target_power=0.8,
        bootstrap=120,
        seed=seed + 211,
    )
    fold_deltas = selected_result.get("fold_deltas", []) if selected_result else []
    discovery_stability = {
        "fold_fraction_positive": float(np.mean([d > 0 for d in fold_deltas])) if fold_deltas else float("nan"),
        "fold_deltas": fold_deltas,
    }

    row: dict[str, Any] = {
        "ok": True,
        "seed": seed,
        "target_effect": target_effect,
        "true_effect": float(true_effect),
        "n_discovery": int(len(discovery)),
        "n_confirmation": int(len(confirmation)),
        "n_groups_confirmation": int(len(set(_groups(confirmation, contract)))),
        "baseline_auc_confirmation": float(observed["auc_baseline"]),
        "power_verdict": power.verdict,
        "power_mde_search": float(power.mde_search),
        "power_mde_plain": float(power.mde_plain),
        "power_se_conf": float(power.se_delta),
        "se_confirmation": float(observed["se"]),
        "selected_candidate": selected.id,
        "selected_family": selected.family,
        "discovery_delta": float(selected_result["mean_delta"]) if selected_result else None,
        "delta": observed["delta"],
        "ci_low": observed["ci_low"],
        "ci_high": observed["ci_high"],
        "p_confirmation": observed["p_one_sided"],
        "stability_confirmation": observed["stability_confirmation"],
        "p_selection": p_sel,
        "effective_independent": int(eff_k),
        "proposed": int(len(candidates)),
        "null_max_stats": null_summary.get("null_max_stats", []),
        "observed_max_stat": null_summary.get("observed_max_stat"),
    }

    if payload.get("return_objects"):
        # In-process callers (the benchmark's AutoResearch agent) need the rule itself.
        row["_objects"] = {"candidate": selected, "contract": contract}

    if power.verdict == "NOT_ANSWERABLE" and contract.decision.if_underpowered == "refuse":
        row.update(
            {
                "tier": "power_refusal",
                "verdict": "refused",
                "claim_type": "refusal",
                "detected_confirmed": False,
                "detected_affirmed": False,
                "false_positive_alpha": bool(observed["delta"] > 0 and observed["p_one_sided"] <= 0.05),
            }
        )
        return row

    def decide(attacks: list[dict[str, Any]]):
        return assign_tier(
            contract,
            selected.to_dict(),
            observed,
            {**null_summary, "effective_independent": eff_k, "proposed": len(candidates)},
            discovery_stability,
            attacks,
            power,
            0.05,
            calibration_table=None,
            budget_info=budget.to_dict(),
        )

    def promoted(dec) -> bool:
        return bool(dec.tier in ("confirmed", "supported") and dec.verdict == "affirmed")

    # The real red team, exactly as the engine runs it (spec §9: calibrate the
    # adversary too). The gate is also evaluated without it, so an honest
    # result the adversary kills is counted as an adversary false alarm.
    attacks = run_adversary(
        contract,
        discovery,
        fm,
        selected,
        y,
        groups,
        baseline_discovery,
        float(selected_result["mean_delta"]),
        budget,
        alpha=0.05,
        seed=seed,
        validation=contract.validation.to_dict() if contract.validation else None,
        discovery_groups=sorted(map(str, set(groups))),
        # As in the engine: the vault records no group hash for a no-group dataset.
        confirmation_group_hash=(
            sha256_obj(sorted(map(str, set(_groups(confirmation, contract))))) if contract.group_column else None
        ),
        code_files=CODE_FILES,
    )
    decision = decide(attacks)
    without = decide(_pass_attacks())
    failed_critical = [a["attack"] for a in attacks if a.get("critical") and a.get("result") == "fail"]
    row.update(
        {
            "tier": decision.tier,
            "verdict": decision.verdict,
            "claim_type": decision.claim_type,
            "detected_confirmed": bool(decision.tier == "confirmed" and decision.verdict == "affirmed"),
            "detected_affirmed": promoted(decision),
            "detected_without_adversary": promoted(without),
            "adversary_killed": bool(promoted(without) and not promoted(decision)),
            "adversary_failed_critical": failed_critical,
            "adversary_results": {a["attack"]: a.get("result") for a in attacks},
            "false_positive_alpha": bool(observed["delta"] > 0 and observed["p_one_sided"] <= 0.05),
        }
    )
    return row


def _print_progress(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
    print(*args, **kwargs, flush=True)


def summarize_replicates(rows: list[dict[str, Any]], effects: tuple[float, ...]) -> dict[str, Any]:
    """Build the published OC table from per-replicate rows.

    ``detect`` is the promotion rate: the fraction of replicates the gate
    affirmed at either the ``confirmed`` or ``supported`` tier. The stricter
    ``confirmed``-only rate is reported separately, because a table that showed
    only the strict tier would make a working gate look broken.
    """
    good = [r for r in rows if r.get("ok", True)]
    null_rows = [r for r in good if float(r.get("target_effect", 1)) == 0.0]
    fpr_alpha = (
        float(np.mean([bool(r.get("false_positive_alpha")) for r in null_rows])) if null_rows else float("nan")
    )
    fpr_decision = (
        float(np.mean([bool(r.get("detected_affirmed")) for r in null_rows])) if null_rows else float("nan")
    )

    table: list[dict[str, Any]] = []
    for effect in effects:
        group = [r for r in good if float(r.get("target_effect", -1)) == float(effect)]
        if not group:
            continue
        detected = sum(1 for r in group if r.get("detected_affirmed"))
        confirmed = sum(1 for r in group if r.get("detected_confirmed"))
        det_lo, det_hi = _wilson(detected, len(group))
        conf_lo, conf_hi = _wilson(confirmed, len(group))
        table.append(
            {
                "effect": float(effect),
                "n": int(len(group)),
                "mean_true_effect": float(np.mean([r["true_effect"] for r in group])),
                "detect": float(detected / len(group)),
                "detect_affirmed": float(detected / len(group)),
                "detect_ci_low": float(det_lo),
                "detect_ci_high": float(det_hi),
                "detect_confirmed": float(confirmed / len(group)),
                "confirmed_ci_low": float(conf_lo),
                "confirmed_ci_high": float(conf_hi),
                "fpr": float(fpr_alpha),
                "fpr_decision": float(fpr_decision),
                "mean_delta": float(np.mean([r["delta"] for r in group])),
                "mean_p_confirmation": float(np.mean([r["p_confirmation"] for r in group])),
                "mean_p_selection": float(
                    np.mean(
                        [
                            r["p_selection"]
                            for r in group
                            if np.isfinite(r.get("p_selection", float("nan")))
                        ]
                    )
                ),
                "power_refusals": int(sum(1 for r in group if r.get("tier") == "power_refusal")),
                "detect_without_adversary": float(np.mean([bool(r.get("detected_without_adversary", r.get("detected_affirmed"))) for r in group])),
                "adversary_kills": int(sum(1 for r in group if r.get("adversary_killed"))),
            }
        )
    # Adversary false alarm (spec §9): among replicates with a real planted
    # effect that the gate would promote without the adversary, the fraction
    # the adversary killed.
    honest = [r for r in good if float(r.get("target_effect", 0)) > 0 and r.get("detected_without_adversary")]
    killed = sum(1 for r in honest if r.get("adversary_killed"))
    lo, hi = _wilson(killed, len(honest)) if honest else (float("nan"), float("nan"))
    per_attack: dict[str, dict[str, int]] = {}
    for r in good:
        if float(r.get("target_effect", 0)) <= 0:
            continue
        for name, res in (r.get("adversary_results") or {}).items():
            slot = per_attack.setdefault(name, {"n": 0, "fail": 0, "warn": 0})
            slot["n"] += 1
            if res in ("fail", "warn"):
                slot[res] += 1
    adversary = {
        "honest_promotions": len(honest),
        "killed": int(killed),
        "false_alarm_rate": float(killed / len(honest)) if honest else float("nan"),
        "false_alarm_ci_low": float(lo),
        "false_alarm_ci_high": float(hi),
        "per_attack_on_real_effects": per_attack,
    }
    return {"rows": table, "fpr_alpha": fpr_alpha, "fpr_decision": fpr_decision, "adversary": adversary}


def _render_oc_markdown(payload: dict[str, Any]) -> str:
    params = payload.get("parameters") or {}
    rows = payload.get("rows") or []
    lines = [
        "# Operating characteristics of the certification gate",
        "",
        f"Generated: {payload.get('generated_at', 'unknown')}",
        f"Substrate: {payload.get('domain', 'unknown')}",
        "",
        "`Detected` is the promotion rate: the fraction of replicates the gate "
        "affirmed at either the **confirmed** or **supported** tier. "
        "`Confirmed` is the stricter top-tier-only rate. `FPR (alpha)` is the "
        "fraction of pure-null replicates with a positive confirmation delta and "
        "one-sided p <= alpha; `FPR (decision)` is the fraction the gate actually promoted.",
        "",
        "| True effect | n | Mean realised effect | Detected | 95% CI | Confirmed | Without adversary | Adversary kills | FPR (alpha) | FPR (decision) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['effect']:.3f} | {r['n']} | {r['mean_true_effect']:.3f} | "
            f"{r['detect']:.2f} | {r['detect_ci_low']:.2f}-{r['detect_ci_high']:.2f} | "
            f"{r.get('detect_confirmed', float('nan')):.2f} | {r.get('detect_without_adversary', float('nan')):.2f} | "
            f"{r.get('adversary_kills', 0)} | {r['fpr']:.2f} | {r['fpr_decision']:.2f} |"
        )
    adv = payload.get("adversary") or {}
    if adv.get("honest_promotions"):
        lines += [
            "",
            "## Adversary false-alarm rate (spec §9)",
            "",
            f"Of {adv['honest_promotions']} replicates with a real planted effect that the gate would promote without "
            f"the adversary, the adversary killed {adv['killed']}: false-alarm rate {adv['false_alarm_rate']:.2f} "
            f"(95% CI {adv['false_alarm_ci_low']:.2f}-{adv['false_alarm_ci_high']:.2f}).",
            "",
            "| Attack | Runs on real effects | Fail | Warn |",
            "|---|---:|---:|---:|",
        ]
        for name, c in sorted((adv.get("per_attack_on_real_effects") or {}).items()):
            lines.append(f"| {name} | {c['n']} | {c.get('fail', 0)} | {c.get('warn', 0)} |")
    lines += [
        "",
        "## Setup",
        "",
        f"- Replicates per effect: {params.get('reps_per_effect')}",
        f"- Candidate bank: {params.get('candidates')}",
        f"- Selection-null permutations: {params.get('permutations')}",
        f"- Discovery patients: {params.get('patients')}",
        f"- Confirmation groups: {params.get('confirmation_groups')}",
        f"- Wall clock: {params.get('elapsed_seconds')} s",
        "",
        "## Method",
        "",
        payload.get("method", ""),
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in payload.get("limitations", [])]
    lines.append("")
    return "\n".join(lines)


def publish_oc_table(root: str | Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Re-summarize an existing OC table and write the human-readable copy.

    This can rebuild the published rows from ``replicates`` without re-running
    the calibration, so a summary fix does not cost another calibration run.
    """
    root = Path(root)
    path = root / "calibration" / "oc_table.json"
    if payload is None:
        payload = json.loads(path.read_text(encoding="utf-8"))
    effects = tuple(float(e) for e in (payload.get("parameters") or {}).get("effects", []))
    replicates = list(payload.get("replicates") or [])
    summary = summarize_replicates(replicates, effects)
    payload["rows"] = summary["rows"]
    payload["fpr_alpha"] = summary["fpr_alpha"]
    payload["fpr_decision"] = summary["fpr_decision"]
    payload["adversary"] = summary["adversary"]
    payload["summarized_at"] = now_iso()
    atomic_write_json(path, payload)
    (root / "calibration" / "oc_table.md").write_text(_render_oc_markdown(payload), encoding="utf-8")
    return payload


def run_calibration(
    root: str | Path,
    reps: int = 20,
    effects: tuple[float, ...] = (0.0, 0.03, 0.06, 0.10, 0.15),
    candidates: int = 32,
    permutations: int = 39,
    workers: int = 0,
    patients: int = 420,
    confirmation_groups: int = 140,
) -> dict[str, Any]:
    root = Path(root)
    ensure_dir(root / "calibration")
    workers = workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    jobs = []
    for ei, effect in enumerate(effects):
        for r in range(reps):
            jobs.append(
                {
                    "seed": 900000 + ei * 1000 + r,
                    "target_effect": float(effect),
                    "candidates": int(candidates),
                    "permutations": int(permutations),
                    "patients": int(patients),
                    "confirmation_groups": int(confirmation_groups),
                }
            )
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    _print_progress(f"Calibration: {len(jobs)} replicates across {len(effects)} effect sizes, {workers} workers")
    if workers <= 1:
        for i, job in enumerate(jobs):
            rows.append(run_one(job))
            if (i + 1) % max(1, len(jobs) // 10) == 0:
                _print_progress(f"  {i + 1}/{len(jobs)}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(run_one, job): job for job in jobs}
            done = 0
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                except Exception as exc:  # pragma: no cover - worker isolation
                    job = futures[fut]
                    rows.append(
                        {
                            "ok": False,
                            "seed": job["seed"],
                            "target_effect": job["target_effect"],
                            "error": {"reason_code": "worker_error", "message": str(exc)},
                        }
                    )
                done += 1
                if done % max(1, len(jobs) // 10) == 0:
                    _print_progress(f"  {done}/{len(jobs)} ({time.time() - t0:.1f}s)")

    good = [r for r in rows if r.get("ok")]
    summary = summarize_replicates(good, effects)
    table = summary["rows"]
    fpr_alpha = summary["fpr_alpha"]
    fpr_decision = summary["fpr_decision"]

    payload = {
        "schema": "autoresearch.oc_table/v1",
        "generated_at": now_iso(),
        "domain": "immunotherapy synthetic planted-signal fixture",
        "method": (
            "Replicates use the real contract validator, proposer bank, group-aware CV, discovery-only "
            "selection-aware null, single-candidate confirmation, power oracle, full adversary and gate. "
            "Effects are planted in a known latent score and realised on the generated cohort; the search "
            "never sees that score."
        ),
        "parameters": {
            "reps_per_effect": int(reps),
            "effects": [float(e) for e in effects],
            "candidates": int(candidates),
            "permutations": int(permutations),
            "patients": int(patients),
            "confirmation_groups": int(confirmation_groups),
            "workers": int(workers),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "fpr_alpha": float(fpr_alpha),
        "fpr_decision": float(fpr_decision),
        "adversary": summary["adversary"],
        "rows": table,
        "replicates": [
            {
                "seed": r.get("seed"),
                "target_effect": r.get("target_effect"),
                "true_effect": r.get("true_effect"),
                "tier": r.get("tier"),
                "verdict": r.get("verdict"),
                "power_verdict": r.get("power_verdict"),
                "delta": r.get("delta"),
                "ci_low": r.get("ci_low"),
                "ci_high": r.get("ci_high"),
                "p_confirmation": r.get("p_confirmation"),
                "p_selection": r.get("p_selection"),
                "stability_confirmation": r.get("stability_confirmation"),
                "detected_confirmed": r.get("detected_confirmed"),
                "detected_affirmed": r.get("detected_affirmed"),
                "detected_without_adversary": r.get("detected_without_adversary"),
                "adversary_killed": r.get("adversary_killed"),
                "adversary_failed_critical": r.get("adversary_failed_critical"),
                "adversary_results": r.get("adversary_results"),
                "false_positive_alpha": r.get("false_positive_alpha"),
            }
            for r in good
        ],
        "limitations": CALIBRATION_LIMITATIONS,
    }
    payload = publish_oc_table(root, payload)
    return payload
