"""Operating characteristics on real data (spec §7 and §14, proof 1).

Plasmode design: the 569 real WDBC tumours keep their real 30 measurements,
their real joint distribution, correlations and tails, and the real sample
size. Only the outcome is re-drawn, so the true effect is known:

    logit P(y = 1) = a + s * (z_baseline + beta * z_extra)

``z_baseline`` is the deployed baseline (tumour size, ``mean_radius``),
``z_extra`` the mean of three other real measurements. With ``beta = 0`` the
baseline is the whole truth and any rule that "beats" it is a false positive.
``beta`` is bisected per replicate so the realised AUC gain over the baseline
matches the requested row. Each replicate also re-draws the sealed split.

The engine never sees ``z_extra`` or ``beta``: it gets the contract's ten
features and has to find the signal.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics
from .calibration import _render_oc_markdown, evaluate_replicate, summarize_replicates
from .demo import _sigmoid
from .utils import atomic_write_json, ensure_dir, now_iso, rng_for
from .wdbc import load_frame, make_contract_dict

BASELINE = "mean_radius"
EXTRA = ("mean_concave_points", "mean_texture", "mean_smoothness")
SCALE = 1.2
PREVALENCE = 0.37  # the real malignant fraction in WDBC
CONFIRM_FRACTION = 0.4
CONTRACT_EFFECT = 0.10  # 0.04 is refused by the power oracle at n=228 (MDE_search ~0.08-0.10)

LIMITATIONS = [
    "Plasmode design: real measurements, real correlations and the real sample size (569 tumours); "
    "the outcome is re-drawn from a known model so the true effect is known. Real outcomes cannot carry a planted effect of known size.",
    "One real dataset, no group structure. The group-aware machinery is calibrated only on the synthetic table.",
    "The planting model is logistic-linear in rank-normalised features, which the rule grammar can express. "
    "A truth the grammar cannot express would be harder to find.",
    "Candidates come from the operator-family seed bank (no adaptive generations), as in the synthetic table.",
    "A refusal by the power oracle is counted as a non-detection.",
]


def _z_rank(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x) / (len(x) + 1)
    return stats.norm.ppf(r)


def _intercept(signal: np.ndarray, prevalence: float) -> float:
    lo, hi = -14.0, 14.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(mid + signal).mean() < prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def plant(frame: pd.DataFrame, target_effect: float, seed: int) -> tuple[pd.DataFrame, float]:
    """Replace the real outcome with one carrying a known AUC gain over the baseline."""
    zb = _z_rank(frame[BASELINE].to_numpy())
    zx = np.mean([_z_rank(frame[c].to_numpy()) for c in EXTRA], axis=0)
    uniforms = rng_for("real-plant", seed).random(len(frame))

    def draw(beta: float) -> tuple[np.ndarray, float]:
        truth = zb + beta * zx
        signal = SCALE * truth
        y = (uniforms < _sigmoid(_intercept(signal, PREVALENCE) + signal)).astype(int)
        return y, float(metrics.auc(y, truth) - metrics.auc(y, zb))

    if target_effect <= 0:
        y, realised = draw(0.0)
    else:
        lo, hi = 0.0, 40.0
        y, realised = draw(hi)
        if realised >= target_effect:
            best = draw(0.0)
            for _ in range(40):
                mid = (lo + hi) / 2
                cand = draw(mid)
                if abs(cand[1] - target_effect) < abs(best[1] - target_effect):
                    best = cand
                lo, hi = (mid, hi) if cand[1] < target_effect else (lo, mid)
            y, realised = best
    out = frame.copy()
    out["malignant"] = y
    return out, realised


def split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified unit split, re-drawn per replicate."""
    rng = rng_for("real-split", seed)
    y = frame["malignant"].to_numpy()
    conf = np.zeros(len(frame), dtype=bool)
    for label in (0, 1):
        idx = np.where(y == label)[0]
        conf[rng.permutation(idx)[: int(round(len(idx) * CONFIRM_FRACTION))]] = True
    return frame[~conf].reset_index(drop=True), frame[conf].reset_index(drop=True)


def contract_dict(confirmation_n: int, k_candidates: int) -> dict[str, Any]:
    raw = make_contract_dict(confirmation_n=confirmation_n, effect=CONTRACT_EFFECT)
    raw["power"]["k_candidates"] = int(k_candidates)
    raw["commitments"] = {}
    raw["outcome"]["measured_by"] = "planted outcome (plasmode calibration on real WDBC measurements)"
    return raw


def run_one_real(payload: dict[str, Any]) -> dict[str, Any]:
    seed = int(payload["seed"])
    full, realised = plant(load_frame(), float(payload["target_effect"]), seed)
    discovery, confirmation = split(full, seed)
    raw = contract_dict(len(confirmation), int(payload["candidates"]))
    row = evaluate_replicate(payload, raw, discovery, confirmation, None, realised)
    row["realised_effect"] = realised
    return row


def run_real_calibration(
    root: str | Path,
    reps: int = 20,
    effects: tuple[float, ...] = (0.0, 0.03, 0.06, 0.10, 0.15),
    candidates: int = 32,
    permutations: int = 39,
    workers: int = 0,
) -> dict[str, Any]:
    root = Path(root)
    out_dir = ensure_dir(root / "calibration")
    workers = workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    jobs = [
        {"seed": 700000 + ei * 1000 + r, "target_effect": float(e), "candidates": candidates, "permutations": permutations}
        for ei, e in enumerate(effects)
        for r in range(reps)
    ]
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    print(f"Real-data calibration: {len(jobs)} replicates, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one_real, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:  # pragma: no cover - worker isolation
                rows.append({"ok": False, "seed": job["seed"], "target_effect": job["target_effect"],
                             "error": {"reason_code": "worker_error", "message": str(exc)}})
            if i % max(1, len(jobs) // 10) == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

    good = [r for r in rows if r.get("ok")]
    summary = summarize_replicates(good, effects)
    payload = {
        "schema": "autoresearch.oc_table/v1",
        "generated_at": now_iso(),
        "domain": "real data: WDBC (569 tumours), plasmode planted outcomes",
        "method": __doc__.strip(),
        "parameters": {
            "reps_per_effect": reps,
            "effects": list(effects),
            "candidates": candidates,
            "permutations": permutations,
            "patients": "569 tumours (no groups)",
            "confirmation_groups": f"{CONFIRM_FRACTION:.0%} stratified split, re-drawn per replicate",
            "contract_effect": CONTRACT_EFFECT,
            "workers": workers,
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "failed_replicates": [r for r in rows if not r.get("ok")],
        "fpr_alpha": summary["fpr_alpha"],
        "fpr_decision": summary["fpr_decision"],
        "adversary": summary["adversary"],
        "rows": summary["rows"],
        "replicates": [
            {k: r.get(k) for k in (
                "seed", "target_effect", "true_effect", "tier", "verdict", "power_verdict", "power_mde_search",
                "delta", "ci_low", "ci_high", "p_confirmation", "p_selection", "detected_confirmed",
                "detected_affirmed", "detected_without_adversary", "adversary_killed", "adversary_results",
                "false_positive_alpha",
            )}
            for r in good
        ],
        "limitations": LIMITATIONS,
    }
    atomic_write_json(out_dir / "oc_table_real.json", payload)
    md = _render_oc_markdown(payload).replace(
        "# Operating characteristics of the certification gate",
        "# Operating characteristics of the certification gate on real data",
    )
    (out_dir / "oc_table_real.md").write_text(md, encoding="utf-8", newline="\n")
    return payload
