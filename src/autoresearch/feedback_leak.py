"""The feedback-leak test (spec §5).

"A proposer fed a leaking feedback channel (scores) measurably increases its
false-positive rate at the gate; with categorical feedback it does not. Build
this test - it is the proof the constraint matters."

Setup, repeated over independent pure-null datasets (outcomes permuted within
patient, so no candidate truly beats the baseline):

* One proposer, ``ScoreSeekingProposer``: a generic optimiser that climbs any
  number it is given, the way an LLM told its scores would. With no number it
  can only breed from candidates marked ``kept``.
* Two channels. The **leaky** channel reports each candidate's score on the
  data the gate will judge. The **guarded** channel is the engine's own:
  status from discovery evaluation, passed through
  :func:`autoresearch.proposer.sanitize_feedback`.
* The proposer nominates its best candidate; the gate tests it once on the
  confirmation data (paired group bootstrap; promoted when the CI excludes zero
  and one-sided p <= alpha). Under the null, every promotion is a false
  positive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics
from .contract import Contract
from .demo import generate_immunotherapy_data, make_contract_dict
from .evaluator import permute_within_groups
from .proposer import sanitize_feedback
from .scoring import Candidate, FeatureMatrix, make_candidate
from .utils import atomic_write_json, ensure_dir, now_iso, rng_for

CHANNELS = ("leaky", "guarded")


class ScoreSeekingProposer:
    """Mutation hill-climber over feature weights.

    Parents are the top candidates by score when feedback carries a number,
    otherwise candidates marked ``kept``. Nothing else differs between arms.
    """

    def __init__(self, features: list[str], max_features: int, fm: FeatureMatrix, rng: np.random.Generator):
        self.features = features
        self.max_features = max_features
        self.fm = fm
        self.rng = rng
        self.counter = 0

    def _make(self, weights: dict[str, float], generation: int) -> Candidate:
        items = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[: self.max_features]
        w = {k: float(v) for k, v in items if abs(v) > 1e-9} or {self.features[0]: 1.0}
        self.counter += 1
        return make_candidate(
            w,
            mechanism="leak-test hill-climber",
            family="leak_test",
            cid=f"lk{generation:02d}-{self.counter:04d}",
            fit_stats=self.fm.export_fit_stats(list(w)),
        )

    def initial(self, n: int) -> list[Candidate]:
        return [self._make({f: float(self.rng.normal()) for f in self.features}, 0) for _ in range(n)]

    def next_generation(
        self,
        population: list[Candidate],
        feedback: list[dict[str, Any]],
        n: int,
        generation: int,
        n_parents: int = 3,
    ) -> list[Candidate]:
        by_id = {c.id: c for c in population}
        scored = [f for f in feedback if isinstance(f.get("score"), (int, float)) and np.isfinite(f["score"])]
        if scored:
            parents = [by_id[f["id"]] for f in sorted(scored, key=lambda f: -f["score"])[:n_parents]]
        else:
            kept = [by_id[f["id"]] for f in feedback if f.get("status") == "kept" and f["id"] in by_id]
            parents = kept or population
            parents = [parents[i] for i in self.rng.permutation(len(parents))[:n_parents]]
        children = []
        for i in range(n):
            base = dict(parents[i % len(parents)].params["weights"])
            child = {f: base.get(f, 0.0) + float(self.rng.normal(scale=0.35)) for f in self.features}
            children.append(self._make(child, generation))
        return children


def _null_split(seed: int, n_patients: int) -> tuple[pd.DataFrame, pd.DataFrame, Contract]:
    df = generate_immunotherapy_data(seed=seed, n_patients=n_patients)
    groups = df["patient_id"].astype(str).to_numpy()
    df["immunogenic"] = permute_within_groups(df["immunogenic"].to_numpy(), groups, rng_for("leak-null", seed))
    pids = sorted(df["patient_id"].unique())
    conf = set(rng_for("leak-split", seed).permutation(pids)[: len(pids) // 2])
    discovery = df[~df["patient_id"].isin(conf)].reset_index(drop=True)
    confirmation = df[df["patient_id"].isin(conf)].reset_index(drop=True)
    contract = Contract(make_contract_dict(effect=0.04, k_candidates=64, confirmation_n=len(confirmation)))
    contract.validate_static()
    contract.attach_data(discovery, role="discovery")
    return discovery, confirmation, contract


def _point_delta(fm: FeatureMatrix, cand: Candidate, base: np.ndarray, y: np.ndarray) -> float:
    return float(metrics.auc(y, fm.score_candidate(cand)) - metrics.auc(y, base))


def run_replicate(
    seed: int,
    channel: str,
    n_patients: int = 40,
    generations: int = 20,
    per_generation: int = 16,
    alpha: float = 0.05,
    bootstrap: int = 300,
) -> dict[str, Any]:
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}")
    discovery, confirmation, contract = _null_split(seed, n_patients)
    fm_d = FeatureMatrix(discovery, contract, fit=True)
    features = [f.name for f in contract.features]
    rng = rng_for("leak-proposer", seed)  # same proposer randomness in both arms
    proposer = ScoreSeekingProposer(features, max_features=4, fm=fm_d, rng=rng)

    y_d = discovery[contract.outcome_column].to_numpy().astype(int)
    y_c = confirmation[contract.outcome_column].to_numpy().astype(int)
    g_c = confirmation[contract.group_column].astype(str).to_numpy()
    base_d = fm_d.score_candidate(contract.baseline_candidate())

    # Confirmation transforms use discovery fit statistics only; built once.
    fm_c = FeatureMatrix(confirmation, contract, fit=False, fit_stats=fm_d.export_fit_stats())
    base_c = fm_c.score_candidate(contract.baseline_candidate())

    def conf_scores(cand: Candidate) -> tuple[np.ndarray, np.ndarray]:
        return fm_c.score_candidate(cand), base_c

    population = proposer.initial(per_generation)
    history: list[tuple[Candidate, float, float]] = []  # (candidate, discovery delta, leaked score)
    for gen in range(1, generations + 1):
        disc = [_point_delta(fm_d, c, base_d, y_d) for c in population]
        cut = np.quantile(disc, 0.75)
        raw = [{"id": c.id, "status": "kept" if d >= cut else "rejected",
                "reason_code": "kept_top_fraction" if d >= cut else "rejected_below_cut"}
               for c, d in zip(population, disc)]
        leaked = []
        for c in population:
            s_c, s_b = conf_scores(c)
            leaked.append(float(metrics.auc(y_c, s_c) - metrics.auc(y_c, s_b)))
        history.extend(zip(population, disc, leaked))
        if channel == "leaky":
            feedback = [{**r, "score": sc} for r, sc in zip(raw, leaked)]
        else:
            feedback = sanitize_feedback([{**r, "score": sc} for r, sc in zip(raw, leaked)])
        if gen < generations:
            population = proposer.next_generation(population, feedback, per_generation, gen)

    # Nomination: by the leaked score when the channel carried one, otherwise
    # by the registered rule (best discovery delta).
    key = 2 if channel == "leaky" else 1
    nominee = max(history, key=lambda h: (h[key], h[0].id))[0]
    s_c, s_b = conf_scores(nominee)
    boot = metrics.group_bootstrap_deltas(y_c, g_c, s_c, s_b, n_boot=bootstrap, seed=seed)
    summ = metrics.bootstrap_summary(boot)
    promoted = bool(summ["ci_low"] > 0 and summ["p_one_sided"] <= alpha)
    return {
        "seed": int(seed),
        "channel": channel,
        "candidates_seen": len(history),
        "delta": float(summ["delta"]),
        "ci_low": float(summ["ci_low"]),
        "p_one_sided": float(summ["p_one_sided"]),
        "promoted": promoted,
    }


def run_experiment(
    replicates: int = 40,
    seed0: int = 5000,
    **kwargs: Any,
) -> dict[str, Any]:
    rows = [run_replicate(seed0 + r, ch, **kwargs) for r in range(replicates) for ch in CHANNELS]
    summary = {}
    for ch in CHANNELS:
        arm = [r for r in rows if r["channel"] == ch]
        k = sum(r["promoted"] for r in arm)
        lo, hi = _wilson(k, len(arm))
        summary[ch] = {
            "n": len(arm),
            "false_positives": int(k),
            "fpr": k / len(arm),
            "fpr_ci_low": lo,
            "fpr_ci_high": hi,
            "mean_delta": float(np.mean([r["delta"] for r in arm])),
        }
    return {
        "schema": "autoresearch.feedback_leak/v1",
        "generated_at": now_iso(),
        "parameters": {"replicates": replicates, "seed0": seed0, **kwargs},
        "summary": summary,
        "rows": rows,
    }


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def publish(root: str | Path, result: dict[str, Any]) -> Path:
    out = ensure_dir(Path(root) / "calibration")
    atomic_write_json(out / "feedback_leak.json", result)
    s = result["summary"]
    lines = [
        "# Feedback-leak test (spec §5)",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "Pure-null data; one score-seeking proposer; two feedback channels. Every promotion is a false positive.",
        "",
        "| Channel | Replicates | False positives | FPR | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for ch, label in (("leaky", "Leaky (scores on gate data)"), ("guarded", "Guarded (`sanitize_feedback`)")):
        a = s[ch]
        lines.append(f"| {label} | {a['n']} | {a['false_positives']} | {a['fpr']:.2f} | {a['fpr_ci_low']:.2f}-{a['fpr_ci_high']:.2f} |")
    lines += ["", f"Parameters: `{result['parameters']}`", ""]
    (out / "feedback_leak.md").write_text("\n".join(lines), encoding="utf-8")
    return out / "feedback_leak.md"
