"""Discovery Validity Benchmark: how often does a discovery agent claim something false?

Each task is a real dataset with an outcome planted from a known model
(plasmode). Real measurements, real correlations and real sample sizes are
kept; only the outcome is drawn, so the truth is known. Half of the tasks are
null: the deployed baseline is the whole truth, and any claim to beat it is a
false discovery.

An agent sees the discovery rows with outcomes and the remaining rows without
outcomes. It returns a claim (does a rule beat the baseline?) and its rule.
The benchmark scores the rule against the planting model: a claim is a true
discovery only if the task has a real effect and the rule's expected AUC beats
the baseline's over fresh outcome draws.

Why it can't be gamed: tasks are drawn from a seed the agent never sees
(commit-reveal through ``salt``), outcomes are re-drawn per task so memorising
a public dataset doesn't help, and the truth never leaves this process.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics
from .demo import _sigmoid
from .utils import atomic_write_json, ensure_dir, now_iso, rng_for, sha256_obj, sha256_text, stable_seed

MIN_EFFECT = 0.05
EFFECT_SIZES = (0.08, 0.12, 0.18)
FAMILIES = ("improve", "dredge")
DISCOVERY_FRACTION = 0.6
SCALE = 1.2
TRUE_GAIN_FLOOR = 0.0  # lenient: any positive expected gain over the baseline counts as true
MC_DRAWS = 300


# ---------------------------------------------------------------------------
# Real datasets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Source:
    name: str
    citation: str
    max_rows: int | None


SOURCES = (
    Source("breast_cancer", "Wolberg, Street and Mangasarian (1995), UCI; via scikit-learn", None),
    Source("diabetes", "Efron, Hastie, Johnstone and Tibshirani (2004); via scikit-learn", None),
    Source("california_housing", "Pace and Barry (1997), 1990 US census; via scikit-learn", 3000),
    Source("covertype", "Blackard and Dean (1999), US Forest Service; via scikit-learn", 3000),
)


@lru_cache(maxsize=None)
def load_source(name: str) -> pd.DataFrame:
    """Numeric real features only; outcomes are never taken from the source."""
    from sklearn import datasets

    if name == "breast_cancer":
        df = datasets.load_breast_cancer(as_frame=True).frame.iloc[:, :10]
    elif name == "diabetes":
        df = datasets.load_diabetes(as_frame=True, scaled=False).frame.drop(columns="target")
    elif name == "california_housing":
        df = datasets.fetch_california_housing(as_frame=True).frame.drop(columns="MedHouseVal")
    elif name == "covertype":
        df = datasets.fetch_covtype(as_frame=True).frame.iloc[:, :10]
    else:
        raise KeyError(name)
    df = df.rename(columns=lambda c: "f_" + "".join(ch if ch.isalnum() else "_" for ch in str(c)).lower())
    return df.astype(float).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def z_rank(x: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(stats.rankdata(x) / (len(x) + 1))


@dataclass
class Task:
    task_id: str
    source: str
    features: pd.DataFrame  # unit_id + real features, all rows
    y: np.ndarray  # planted outcome, all rows
    is_discovery: np.ndarray
    baseline: str
    truth_prob: np.ndarray  # P(y=1) under the planting model (never shown to agents)
    is_null: bool
    planted_effect: float
    family: str = "improve"
    truth_features: dict[str, float] = field(default_factory=dict)

    def public_spec(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "unit": "unit_id",
            "outcome": "y",
            "features": [c for c in self.features.columns if c != "unit_id"],
            "baseline": {"column": self.baseline, "higher_is_better": True},
            "question": (
                f"Is there a rule over these features that ranks y better than the baseline "
                f"'{self.baseline}' by at least {MIN_EFFECT} AUC in the population these units come from?"
            ),
            "min_effect": MIN_EFFECT,
            "n_discovery": int(self.is_discovery.sum()),
            "n_unlabelled": int((~self.is_discovery).sum()),
            "source": self.source,
            "family": self.family,
        }

    def discovery_frame(self) -> pd.DataFrame:
        df = self.features[self.is_discovery].copy()
        df["y"] = self.y[self.is_discovery]
        return df.reset_index(drop=True)


def _intercept(signal: np.ndarray, prevalence: float) -> float:
    lo, hi = -14.0, 14.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(mid + signal).mean() < prevalence else (lo, mid)
    return (lo + hi) / 2


def make_task(salt: str, index: int) -> Task:
    seed = stable_seed("bench-task", salt, index)
    rng = rng_for("bench-task", salt, index)
    source = SOURCES[index % len(SOURCES)]
    full = load_source(source.name)
    if source.max_rows and len(full) > source.max_rows:
        full = full.iloc[np.sort(rng.choice(len(full), source.max_rows, replace=False))].reset_index(drop=True)
    cols = list(full.columns)
    baseline = str(rng.choice(cols))
    others = [c for c in cols if c != baseline]
    is_null = bool(rng.random() < 0.5)
    # "improve": the baseline carries real signal and the question is whether a rule adds to it.
    # "dredge": the deployed criterion is uninformative; under the null nothing predicts y.
    family = FAMILIES[(index // len(SOURCES)) % len(FAMILIES)]
    base_weight = 1.0 if family == "improve" else 0.0
    target = 0.0 if is_null else float(rng.choice(EFFECT_SIZES))
    extra = list(rng.choice(others, size=int(rng.integers(1, 4)), replace=False))
    signs = {c: float(rng.choice([-1.0, 1.0])) for c in extra}
    prevalence = float(rng.uniform(0.2, 0.5))

    zb = z_rank(full[baseline].to_numpy())
    zx = np.mean([signs[c] * z_rank(full[c].to_numpy()) for c in extra], axis=0)
    uniforms = rng.random(len(full))

    def draw(beta: float) -> tuple[np.ndarray, np.ndarray, float]:
        truth = base_weight * zb + beta * zx
        p = _sigmoid(_intercept(SCALE * truth, prevalence) + SCALE * truth)
        y = (uniforms < p).astype(int)
        return y, p, float(metrics.auc(y, truth) - metrics.auc(y, zb))

    if is_null:
        y, p, realised = draw(0.0)
    else:
        lo, hi = 0.0, 40.0
        best = draw(hi)
        for _ in range(40):
            mid = (lo + hi) / 2
            cand = draw(mid)
            if abs(cand[2] - target) < abs(best[2] - target):
                best = cand
            lo, hi = (mid, hi) if cand[2] < target else (lo, mid)
        y, p, realised = best

    is_disc = np.zeros(len(full), dtype=bool)
    is_disc[rng.permutation(len(full))[: int(round(len(full) * DISCOVERY_FRACTION))]] = True
    features = full.copy()
    features.insert(0, "unit_id", [f"u{seed % 100000:05d}-{i:05d}" for i in range(len(full))])
    return Task(
        task_id=f"t{index:04d}",
        source=source.name,
        features=features,
        y=y,
        is_discovery=is_disc,
        baseline=baseline,
        truth_prob=p,
        is_null=is_null,
        planted_effect=0.0 if is_null else realised,
        family=family,
        truth_features={} if is_null else signs,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def expected_auc_gain(prob: np.ndarray, rule: np.ndarray, baseline: np.ndarray, draws: int, seed: int) -> float:
    """E[AUC(rule) - AUC(baseline)] over fresh outcome draws from the planting model."""
    rng = np.random.default_rng(seed)
    y = (rng.random((draws, len(prob))) < prob).astype(float)
    n1 = y.sum(axis=1)
    n0 = len(prob) - n1
    ok = (n1 > 0) & (n0 > 0)

    def auc(scores: np.ndarray) -> np.ndarray:
        r = stats.rankdata(scores)
        return (y @ r - n1 * (n1 + 1) / 2) / np.where(ok, n1 * n0, 1)

    return float(np.mean((auc(rule) - auc(baseline))[ok]))


def rule_scores(task: Task, claim: dict[str, Any]) -> np.ndarray | None:
    ids = task.features["unit_id"].to_numpy()
    if claim.get("scores") is not None:
        s = claim["scores"]
        arr = np.array([s.get(u, np.nan) for u in ids], dtype=float) if isinstance(s, dict) else np.asarray(s, dtype=float)
        return arr if arr.shape == (len(ids),) and np.all(np.isfinite(arr)) else None
    weights = claim.get("weights") or {}
    valid = {k: float(v) for k, v in weights.items() if k in task.features.columns and k != "unit_id"}
    if not valid:
        return None
    return sum(w * z_rank(task.features[k].to_numpy()) for k, w in valid.items())


#: A non-claim with one of these ``detail["stopped"]`` reasons, or with an ``error``, is not a
#: decision: the agent never answered. Scoring it as a correct silence would reward failure, so
#: these tasks are labelled ``no_response`` and left out of the rate denominators.
NON_RESPONSE_STOPS = frozenset({"turn limit", "no submission", "refusal"})


def responded(claim: dict[str, Any]) -> bool:
    """True when the agent returned a decision rather than crashing or running out of turns."""
    detail = claim.get("detail")
    if not isinstance(detail, dict):
        return True
    return not (detail.get("error") or str(detail.get("stopped", "")) in NON_RESPONSE_STOPS)


def score_claim(task: Task, claim: dict[str, Any], seed: int) -> dict[str, Any]:
    claimed = bool(claim.get("claim"))
    out: dict[str, Any] = {"task_id": task.task_id, "source": task.source, "family": task.family, "is_null": task.is_null,
                           "planted_effect": task.planted_effect, "claimed": claimed, "responded": True}
    if not claimed:
        refused = bool(claim.get("refused") or (claim.get("detail") or {}).get("tier") == "power_refusal")
        out["refused_underpowered"] = refused
        if not responded(claim):
            out.update(responded=False, outcome="no_response")
            return out
        out["outcome"] = "missed" if not task.is_null else "correct_silence"
        return out
    scores = rule_scores(task, claim)
    if scores is None:
        out.update(outcome="false_discovery", expected_gain=None, note="claim without a usable rule")
        return out
    zb = z_rank(task.features[task.baseline].to_numpy())
    gain = expected_auc_gain(task.truth_prob, scores, zb, MC_DRAWS, seed)
    sealed = ~task.is_discovery
    out["expected_gain"] = gain
    out["sealed_delta"] = float(metrics.auc(task.y[sealed], scores[sealed]) - metrics.auc(task.y[sealed], zb[sealed]))
    out["outcome"] = "true_discovery" if (not task.is_null and gain > TRUE_GAIN_FLOOR) else "false_discovery"
    # Strict: the question asked was "by at least MIN_EFFECT".
    out["strict_true"] = bool(not task.is_null and gain >= MIN_EFFECT)
    return out


# ---------------------------------------------------------------------------
# Reference agents
# ---------------------------------------------------------------------------

def _candidate_rules(disc: pd.DataFrame, features: list[str], baseline: str, k: int, seed: int) -> list[dict[str, float]]:
    """What a typical automated analyst tries: single features and signed combinations with the baseline."""
    rng = rng_for("bench-rules", seed)
    y = disc["y"].to_numpy()
    sign = {f: (1.0 if stats.spearmanr(disc[f], y).statistic >= 0 else -1.0) for f in features}
    rules = [{f: sign[f]} for f in features if f != baseline]
    while len(rules) < k:
        chosen = rng.choice([f for f in features if f != baseline], size=int(rng.integers(1, 4)), replace=False)
        rule = {baseline: 1.0}
        rule.update({f: sign[f] * float(rng.choice([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5])) for f in chosen})
        rules.append(rule)
    return rules[:k]


def _scores(df: pd.DataFrame, rule: dict[str, float]) -> np.ndarray:
    return sum(w * z_rank(df[f].to_numpy()) for f, w in rule.items())


def _bootstrap_p(y: np.ndarray, s: np.ndarray, b: np.ndarray, seed: int, n: int = 200) -> float:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            deltas.append(metrics.auc(y[i], s[i]) - metrics.auc(y[i], b[i]))
    return float((np.sum(np.asarray(deltas) <= 0) + 1) / (len(deltas) + 1))


def _best_rule(df: pd.DataFrame, rules: list[dict[str, float]], baseline: str) -> tuple[dict[str, float], float]:
    y = df["y"].to_numpy()
    b = z_rank(df[baseline].to_numpy())
    gains = [metrics.auc(y, _scores(df, r)) - metrics.auc(y, b) for r in rules]
    i = int(np.argmax(gains))
    return rules[i], float(gains[i])


def agent_naive(task: Task, seed: int, k: int = 60) -> dict[str, Any]:
    """Search K rules on all discovery data and report the best if its in-sample p < 0.05."""
    disc, spec = task.discovery_frame(), task.public_spec()
    rule, gain = _best_rule(disc, _candidate_rules(disc, spec["features"], task.baseline, k, seed), task.baseline)
    p = _bootstrap_p(disc["y"].to_numpy(), _scores(disc, rule), z_rank(disc[task.baseline].to_numpy()), seed)
    return {"claim": bool(gain > 0 and p < 0.05), "weights": rule, "detail": {"in_sample_gain": gain, "p": p}}


def agent_holdout(task: Task, seed: int, k: int = 60) -> dict[str, Any]:
    """Search on 70% of discovery, test the single winner on the other 30%."""
    disc, spec = task.discovery_frame(), task.public_spec()
    idx = rng_for("bench-holdout", seed).permutation(len(disc))
    cut = int(len(disc) * 0.7)
    search, test = disc.iloc[idx[:cut]], disc.iloc[idx[cut:]]
    rule, _ = _best_rule(search, _candidate_rules(search, spec["features"], task.baseline, k, seed), task.baseline)
    y = test["y"].to_numpy()
    s, b = _scores(test, rule), z_rank(test[task.baseline].to_numpy())
    gain = metrics.auc(y, s) - metrics.auc(y, b)
    p = _bootstrap_p(y, s, b, seed)
    return {"claim": bool(gain > 0 and p < 0.05), "weights": rule, "detail": {"holdout_gain": gain, "p": p}}


def agent_iterating(task: Task, seed: int, k: int = 60, rounds: int = 10) -> dict[str, Any]:
    """An agentic analyst: test the best rule on the holdout; if it fails, try the next one.

    This is what a tool-using agent does when a check fails: it revises and
    re-checks against the same held-out data. Each check is honest; the
    reuse is what inflates the error rate.
    """
    disc, spec = task.discovery_frame(), task.public_spec()
    idx = rng_for("bench-holdout", seed).permutation(len(disc))
    cut = int(len(disc) * 0.7)
    search, test = disc.iloc[idx[:cut]], disc.iloc[idx[cut:]]
    rules = _candidate_rules(search, spec["features"], task.baseline, k, seed)
    ys, bs = search["y"].to_numpy(), z_rank(search[task.baseline].to_numpy())
    order = np.argsort([-(metrics.auc(ys, _scores(search, r)) - metrics.auc(ys, bs)) for r in rules])
    y, b = test["y"].to_numpy(), z_rank(test[task.baseline].to_numpy())
    for attempt, i in enumerate(order[:rounds], 1):
        s = _scores(test, rules[int(i)])
        gain = metrics.auc(y, s) - metrics.auc(y, b)
        p = _bootstrap_p(y, s, b, seed + attempt)
        if gain > 0 and p < 0.05:
            return {"claim": True, "weights": rules[int(i)], "detail": {"attempts": attempt, "holdout_gain": gain, "p": p}}
    return {"claim": False, "weights": rules[int(order[0])], "detail": {"attempts": rounds}}


def _contract_for(task: Task, disc: pd.DataFrame, confirmation_n: int, k: int) -> dict[str, Any]:
    from .wdbc import make_contract_dict

    raw = make_contract_dict(confirmation_n=confirmation_n, effect=MIN_EFFECT)
    y = disc["y"].to_numpy()
    feats = [c for c in task.features.columns if c != "unit_id"]
    raw["unit"] = "unit_id"
    raw["outcome"] = {"column": "y", "type": "binary", "measured_by": "planted outcome (benchmark task)"}
    raw["candidates"]["features"] = [
        {"name": f, "direction": "higher_better" if stats.spearmanr(disc[f], y).statistic >= 0 else "lower_better",
         "transform": "rank"}
        for f in feats
    ]
    raw["baseline"] = {"name": "deployed_baseline", "column": task.baseline, "higher_is_better": True,
                       "description": "the task's deployed criterion"}
    raw["commitments"] = {}
    raw["power"]["k_candidates"] = k
    raw["decision"]["action"] = "adopt the rule in place of the baseline"
    raw["provenance"]["notes"] = f"benchmark task on {task.source}"
    return raw


def agent_autoresearch_proceed(task: Task, seed: int) -> dict[str, Any]:
    """The engine with the contract option ``if_underpowered: proceed``."""
    return agent_autoresearch(task, seed, if_underpowered="proceed")


def agent_autoresearch(task: Task, seed: int, k: int = 32, permutations: int = 39,
                       if_underpowered: str = "refuse") -> dict[str, Any]:
    """The engine: price the search, seal 40% of discovery, confirm one rule, red team, gate."""
    from .calibration import evaluate_replicate
    from .scoring import FeatureMatrix

    disc = task.discovery_frame()
    rng = rng_for("bench-ar-split", seed)
    conf = np.zeros(len(disc), dtype=bool)
    for label in (0, 1):
        idx = np.where(disc["y"].to_numpy() == label)[0]
        conf[rng.permutation(idx)[: int(round(len(idx) * 0.4))]] = True
    search, sealed = disc[~conf].reset_index(drop=True), disc[conf].reset_index(drop=True)
    raw = _contract_for(task, search, len(sealed), k)
    raw["decision"]["if_underpowered"] = if_underpowered
    if if_underpowered == "proceed":
        raw["decision"]["if_underpowered_chosen_by"] = "benchmark agent configuration"
    payload = {"seed": seed, "target_effect": -1.0, "candidates": k, "permutations": permutations, "return_objects": True}
    row = evaluate_replicate(payload, raw, search, sealed, None, float("nan"))
    if not row.get("ok"):
        return {"claim": False, "detail": {"error": row.get("error")}}
    objects = row.pop("_objects")
    cand, contract = objects["candidate"], objects["contract"]
    allf = task.features.copy()
    allf["y"] = 0  # placeholder column so the matrix builds; outcomes are not used for scoring
    fm = FeatureMatrix(allf, contract, fit=False, fit_stats=cand.params.get("fit_stats", {}))
    scores = fm.score_candidate(cand)
    return {
        "claim": bool(row.get("detected_affirmed")),
        "scores": [float(x) for x in scores],
        "detail": {"tier": row.get("tier"), "power_verdict": row.get("power_verdict"),
                   "power_se_conf": row.get("power_se_conf"), "se_confirmation": row.get("se_confirmation"),
                   "power_mde_plain": row.get("power_mde_plain"),
                   "delta": row.get("delta"), "p_selection": row.get("p_selection")},
    }


def external_agent(command: str) -> Callable[[Task, int], dict[str, Any]]:
    """Any agent, any language: it gets a task folder and writes claim.json.

    The folder holds task.json (the public spec), discovery.csv (with y) and
    unlabelled.csv (features only). ``{task_dir}`` in the command is replaced.
    """
    def run(task: Task, seed: int) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"bench-{task.task_id}-") as tmp:
            d = Path(tmp)
            (d / "task.json").write_text(json.dumps(task.public_spec(), indent=2), encoding="utf-8")
            task.discovery_frame().to_csv(d / "discovery.csv", index=False)
            task.features[~task.is_discovery].to_csv(d / "unlabelled.csv", index=False)
            tokens = shlex.split(command, posix=os.name != "nt")
            if os.name == "nt":  # non-POSIX splitting keeps the quotes
                tokens = [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]
            args = [a.replace("{task_dir}", str(d)) for a in tokens]
            subprocess.run(args, check=False, timeout=3600)
            path = d / "claim.json"
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"claim": False}
    return run


AGENTS: dict[str, Callable[[Task, int], dict[str, Any]]] = {
    "naive_search": agent_naive,
    "holdout_search": agent_holdout,
    "iterating_agent": agent_iterating,
    "autoresearch": agent_autoresearch,
    "autoresearch_proceed": agent_autoresearch_proceed,
}


# ---------------------------------------------------------------------------
# Runner and leaderboard
# ---------------------------------------------------------------------------

def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def run_task(job: tuple[str, int, str, str | None]) -> dict[str, Any]:
    salt, index, agent_name, command = job
    task = make_task(salt, index)
    agent = external_agent(command) if command else AGENTS[agent_name]
    seed = stable_seed("bench-agent", salt, index) % (2**31 - 1)
    t0 = time.time()
    try:
        claim = agent(task, seed)
    except Exception as exc:  # an agent crash is a non-claim, recorded
        claim = {"claim": False, "detail": {"error": f"{type(exc).__name__}: {exc}"}}
    row = score_claim(task, claim, seed)
    row.update(agent=agent_name, seconds=round(time.time() - t0, 2), detail=claim.get("detail"))
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Rates are over tasks the agent actually answered. A task it never answered is no evidence
    # either way, and counting a non-answer as a correct silence would flatter a failing agent.
    answered = [r for r in rows if r.get("responded", True)]
    nulls = [r for r in answered if r["is_null"]]
    effects = [r for r in answered if not r["is_null"]]
    claims = [r for r in rows if r["claimed"]]
    false = [r for r in claims if r["outcome"] == "false_discovery"]
    true = [r for r in claims if r["outcome"] == "true_discovery"]
    fp_null = sum(1 for r in nulls if r["claimed"])
    out = {
        "tasks": len(rows), "answered": len(answered), "no_response": len(rows) - len(answered),
        "response_rate": len(answered) / len(rows) if rows else None,
        "null_tasks": len(nulls), "effect_tasks": len(effects), "claims": len(claims),
        "false_discoveries": len(false), "true_discoveries": len(true),
        "false_claim_share": len(false) / len(claims) if claims else None,
        "false_claim_share_ci": _wilson(len(false), len(claims)),
        "null_claim_rate": fp_null / len(nulls) if nulls else None,
        "null_claim_rate_ci": _wilson(fp_null, len(nulls)),
        "power": len(true) / len(effects) if effects else None,
        "power_ci": _wilson(len(true), len(effects)),
    }
    strict_false = sum(1 for r in claims if not r.get("strict_true"))
    out.update({
        "strict_false_discoveries": strict_false,
        "strict_false_claim_share": strict_false / len(claims) if claims else None,
        "strict_power": sum(1 for r in claims if r.get("strict_true")) / len(effects) if effects else None,
        "refused_underpowered": sum(1 for r in rows if r.get("refused_underpowered")),
        "dredge_null_claim_rate": (lambda d: sum(r["claimed"] for r in d) / len(d) if d else None)(
            [r for r in nulls if r.get("family") == "dredge"]),
        "dredge_null_tasks": sum(1 for r in nulls if r.get("family") == "dredge"),
    })
    out["by_source"] = {}
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        c = [r for r in sub if r["claimed"]]
        out["by_source"][src] = {
            "tasks": len(sub), "claims": len(c),
            "false_discoveries": sum(1 for r in c if r["outcome"] == "false_discovery"),
            "true_discoveries": sum(1 for r in c if r["outcome"] == "true_discovery"),
        }
    return out


def rebuild_leaderboard(root: str | Path) -> dict[str, Any]:
    """Re-summarise saved rows (after a scoring-label fix) without re-running any agent."""
    out_dir = Path(root) / "bench"
    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    for r in payload["rows"]:
        if not r["claimed"]:
            r["refused_underpowered"] = (r.get("detail") or {}).get("tier") == "power_refusal"
            r["responded"] = responded(r)
            if not r["responded"]:
                r["outcome"] = "no_response"
    agents = list(payload["leaderboard"])
    payload["leaderboard"] = {a: summarise([r for r in payload["rows"] if r["agent"] == a]) for a in agents}
    payload["results_hash"] = sha256_obj({k: v for k, v in payload.items() if k not in ("generated_at", "results_hash")})
    atomic_write_json(out_dir / "results.json", payload)
    (out_dir / "LEADERBOARD.md").write_text(to_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def run_bench(
    root: str | Path,
    agents: list[str],
    n_tasks: int = 200,
    salt: str = "public-dev-v1",
    reveal_salt: bool = True,
    command: str | None = None,
    workers: int = 0,
) -> dict[str, Any]:
    root = Path(root)
    out_dir = ensure_dir(root / "bench")
    workers = workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    for src in SOURCES:  # download once, before the workers start
        load_source(src.name)
    jobs = [(salt, i, a, command if a.startswith("external") else None) for a in agents for i in range(n_tasks)]
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_task, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % max(1, len(jobs) // 10) == 0:
                print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["agent"], r["task_id"]))
    board = {a: summarise([r for r in rows if r["agent"] == a]) for a in agents}
    payload = {
        "schema": "autoresearch.bench/v1",
        "generated_at": now_iso(),
        "benchmark": "Discovery Validity Benchmark",
        "method": __doc__.strip(),
        "salt_sha256": sha256_text(salt),
        "salt": salt if reveal_salt else None,
        "parameters": {"n_tasks": n_tasks, "min_effect": MIN_EFFECT, "effect_sizes": list(EFFECT_SIZES),
                       "discovery_fraction": DISCOVERY_FRACTION, "mc_draws": MC_DRAWS,
                       "true_discovery_rule": "task has a planted effect and the rule's expected AUC gain over the baseline is > 0",
                       "elapsed_seconds": round(time.time() - t0, 1)},
        "sources": [{"name": s.name, "citation": s.citation, "max_rows": s.max_rows} for s in SOURCES],
        "leaderboard": board,
        "rows": rows,
    }
    payload["results_hash"] = sha256_obj({k: v for k, v in payload.items() if k not in ("generated_at",)})
    atomic_write_json(out_dir / "results.json", payload)
    (out_dir / "LEADERBOARD.md").write_text(to_markdown(payload), encoding="utf-8", newline="\n")
    return payload


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _ci(ci: Any) -> str:
    lo, hi = ci
    return "n/a" if lo is None else f"{100 * lo:.0f}–{100 * hi:.0f}%"


def to_markdown(payload: dict[str, Any]) -> str:
    p = payload["parameters"]
    lines = [
        "# Discovery Validity Benchmark: leaderboard",
        "",
        f"{p['n_tasks']} tasks on 4 real datasets with planted outcomes; about half are null. "
        f"Salt SHA-256 `{payload['salt_sha256'][:16]}…`. Results hash `{payload['results_hash'][:16]}…`.",
        "",
        "| Agent | Answered | Claims | False claims | Claims on null tasks | Claims on pure-noise (dredge) nulls | Real effects found | Strict: false claims (gain < min effect) | Strict: effects found | Refused as underpowered |",
        "|---|---|---:|---|---|---|---|---|---|---:|",
    ]
    for name, s in payload["leaderboard"].items():
        lines.append(
            f"| {name} | {s['answered']}/{s['tasks']} ({_pct(s['response_rate'])}) | {s['claims']} | {s['false_discoveries']} ({_pct(s['false_claim_share'])}) | "
            f"{_pct(s['null_claim_rate'])} ({_ci(s['null_claim_rate_ci'])}) | {_pct(s['dredge_null_claim_rate'])} of {s['dredge_null_tasks']} | "
            f"{_pct(s['power'])} ({_ci(s['power_ci'])}) | {s['strict_false_discoveries']} ({_pct(s['strict_false_claim_share'])}) | "
            f"{_pct(s['strict_power'])} | {s['refused_underpowered']} |"
        )
    lines += ["", "## By dataset", "", "| Agent | Dataset | Tasks | Claims | False | True |", "|---|---|---:|---:|---:|---:|"]
    for name, s in payload["leaderboard"].items():
        for src, b in s["by_source"].items():
            lines.append(f"| {name} | {src} | {b['tasks']} | {b['claims']} | {b['false_discoveries']} | {b['true_discoveries']} |")
    lines += [
        "",
        "## Agents",
        "",
        "- **naive_search**: tries 60 rules on all discovery data and reports the best if its in-sample p < 0.05. "
        "This is the default behaviour of an automated analyst that doesn't price its search.",
        "- **holdout_search**: the same search on 70% of the discovery data; the single winner is tested on the other 30%.",
        "- **iterating_agent**: like holdout_search, but when the check fails it tries the next-best rule against the same holdout, up to 10 times. This is how a tool-using agent behaves.",
        "- **autoresearch_proceed**: the same engine with the contract option `if_underpowered: proceed`: underpowered questions are run and labelled, not refused.",
        "- **autoresearch**: this engine. It prices the search, seals 40% of the discovery data, confirms one rule, runs the red team and the gate.",
        "",
        "## Rules",
        "",
        f"- A claim is a **true discovery** only if the task has a planted effect and the submitted rule's expected AUC "
        f"over {p['mc_draws']} fresh outcome draws beats the baseline's. This is lenient: any positive gain counts.",
        "- Every claim on a null task is a false discovery.",
        "- **Rates are over answered tasks.** If an agent crashes, declines or runs out of turns it is recorded as "
        "`no_response` and that task leaves both denominators: a non-answer is no evidence either way, and counting "
        "it as a correct silence would reward an agent for failing. Read the Answered column first — a low "
        "null-task claim rate on few answered tasks says little.",
        "- Datasets: " + "; ".join(f"{s['name']} ({s['citation']})" for s in payload["sources"]) + ".",
        "- Submit your own agent: `autoresearch bench --agent external --command \"python my_agent.py {task_dir}\"`. "
        "It receives task.json, discovery.csv and unlabelled.csv and writes claim.json "
        "(`{\"claim\": true, \"weights\": {feature: weight}}` or `\"scores\"`).",
        "",
    ]
    return "\n".join(lines)
