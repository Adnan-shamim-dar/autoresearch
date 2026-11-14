from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .contract import Contract, FeatureSpec
from .utils import normalize_weights


@dataclass
class Candidate:
    id: str
    name: str
    mechanism: str
    family: str
    params: dict[str, Any]
    source: str = "operator"
    parents: list[str] = field(default_factory=list)
    generation: int = 0

    @property
    def complexity(self) -> int:
        return candidate_complexity(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "mechanism": self.mechanism,
            "family": self.family,
            "params": self.params,
            "source": self.source,
            "parents": list(self.parents),
            "generation": int(self.generation),
            "complexity": int(self.complexity),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Candidate":
        return cls(
            id=str(d["id"]),
            name=str(d.get("name", d["id"])),
            mechanism=str(d.get("mechanism", "")),
            family=str(d.get("family", "unknown")),
            params=dict(d.get("params", {})),
            source=str(d.get("source", "operator")),
            parents=list(d.get("parents", [])),
            generation=int(d.get("generation", 0)),
        )


def candidate_complexity(candidate: Candidate) -> int:
    params = candidate.params
    if "column" in params:
        return 1
    weights = params.get("weights", {}) or {}
    n_terms = sum(1 for v in weights.values() if abs(float(v)) >= 0.02)
    gates = params.get("gates", []) or []
    transforms = params.get("transforms", {}) or {}
    penalty = sum(0.25 for t in transforms.values() if t not in ("rank", None))
    return int(round(n_terms + 2 * len(gates) + penalty))


def human_weights(weights: dict[str, float], transforms: dict[str, str] | None = None) -> str:
    transforms = transforms or {}
    parts = []
    for k, v in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        t = transforms.get(k, "rank")
        parts.append(f"{k}[{t}]={v:+.2f}")
    return ", ".join(parts)


class FeatureMatrix:
    """Precomputed, group-local feature transforms.

    Ranks are computed *within each group*. They use features only, never the
    outcome, and are therefore safe to recompute on a held-out confirmation set.
    Binary thresholds and standardisation statistics are fitted on discovery and
    carried inside the candidate specification so confirmation never re-fits on
    its own labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        contract: Contract,
        fit: bool = True,
        fit_stats: dict[str, Any] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.contract = contract
        self.features: list[FeatureSpec] = contract.features
        self.group_col = contract.group_column
        if self.group_col and self.group_col in self.df.columns:
            self.groups = self.df[self.group_col].astype(str).to_numpy()
        else:
            self.groups = np.array(["__all__"] * len(self.df))
        self.fit = fit
        self.fit_stats: dict[str, Any] = dict(fit_stats or {})
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        if fit:
            self._compute_fit_stats()

    # ------------------------------------------------------------------
    # Fit statistics
    # ------------------------------------------------------------------
    def _compute_fit_stats(self) -> None:
        for f in self.features:
            if f.name not in self.df.columns:
                continue
            x = pd.to_numeric(self.df[f.name], errors="coerce").astype(float)
            stats = {
                "mean": float(x.mean()),
                "std": float(x.std(ddof=0)) if x.std(ddof=0) > 0 else 1.0,
                "median": float(x.median()),
            }
            if f.transform == "binary_median":
                stats["threshold"] = stats["median"]
            self.fit_stats[f.name] = stats

    def export_fit_stats(self, feature_names: list[str] | None = None) -> dict[str, Any]:
        names = feature_names or [f.name for f in self.features]
        return {k: dict(v) for k, v in self.fit_stats.items() if k in names}

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    def _raw(self, name: str) -> np.ndarray:
        return pd.to_numeric(self.df[name], errors="coerce").astype(float).to_numpy()

    def _group_rank(self, x: np.ndarray) -> np.ndarray:
        """Within-group average ranks scaled to [0, 1]; singleton groups get 0.5.

        Vectorised with one groupby: the per-group Python loop this replaces
        dominated the cost of every search, null and attack.
        """
        s = pd.Series(np.asarray(x, dtype=float))
        by = s.groupby(self.groups, sort=False)
        r = by.rank(method="average").to_numpy()
        n = by.transform("size").to_numpy().astype(float)
        out = np.full(len(s), 0.5, dtype=float)
        multi = n > 1
        out[multi] = (r[multi] - 1.0) / (n[multi] - 1.0)
        return out

    def _group_z(self, x: np.ndarray) -> np.ndarray:
        s = pd.Series(np.asarray(x, dtype=float))
        by = s.groupby(self.groups, sort=False)
        mu = by.transform("mean").to_numpy()
        sd = by.transform(lambda v: v.std(ddof=0)).to_numpy()
        ok = np.isfinite(sd) & (sd > 0)
        out = np.zeros(len(s), dtype=float)
        out[ok] = np.clip((s.to_numpy()[ok] - mu[ok]) / sd[ok], -4, 4)
        return out

    def array(self, feature: FeatureSpec, transform: str | None = None) -> np.ndarray:
        t = (transform or feature.transform or "rank").lower()
        key = (feature.name, t)
        if key in self._cache:
            return self._cache[key]
        x = self._raw(feature.name)
        # Missing values are imputed with the discovery median / 0 after ranking.
        if np.isnan(x).any():
            fill = self.fit_stats.get(feature.name, {}).get("median", float(np.nanmedian(x)))
            if not np.isfinite(fill):
                fill = 0.0
            x = np.where(np.isnan(x), fill, x)
        if t == "rank":
            arr = self._group_rank(x)
        elif t == "z":
            arr = self._group_z(x)
        elif t == "log_rank":
            sx = np.sign(x) * np.log1p(np.abs(x))
            arr = self._group_rank(sx)
        elif t == "binary_median":
            stats = self.fit_stats.get(feature.name, {})
            thr = float(stats.get("threshold", stats.get("median", 0.0)))
            arr = (x >= thr).astype(float)
        elif t == "raw":
            stats = self.fit_stats.get(feature.name, {})
            mu = float(stats.get("mean", 0.0))
            sd = float(stats.get("std", 1.0)) or 1.0
            arr = (x - mu) / sd
        else:
            raise ValueError(f"Unknown transform {t!r}")
        if feature.direction == "lower_better":
            if t == "binary_median":
                arr = 1.0 - arr
            else:
                arr = -arr if t == "z" else 1.0 - arr
        self._cache[key] = arr.astype(float)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_candidate(self, candidate: Candidate) -> np.ndarray:
        params = candidate.params
        if "column" in params:
            col = str(params["column"])
            if col not in self.df.columns:
                raise ValueError(f"Baseline score column {col!r} is absent")
            s = pd.to_numeric(self.df[col], errors="coerce").astype(float).to_numpy()
            if not params.get("higher_is_better", True):
                s = -s
            return s
        return self.score_weights(
            weights=params.get("weights", {}) or {},
            transforms=params.get("transforms", {}) or {},
            gates=params.get("gates", []) or [],
        )

    def score_weights(
        self,
        weights: dict[str, float],
        transforms: dict[str, str] | None = None,
        gates: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        transforms = transforms or {}
        by_name = {f.name: f for f in self.features}
        n = len(self.df)
        score = np.zeros(n, dtype=float)
        for name, w in weights.items():
            if abs(float(w)) < 1e-12:
                continue
            f = by_name.get(name)
            if f is None:
                raise ValueError(f"Candidate uses undeclared feature {name!r}")
            score += float(w) * self.array(f, transforms.get(name))
        for gate in gates or []:
            name = str(gate.get("feature"))
            f = by_name.get(name)
            if f is None:
                continue
            arr = self.array(f, transforms.get(name, gate.get("transform")))
            op = str(gate.get("op", ">="))
            thr = float(gate.get("threshold", 0.5))
            if op == ">=":
                mask = arr >= thr
            elif op == "<=":
                mask = arr <= thr
            elif op == ">":
                mask = arr > thr
            elif op == "<":
                mask = arr < thr
            else:
                mask = arr >= thr
            gw = gate.get("weights", {}) or {}
            add = np.zeros(n, dtype=float)
            for gname, w in gw.items():
                gf = by_name.get(gname)
                if gf is None:
                    continue
                add += float(w) * self.array(gf, transforms.get(gname))
            score = np.where(mask, score + add, score)
        return score


def baseline_scores(df: pd.DataFrame, contract: Contract) -> np.ndarray:
    fm = FeatureMatrix(df, contract, fit=False, fit_stats={})
    return fm.score_candidate(contract.baseline_candidate())


def make_candidate(
    weights: dict[str, float],
    mechanism: str,
    family: str,
    cid: str | None = None,
    name: str | None = None,
    transforms: dict[str, str] | None = None,
    gates: list[dict[str, Any]] | None = None,
    fit_stats: dict[str, Any] | None = None,
    source: str = "operator",
    parents: list[str] | None = None,
    generation: int = 0,
) -> Candidate:
    w = normalize_weights(weights)
    params: dict[str, Any] = {
        "weights": w,
        "transforms": dict(transforms or {}),
        "gates": list(gates or []),
    }
    if fit_stats:
        # Keep only the stats needed by non-rank transforms.
        needed = {k: v for k, v in fit_stats.items() if k in w or any(k in g.get("weights", {}) for g in (gates or []))}
        if needed:
            params["fit_stats"] = needed
    if not cid:
        from .utils import sha256_obj

        cid = "c-" + sha256_obj(params)[:10]
    return Candidate(
        id=cid,
        name=name or cid,
        mechanism=mechanism,
        family=family,
        params=params,
        source=source,
        parents=list(parents or []),
        generation=generation,
    )


def candidate_from_dict(d: dict[str, Any]) -> Candidate:
    return Candidate.from_dict(d)
