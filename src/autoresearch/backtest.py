"""Replication back-test (spec milestone 9, proof 4).

Pre-registered in ``docs/backtest/PREREGISTRATION.md``. The rules are applied
blind: :func:`decide` receives a frame that holds only the information
available at original publication, and replication outcomes are joined only
after every decision has been made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .utils import atomic_write_json, sha256_file

INPUT_COLS = ("T.N.O", "T.pval.USE.O", "T.r.O")
OUTCOME_COLS = ("T.pval.USE.R", "Direction.R")

PRIMARY_K = 10
SENSITIVITY_K = (1, 5, 10, 20, 50)
POWER = 0.80
BASE_ALPHA = 0.05


@dataclass(frozen=True)
class Rule:
    name: str
    k: int
    kind: str  # "p" (search-priced threshold) or "mde" (search-priced MDE)

    @property
    def alpha(self) -> float:
        return BASE_ALPHA / self.k

    @property
    def z_threshold(self) -> float:
        z = stats.norm.ppf(1 - self.alpha / 2)
        return float(z + stats.norm.ppf(POWER)) if self.kind == "mde" else float(z)

    def describe(self) -> str:
        if self.kind == "mde":
            return f"accept iff |z| >= {self.z_threshold:.3f} (search-priced MDE, K={self.k}, {POWER:.0%} power)"
        return f"accept iff p <= {self.alpha:.4g} (0.05 / K, K={self.k})"


PRIMARY = Rule("primary", PRIMARY_K, "p")
SECONDARY = Rule("secondary", PRIMARY_K, "mde")


def z_from_p(p: pd.Series) -> pd.Series:
    clipped = p.clip(lower=1e-300, upper=1.0)
    return pd.Series(stats.norm.isf(clipped / 2), index=p.index)


def required_n(r: float, alpha: float, power: float = POWER) -> int | None:
    """Fisher-z sample size to detect correlation ``r`` at two-sided ``alpha``."""
    if r is None or not np.isfinite(r) or r == 0 or abs(r) >= 1:
        return None
    zr = math.atanh(abs(r))
    n = ((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / zr) ** 2 + 3
    return int(math.ceil(n))


def include(frame: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered inclusion: inputs and outcomes present, original p < 0.05."""
    needed = ["T.pval.USE.O", "T.N.O", "T.pval.USE.R", "Direction.R"]
    out = frame.copy()
    for col in ("T.pval.USE.O", "T.N.O", "T.pval.USE.R", "T.r.O"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Direction.R"] = out["Direction.R"].astype("string").str.strip().str.lower()
    out = out.dropna(subset=needed)
    out = out[out["Direction.R"].isin(["same", "opposite"])]
    return out[out["T.pval.USE.O"] < BASE_ALPHA]


def decide(inputs: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    """Accept/refuse from publication-time information only."""
    extra = set(inputs.columns) - set(INPUT_COLS)
    if extra:
        raise ValueError(f"decide() saw non-input columns: {sorted(extra)}")
    p = inputs["T.pval.USE.O"].astype(float)
    if rule.kind == "p":
        accept = p <= rule.alpha
    else:
        accept = z_from_p(p) >= rule.z_threshold
    prescription = [
        None if ok else required_n(r, rule.alpha)
        for ok, r in zip(accept, inputs["T.r.O"].astype(float))
    ]
    return pd.DataFrame({"accept": accept.astype(bool), "prescribed_n": prescription}, index=inputs.index)


def replicated(outcomes: pd.DataFrame) -> pd.Series:
    """Pre-registered outcome: replication p < 0.05 and same direction."""
    return (outcomes["T.pval.USE.R"].astype(float) < BASE_ALPHA) & (outcomes["Direction.R"] == "same")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(decisions: pd.DataFrame, outcome: pd.Series, rule: Rule) -> dict[str, Any]:
    joined = decisions.join(outcome.rename("replicated"))
    acc = joined[joined["accept"]]
    ref = joined[~joined["accept"]]
    a_k, a_n = int(acc["replicated"].sum()), len(acc)
    r_k, r_n = int(ref["replicated"].sum()), len(ref)
    table = [[a_k, a_n - a_k], [r_k, r_n - r_k]]
    fisher_p = float(stats.fisher_exact(table)[1]) if a_n and r_n else None
    prescribed = ref["prescribed_n"].dropna()
    return {
        "rule": rule.name,
        "k": rule.k,
        "description": rule.describe(),
        "n": a_n + r_n,
        "accepted": a_n,
        "refused": r_n,
        "accepted_replicated": a_k,
        "refused_replicated": r_k,
        "accept_rate": a_k / a_n if a_n else None,
        "accept_ci": wilson(a_k, a_n),
        "refuse_rate": r_k / r_n if r_n else None,
        "refuse_ci": wilson(r_k, r_n),
        "difference": (a_k / a_n - r_k / r_n) if a_n and r_n else None,
        "fisher_p": fisher_p,
        "prescribed_n_median": float(prescribed.median()) if len(prescribed) else None,
    }


def run_backtest(frame: pd.DataFrame) -> dict[str, Any]:
    kept = include(frame)
    inputs = kept[list(INPUT_COLS)]
    rules = [PRIMARY, SECONDARY] + [Rule(f"sensitivity_k{k}", k, "p") for k in SENSITIVITY_K]
    decisions = {rule.name: decide(inputs, rule) for rule in rules}  # all decisions first
    outcome = replicated(kept[list(OUTCOME_COLS)])  # outcomes only now
    results = {rule.name: summarise(decisions[rule.name], outcome, rule) for rule in rules}
    return {
        "schema": "autoresearch.backtest/v1",
        "n_rows": int(len(frame)),
        "n_included": int(len(kept)),
        "overall_replication_rate": float(outcome.mean()) if len(outcome) else None,
        "primary": results["primary"],
        "secondary": results["secondary"],
        "sensitivity": [results[f"sensitivity_k{k}"] for k in SENSITIVITY_K],
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _ci(ci: tuple[float | None, float | None]) -> str:
    lo, hi = ci
    return "n/a" if lo is None else f"{100 * lo:.0f}–{100 * hi:.0f}%"


def _row(s: dict[str, Any]) -> str:
    fp = "n/a" if s["fisher_p"] is None else f"{s['fisher_p']:.2g}"
    return (
        f"| {s['rule']} | {s['k']} | {s['accepted']} | {s['accepted_replicated']}/{s['accepted']} "
        f"({_pct(s['accept_rate'])}, {_ci(s['accept_ci'])}) | {s['refused']} | "
        f"{s['refused_replicated']}/{s['refused']} ({_pct(s['refuse_rate'])}, {_ci(s['refuse_ci'])}) | {fp} |"
    )


def to_markdown(result: dict[str, Any]) -> str:
    p = result["primary"]
    lines = [
        "# Replication back-test: results",
        "",
        "Pre-registration: [PREREGISTRATION.md](PREREGISTRATION.md), committed before outcomes were joined.",
        f"Data: RPP `RPPdataConverted.xlsx`, SHA-256 `{result.get('data_sha256', 'n/a')}`.",
        "",
        "## Headline (primary rule, K = 10)",
        "",
        f"Run blind on **{p['n']}** published findings, it accepted **{p['accepted']}** and refused "
        f"**{p['refused']}**; its accepts replicated at **{_pct(p['accept_rate'])}** "
        f"(95% CI {_ci(p['accept_ci'])}) and its refusals at **{_pct(p['refuse_rate'])}** "
        f"(95% CI {_ci(p['refuse_ci'])}).",
        "",
        f"Overall replication rate among included findings: {_pct(result['overall_replication_rate'])}.",
        "",
        "## All rules",
        "",
        "| Rule | K | Accepted | Accepts replicated | Refused | Refusals replicated | Fisher p |",
        "|---|---:|---:|---|---:|---|---:|",
        _row(result["primary"]),
        _row(result["secondary"]),
        *[_row(s) for s in result["sensitivity"]],
        "",
        f"Median prescribed N for refused findings (primary rule): "
        f"{'n/a' if p['prescribed_n_median'] is None else int(p['prescribed_n_median'])}.",
        "",
        "## Reading this",
        "",
        "- This is the gate's search-pricing logic applied to summary statistics, not the full engine.",
        "- The link between small original p-values and replication was already reported by OSC (2015). "
        "This checks that a rule fixed in advance reproduces it; it is not a new finding.",
        "- K = 10 is a convention; the sensitivity rows show how much the answer depends on it.",
        "",
    ]
    return "\n".join(lines)


def load_rpp(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)
    missing = [c for c in (*INPUT_COLS, *OUTCOME_COLS) if c not in frame.columns]
    if missing:
        raise ValueError(f"RPP file is missing columns: {missing}")
    return frame


def write_results(data_path: str | Path, out_json: str | Path, out_md: str | Path) -> dict[str, Any]:
    result = run_backtest(load_rpp(data_path))
    result["data_sha256"] = sha256_file(Path(data_path))
    atomic_write_json(Path(out_json), result)
    Path(out_md).write_text(to_markdown(result), encoding="utf-8", newline="\n")
    return result
