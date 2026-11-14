from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from . import metrics
from .utils import canonical_json, sha256_obj, sha256_text


class ContractError(Exception):
    """A named, actionable validation failure. Never a bare stack trace."""

    def __init__(self, reason_code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"reason_code": self.reason_code, "message": self.message, "details": self.details}


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)


def _normalize_direction(value: Any) -> str:
    if value is None:
        return "higher_better"
    v = str(value).strip().lower()
    if v in ("higher_better", "higher", "+", "ascending", "positive", "1"):
        return "higher_better"
    if v in ("lower_better", "lower", "-", "descending", "negative", "-1"):
        return "lower_better"
    raise ContractError("invalid_feature_direction", f"Unknown feature direction: {value!r}")


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class FeatureSpec:
    name: str
    direction: str = "higher_better"
    transform: str = "rank"
    description: str = ""
    required: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "direction": self.direction,
            "transform": self.transform,
            "description": self.description,
            "required": self.required,
        }


@dataclass
class BaselineSpec:
    name: str
    column: str | None = None
    weights: dict[str, float] = field(default_factory=dict)
    transforms: dict[str, str] = field(default_factory=dict)
    higher_is_better: bool = True
    description: str = ""


@dataclass
class DecisionSpec:
    action: str
    smallest_effect_that_matters: float
    alpha: float = 0.05
    target_power: float = 0.8
    max_runs: int = 1
    # Spec §12.4: the operating point belongs to the decision, and who set it is recorded.
    alpha_chosen_by: str = "default: 0.05 is a historical accident, not a decision"
    certificate_valid_days: int = 180
    # When the declared effect is below the minimum detectable effect: "refuse"
    # (the default, spec §3) or "proceed". Proceeding never weakens the gate's
    # error control; it spends the single-use sealed set on a question that can
    # only be answered if the true effect is larger than declared.
    if_underpowered: str = "refuse"
    if_underpowered_chosen_by: str = "default: refuse underpowered questions"


@dataclass
class PowerSpec:
    k_candidates: int = 100
    confirmation_n: int | None = None
    confirmation_groups: int | None = None
    alpha: float | None = None
    target_power: float | None = None


@dataclass
class AdversarySpec:
    permutations: int = 99
    pseudo_group_splits: int = 20
    prespecified_failure: dict[str, Any] | None = None
    baseline_advocate_candidates: int = 24


@dataclass
class ValidationReport:
    ok: bool
    checks: list[dict[str, Any]]
    warnings: list[str]
    errors: list[dict[str, Any]]
    stats: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "warnings": self.warnings,
            "errors": self.errors,
            "stats": self.stats,
        }


class Contract:
    """A versioned problem definition plus the validator that enforces it."""

    def __init__(self, raw: dict[str, Any], path: str | Path | None = None):
        self.raw = raw
        self.path = Path(path) if path else None
        self.hash = sha256_obj(raw)
        self.data: pd.DataFrame | None = None
        self.confirmation_meta: dict[str, Any] | None = None
        self.validation: ValidationReport | None = None

        self.contract_version = int(raw.get("contract_version", 0))
        self.unit = str(raw.get("unit", "unit"))
        self.group = raw.get("group")
        self.outcome = dict(raw.get("outcome", {}))
        self.candidates_raw = dict(raw.get("candidates", {}))
        self.baseline_raw = dict(raw.get("baseline", {}))
        self.metric_raw = dict(raw.get("metric", {}))
        self.contexts = list(raw.get("contexts", []) or [])
        # Milestone 8 / spec §3 direction aggregation: feature directions double
        # as pre-committed sign predictions, tested once on the sealed set.
        self.commit_signs = bool((raw.get("commitments") or {}).get("signs", False))
        self.constraints = dict(raw.get("constraints", {}) or {})
        self.decision_raw = dict(raw.get("decision", {}))
        self.power_raw = dict(raw.get("power", {}) or {})
        self.adversary_raw = dict(raw.get("adversary", {}) or {})
        self.leakage_raw = dict(raw.get("leakage", {}) or {})
        self.provenance_raw = dict(raw.get("provenance", {}) or {})

        self.features: list[FeatureSpec] = []
        self.baseline = BaselineSpec(name="baseline")
        self.decision = DecisionSpec(action="", smallest_effect_that_matters=float("nan"))
        self.power = PowerSpec()
        self.adversary = AdversarySpec()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def parse(self) -> "Contract":
        self._parse_outcome()
        self._parse_features()
        self._parse_baseline()
        self._parse_metric()
        self._parse_decision()
        self._parse_power()
        self._parse_adversary()
        return self

    def _parse_outcome(self) -> None:
        if not self.outcome:
            raise ContractError("missing_outcome", "contract.outcome is required")
        if not self.outcome.get("column"):
            raise ContractError("missing_outcome_column", "contract.outcome.column is required")
        otype = str(self.outcome.get("type", "binary")).lower()
        if otype not in ("binary", "ordinal", "continuous", "censored", "time_to_event"):
            raise ContractError("unsupported_outcome_type", f"Unknown outcome type: {otype}")
        if otype not in ("binary", "ordinal"):
            raise ContractError(
                "unsupported_outcome_type",
                f"Outcome type {otype!r} is declared but not implemented in v0.1. "
                "Use a pre-specified binary endpoint, or register a plugin.",
                {"supported": ["binary", "ordinal"]},
            )
        self.outcome["type"] = otype
        if not self.outcome.get("measured_by"):
            self.outcome["measured_by"] = "not stated"

    def _parse_features(self) -> None:
        features = self.candidates_raw.get("features")
        if not features:
            raise ContractError("missing_features", "contract.candidates.features is required")
        out: list[FeatureSpec] = []
        seen: set[str] = set()
        for item in features:
            if isinstance(item, str):
                item = {"name": item}
            if not isinstance(item, dict) or not item.get("name"):
                raise ContractError("invalid_feature", f"Invalid feature entry: {item!r}")
            name = str(item["name"])
            if name in seen:
                raise ContractError("duplicate_feature", f"Feature listed twice: {name}")
            seen.add(name)
            transform = str(item.get("transform", "rank")).lower()
            if transform not in ("rank", "z", "log_rank", "binary_median", "raw"):
                raise ContractError("invalid_transform", f"Unknown transform {transform!r} for feature {name}")
            out.append(
                FeatureSpec(
                    name=name,
                    direction=_normalize_direction(item.get("direction")),
                    transform=transform,
                    description=str(item.get("description", "")),
                    required=bool(item.get("required", True)),
                )
            )
        self.features = out

    def _parse_baseline(self) -> None:
        if not self.baseline_raw:
            raise ContractError("missing_baseline", "contract.baseline is required; 'chance' is not a deployed baseline")
        name = self.baseline_raw.get("name")
        if not name:
            raise ContractError("missing_baseline_name", "contract.baseline.name is required")
        column = self.baseline_raw.get("column")
        weights = self.baseline_raw.get("weights") or {}
        if not column and not weights:
            raise ContractError(
                "missing_baseline_spec",
                "baseline must specify either a score column or feature weights",
            )
        if weights and not isinstance(weights, dict):
            raise ContractError("invalid_baseline_weights", "baseline.weights must be a mapping feature->weight")
        allowed = {f.name for f in self.features}
        for f in weights:
            if f not in allowed:
                raise ContractError("unknown_baseline_feature", f"Baseline uses undeclared feature {f!r}")
        self.baseline = BaselineSpec(
            name=str(name),
            column=str(column) if column else None,
            weights={str(k): float(v) for k, v in weights.items()},
            transforms={str(k): str(v) for k, v in (self.baseline_raw.get("transforms") or {}).items()},
            higher_is_better=bool(self.baseline_raw.get("higher_is_better", True)),
            description=str(self.baseline_raw.get("description", "")),
        )

    def _parse_metric(self) -> None:
        primary = str(self.metric_raw.get("primary", "paired_delta_auc"))
        if primary not in ("paired_delta_auc", "delta_auc", "auc"):
            raise ContractError(
                "unsupported_metric",
                f"Metric {primary!r} is not implemented. Use paired_delta_auc.",
                {"supported": ["paired_delta_auc", "delta_auc", "auc"]},
            )
        self.metric_raw["primary"] = primary
        self.metric_raw["secondary"] = list(self.metric_raw.get("secondary", []) or [])
        self.metric_raw["top_k"] = int(self.metric_raw.get("top_k", 3))

    def _parse_decision(self) -> None:
        if not self.decision_raw:
            raise ContractError("missing_decision", "contract.decision is required")
        effect = _as_float(self.decision_raw.get("smallest_effect_that_matters"))
        if effect is None or not math.isfinite(effect) or effect <= 0:
            raise ContractError(
                "missing_smallest_effect",
                "decision.smallest_effect_that_matters is required, has no default, and must be > 0",
            )
        action = self.decision_raw.get("action")
        if not action:
            raise ContractError("missing_decision_action", "decision.action is required")
        alpha = _as_float(self.decision_raw.get("alpha"), 0.05)
        target_power = _as_float(self.decision_raw.get("target_power"), 0.8)
        max_runs = int(self.decision_raw.get("max_runs", 1))
        if not (0 < alpha < 1):
            raise ContractError("invalid_alpha", "decision.alpha must be in (0, 1)")
        if not (0 < target_power < 1):
            raise ContractError("invalid_target_power", "decision.target_power must be in (0, 1)")
        if max_runs < 1:
            raise ContractError("invalid_max_runs", "decision.max_runs must be >= 1")
        valid_days = int(self.decision_raw.get("certificate_valid_days", 180))
        if valid_days < 1:
            raise ContractError("invalid_certificate_valid_days", "decision.certificate_valid_days must be >= 1")
        chosen_by = self.decision_raw.get("alpha_chosen_by")
        policy = str(self.decision_raw.get("if_underpowered", "refuse")).strip().lower()
        if policy not in ("refuse", "proceed"):
            raise ContractError("invalid_if_underpowered", "decision.if_underpowered must be 'refuse' or 'proceed'")
        policy_by = self.decision_raw.get("if_underpowered_chosen_by")
        if policy == "proceed" and not policy_by:
            raise ContractError(
                "if_underpowered_unattributed",
                "decision.if_underpowered: proceed must name who chose it (decision.if_underpowered_chosen_by)",
            )
        self.decision = DecisionSpec(
            action=str(action),
            smallest_effect_that_matters=float(effect),
            alpha=float(alpha),
            target_power=float(target_power),
            max_runs=max_runs,
            alpha_chosen_by=str(chosen_by) if chosen_by else DecisionSpec.alpha_chosen_by,
            certificate_valid_days=valid_days,
            if_underpowered=policy,
            if_underpowered_chosen_by=str(policy_by) if policy_by else DecisionSpec.if_underpowered_chosen_by,
        )

    def _parse_power(self) -> None:
        self.power = PowerSpec(
            k_candidates=int(self.power_raw.get("k_candidates", 100)),
            confirmation_n=int(self.power_raw["confirmation_n"]) if self.power_raw.get("confirmation_n") else None,
            confirmation_groups=int(self.power_raw["confirmation_groups"]) if self.power_raw.get("confirmation_groups") else None,
            alpha=_as_float(self.power_raw.get("alpha")),
            target_power=_as_float(self.power_raw.get("target_power")),
        )

    def _parse_adversary(self) -> None:
        self.adversary = AdversarySpec(
            permutations=int(self.adversary_raw.get("permutations", 99)),
            pseudo_group_splits=int(self.adversary_raw.get("pseudo_group_splits", 20)),
            prespecified_failure=self.adversary_raw.get("prespecified_failure"),
            baseline_advocate_candidates=int(self.adversary_raw.get("baseline_advocate_candidates", 24)),
        )

    # ------------------------------------------------------------------
    # Static and data validation
    # ------------------------------------------------------------------
    def validate_static(self) -> ValidationReport:
        self.parse()
        checks = [
            {"name": "contract_version", "ok": True, "detail": f"v{self.contract_version}"},
            {"name": "decision_effect_declared", "ok": True, "detail": self.decision.smallest_effect_that_matters},
            {"name": "baseline_is_deployed", "ok": True, "detail": self.baseline.name},
            {"name": "outcome_type_supported", "ok": True, "detail": self.outcome["type"]},
        ]
        report = ValidationReport(ok=True, checks=checks, warnings=[], errors=[], stats={"contract_hash": self.hash})
        self.validation = report
        return report

    def attach_data(self, df: pd.DataFrame, role: str = "discovery") -> ValidationReport:
        if self.validation is None:
            self.validate_static()
        report = self.validate_data(df, role=role)
        self.validation = report
        if report.ok:
            self.data = df.copy()
        return report

    def validate_data(self, df: pd.DataFrame, role: str = "discovery") -> ValidationReport:
        errors: list[dict[str, Any]] = []
        warnings: list[str] = []
        checks: list[dict[str, Any]] = []

        def err(code: str, message: str, **details: Any) -> None:
            errors.append({"reason_code": code, "message": message, "details": details})

        def warn(message: str) -> None:
            warnings.append(message)

        def check(name: str, ok: bool, detail: Any = None) -> None:
            checks.append({"name": name, "ok": bool(ok), "detail": detail})

        if df is None or len(df) == 0:
            err("empty_data", "Dataset has no rows")
            return ValidationReport(False, checks, warnings, errors, {})

        cols = list(df.columns)
        outcome_col = self.outcome["column"]
        group_col = self.group
        unit_col = self.unit

        # Unit uniqueness.
        if unit_col in df.columns:
            dup = int(df[unit_col].duplicated().sum())
            check("unit_unique", dup == 0, {"duplicates": dup, "unit": unit_col})
            if dup:
                err("duplicate_units", f"{dup} duplicate values in unit column {unit_col!r}", column=unit_col)
        else:
            warn(f"unit column {unit_col!r} not found; row index will be used as the unit")
            df = df.copy()
            df[unit_col] = [f"row-{i}" for i in range(len(df))]

        # Group structure.
        if group_col:
            if group_col not in df.columns:
                err("missing_group_column", f"Group column {group_col!r} not found", column=group_col)
                groups = np.array(["__all__"] * len(df))
            else:
                if df[group_col].isna().any():
                    err("missing_group_values", f"Group column {group_col!r} has missing values")
                groups = df[group_col].astype(str).to_numpy()
                counts = pd.Series(groups).value_counts()
                n_groups = int(len(counts))
                check("groups_present", n_groups >= 2, {"n_groups": n_groups})
                if n_groups < 2:
                    err("too_few_groups", f"Only {n_groups} group(s); confirmation would not be independent")
                if n_groups == len(df):
                    warn("Every row is its own group; group-aware inference degenerates to unit-level inference")
                check("group_sizes", True, {"min": int(counts.min()), "median": float(counts.median()), "max": int(counts.max())})
        else:
            groups = np.array(["__all__"] * len(df))

        # Outcome.
        if outcome_col not in df.columns:
            err("missing_outcome_column", f"Outcome column {outcome_col!r} not found", column=outcome_col)
            y = np.array([], dtype=int)
            prevalence = float("nan")
        else:
            if df[outcome_col].isna().any():
                err("missing_outcomes", f"Outcome column {outcome_col!r} has missing values")
            try:
                y = pd.to_numeric(df[outcome_col], errors="raise").to_numpy()
            except Exception:
                err("non_numeric_outcome", f"Outcome column {outcome_col!r} is not numeric")
                y = np.array([], dtype=int)
            if len(y):
                uniq = sorted(set(float(v) for v in y if not pd.isna(v)))
                if self.outcome["type"] in ("binary", "ordinal"):
                    if not set(uniq).issubset({0.0, 1.0}):
                        err(
                            "outcome_not_binary",
                            f"Binary contract requires 0/1 outcomes; found {uniq[:8]}",
                            column=outcome_col,
                        )
                    y = y.astype(int)
                    prevalence = float(y.mean()) if len(y) else float("nan")
                    check("both_classes", 0 < y.sum() < len(y), {"positive": int(y.sum()), "negative": int(len(y) - y.sum())})
                    if y.sum() == 0 or y.sum() == len(y):
                        err("single_class_outcome", "Outcome has only one class; AUC is undefined")
                    if not (0.02 <= prevalence <= 0.98):
                        warn(f"Outcome prevalence is extreme ({prevalence:.3f}); power will be fragile")
                else:
                    warn("Non-binary outcome validation is limited in v0.1")

        # Features.
        feature_stats: dict[str, Any] = {}
        forbidden = set(self.leakage_raw.get("forbidden_features", []) or [])
        post_outcome = set(self.leakage_raw.get("post_outcome_columns", []) or [])
        for f in self.features:
            if f.name not in df.columns:
                if f.required:
                    err("missing_feature", f"Feature {f.name!r} not found", feature=f.name)
                continue
            if f.name == outcome_col:
                err("outcome_as_feature", f"Feature {f.name!r} is the outcome column", feature=f.name)
            if f.name in forbidden:
                err("forbidden_feature", f"Feature {f.name!r} is declared forbidden", feature=f.name)
            if f.name in post_outcome:
                err("post_outcome_feature", f"Feature {f.name!r} is only available after the outcome", feature=f.name)
            if not pd.api.types.is_numeric_dtype(df[f.name]):
                err("non_numeric_feature", f"Feature {f.name!r} is not numeric", feature=f.name)
                continue
            miss = float(df[f.name].isna().mean())
            nunique = int(df[f.name].nunique(dropna=True))
            feature_stats[f.name] = {"missing_fraction": miss, "n_unique": nunique}
            if miss > 0.5:
                err("feature_mostly_missing", f"Feature {f.name!r} is {miss:.0%} missing", feature=f.name)
            elif miss > 0.2:
                warn(f"Feature {f.name!r} is {miss:.0%} missing; imputation will be used")
            if nunique < 2:
                warn(f"Feature {f.name!r} is constant; it cannot contribute")
            # Near-perfect outcome leakage check (binary only).
            if len(y) and self.outcome["type"] in ("binary", "ordinal") and nunique >= 2:
                valid = df[[f.name]].join(pd.Series(y, index=df.index, name="__y__")).dropna()
                if len(valid) >= 20:
                    try:
                        r = float(stats.pointbiserialr(valid["__y__"], valid[f.name]).statistic)
                        feature_stats[f.name]["abs_pointbiserial"] = abs(r)
                        if abs(r) >= 0.995:
                            err(
                                "leakage_suspect_feature",
                                f"Feature {f.name!r} is almost perfectly correlated with the outcome (|r|={abs(r):.3f}); this is a label leak",
                                feature=f.name,
                                correlation=r,
                            )
                        elif abs(r) >= 0.95:
                            warn(f"Feature {f.name!r} is highly correlated with the outcome (|r|={abs(r):.3f}); review for leakage")
                    except Exception:
                        pass

        # Contexts.
        for c in self.contexts:
            if c not in df.columns:
                err("missing_context", f"Context column {c!r} not found", context=c)

        # Baseline.
        base_scores = None
        if self.baseline.column:
            if self.baseline.column not in df.columns:
                err("missing_baseline_score", f"Baseline score column {self.baseline.column!r} not found")
            else:
                base_scores = pd.to_numeric(df[self.baseline.column], errors="coerce").to_numpy(dtype=float)
                if np.isnan(base_scores).any():
                    err("baseline_missing_values", f"Baseline score column {self.baseline.column!r} has missing/non-numeric values")
                if not self.baseline.higher_is_better:
                    base_scores = -base_scores
        elif self.baseline.weights:
            missing = [f for f in self.baseline.weights if f not in df.columns]
            if missing:
                err("baseline_feature_missing", f"Baseline features missing: {missing}")
            else:
                # A simple direction-aware z-score composition is enough for validation;
                # the evaluator uses the canonical scorer.
                from .scoring import FeatureMatrix, Candidate

                try:
                    fm = FeatureMatrix(df, self, fit=False)
                    cand = Candidate(
                        id="baseline",
                        name=self.baseline.name,
                        mechanism="declared baseline",
                        family="baseline",
                        params={
                            "weights": self.baseline.weights,
                            "transforms": self.baseline.transforms,
                        },
                    )
                    base_scores = fm.score_candidate(cand)
                except Exception as exc:  # pragma: no cover - defensive
                    err("baseline_unscoreable", f"Could not score baseline: {exc}")

        # Effect size and AUC computability.
        stats_out: dict[str, Any] = {
            "role": role,
            "n_rows": int(len(df)),
            "n_groups": int(len(set(groups))),
            "n_positive": int(y.sum()) if len(y) else 0,
            "n_negative": int(len(y) - y.sum()) if len(y) else 0,
            "prevalence": prevalence,
            "features": feature_stats,
            "columns": cols,
        }
        if len(y) and base_scores is not None and not np.isnan(base_scores).all():
            b_auc = metrics.auc(y, base_scores)
            stats_out["baseline_auc"] = b_auc
            check("baseline_auc_computable", not np.isnan(b_auc), {"baseline_auc": b_auc})
            if not np.isnan(b_auc) and b_auc < 0.5:
                warn(f"Baseline AUC is {b_auc:.3f} (<0.5); check score direction")
            if self.metric_raw["primary"] in ("paired_delta_auc", "delta_auc"):
                check("metric_computable", not np.isnan(b_auc), {"metric": self.metric_raw["primary"]})
        else:
            stats_out["baseline_auc"] = None

        # Group-aware evaluability.
        if len(y):
            eval_groups = metrics.evaluable_groups(y, groups)
            stats_out["n_groups_evaluable"] = int(len(eval_groups))
            check("groups_with_both_classes", len(eval_groups) >= 2, {"n_groups_evaluable": len(eval_groups)})
            if len(eval_groups) < 4:
                warn(f"Only {len(eval_groups)} groups contain both outcome classes; paired inference will be fragile")

        # Confirmation compatibility.
        if role == "confirmation" and self.confirmation_meta:
            stats_out["confirmation_meta"] = self.confirmation_meta

        ok = len(errors) == 0
        report = ValidationReport(ok=ok, checks=checks, warnings=warnings, errors=errors, stats=stats_out)
        if not ok:
            first = errors[0]
            self.validation = report
            raise ContractError(first["reason_code"], first["message"], report.to_dict())
        return report

    # ------------------------------------------------------------------
    # Confirmation metadata (no labels)
    # ------------------------------------------------------------------
    def attach_confirmation_meta(self, meta: dict[str, Any]) -> None:
        self.confirmation_meta = dict(meta)
        required = ["n_rows", "n_groups", "columns"]
        for k in required:
            if k not in meta:
                raise ContractError("invalid_confirmation_meta", f"Sealed dataset meta missing {k!r}")
        cols = set(meta["columns"])
        needed = {self.outcome["column"], self.unit}
        if self.group:
            needed.add(self.group)
        if self.baseline.column:
            needed.add(self.baseline.column)
        for f in self.features:
            if f.required:
                needed.add(f.name)
        missing = sorted(needed - cols)
        if missing:
            raise ContractError(
                "confirmation_schema_mismatch",
                f"Sealed dataset is missing columns required by the contract: {missing}",
                {"missing": missing, "available": sorted(cols)},
            )
        if meta.get("n_groups", 0) < 2:
            raise ContractError("confirmation_too_few_groups", "Sealed dataset has fewer than 2 groups")
        if meta.get("prevalence") is not None:
            p = float(meta["prevalence"])
            if not (0 < p < 1):
                raise ContractError("confirmation_single_class", "Sealed dataset does not contain both outcome classes")

    def check_group_overlap(self, discovery_groups: list[str], confirmation_group_hash: str | None) -> None:
        if not confirmation_group_hash:
            return
        from .utils import sha256_obj

        if sha256_obj(sorted(map(str, discovery_groups))) == confirmation_group_hash:
            raise ContractError(
                "split_overlap",
                "Discovery and confirmation group identifiers are identical; this is not an independent confirmation set",
            )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def outcome_column(self) -> str:
        return str(self.outcome["column"])

    @property
    def group_column(self) -> str | None:
        return str(self.group) if self.group else None

    @property
    def top_k(self) -> int:
        return int(self.metric_raw.get("top_k", 3))

    def baseline_candidate(self):
        from .scoring import Candidate

        if self.baseline.column:
            params = {"column": self.baseline.column, "higher_is_better": self.baseline.higher_is_better}
        else:
            params = {
                "weights": self.baseline.weights,
                "transforms": self.baseline.transforms,
            }
        return Candidate(
            id="baseline",
            name=self.baseline.name,
            mechanism=f"Deployed baseline: {self.baseline.description or self.baseline.name}",
            family="baseline",
            params=params,
            source="baseline",
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)

    def canonical_yaml(self) -> str:
        return yaml.safe_dump(self.raw, sort_keys=True, allow_unicode=True)


def load_contract(
    path: str | Path,
    data_path: str | Path | None = None,
    confirmation_meta: dict[str, Any] | None = None,
) -> Contract:
    p = Path(path)
    if not p.exists():
        raise ContractError("contract_not_found", f"Contract file not found: {p}")
    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except Exception as exc:
        raise ContractError("invalid_yaml", f"Could not parse contract YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractError("invalid_contract", "Contract must be a YAML mapping")
    contract = Contract(raw, path=p)
    contract.validate_static()
    if confirmation_meta:
        contract.attach_confirmation_meta(confirmation_meta)
    if data_path:
        df = read_table(data_path)
        contract.attach_data(df, role="discovery")
    return contract
