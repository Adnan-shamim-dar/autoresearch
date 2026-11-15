"""Bring your own data: any CSV with a binary outcome and a deployed baseline column.

    autoresearch --root myproject init --domain custom --data trial.csv \\
        --outcome responded --baseline current_score [--group site] [--effect 0.05]
    autoresearch --root myproject demo --skip-economics

This writes a contract, splits the rows (by group when one is given), seals the
confirmation split in the vault and registers the contract. The contract is a
plain YAML file under ``examples/custom/`` and can be edited before the run.

Feature directions are inferred from the discovery split only (the sign of
each feature's rank correlation with the outcome). The sealed rows play no
part in any choice made here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .contract import ContractError, load_contract
from .registry import Registry
from .utils import ensure_dir, rng_for
from .vault import Vault

MAX_FEATURES = 40


def _binary(series: pd.Series, positive: str | None) -> pd.Series:
    if positive is not None:
        return (series.astype(str) == str(positive)).astype(int)
    values = set(pd.unique(series.dropna()))
    if values <= {0, 1} or values <= {True, False}:
        return series.astype(int)
    raise ContractError(
        "outcome_not_binary",
        f"Outcome column has values {sorted(map(str, values))[:6]}; pass --positive VALUE to say which one counts as 1.",
    )


def _split(df: pd.DataFrame, outcome: str, group: str | None, fraction: float, seed: int) -> np.ndarray:
    rng = rng_for("custom-split", seed)
    conf = np.zeros(len(df), dtype=bool)
    if group:
        groups = np.array(sorted(df[group].astype(str).unique()))
        chosen = set(rng.permutation(groups)[: max(2, int(round(len(groups) * fraction)))])
        return df[group].astype(str).isin(chosen).to_numpy()
    y = df[outcome].to_numpy()
    for label in (0, 1):
        idx = np.where(y == label)[0]
        conf[rng.permutation(idx)[: int(round(len(idx) * fraction))]] = True
    return conf


def make_contract_dict(
    disc: pd.DataFrame, *, unit: str, group: str | None, outcome: str, baseline: str, features: list[str],
    baseline_higher: bool, effect: float, confirmation_n: int, confirmation_groups: int,
    if_underpowered: str, chosen_by: str, source: str,
) -> dict[str, Any]:
    y = disc[outcome].to_numpy()
    directions = {
        f: ("higher_better" if (stats.spearmanr(disc[f], y, nan_policy="omit").statistic or 0) >= 0 else "lower_better")
        for f in features
    }
    decision: dict[str, Any] = {
        "action": f"replace '{baseline}' with the certified rule",
        "smallest_effect_that_matters": float(effect),
        "alpha": 0.05,
        "target_power": 0.8,
        "max_runs": 1,
        "if_underpowered": if_underpowered,
    }
    if if_underpowered == "proceed":
        decision["if_underpowered_chosen_by"] = chosen_by
    return {
        "contract_version": 1,
        "unit": unit,
        "group": group,
        "outcome": {"column": outcome, "type": "binary", "measured_by": f"column '{outcome}' of {source}"},
        "candidates": {
            "grammar": "weighted_rank_fusion",
            "max_terms": None,
            "features": [{"name": f, "direction": d, "transform": "rank"} for f, d in directions.items()],
        },
        "baseline": {"name": "deployed_baseline", "column": baseline, "higher_is_better": bool(baseline_higher),
                     "description": f"the criterion in use today: '{baseline}'"},
        "metric": {"primary": "paired_delta_auc",
                   "secondary": ["auc", "top3_enrichment", "frac_improved", "mean_pct_rank", "rank_coef"], "top_k": 3},
        "contexts": [],
        "constraints": {"interpretable": True, "max_terms": None},
        "decision": decision,
        "commitments": {},
        "power": {"k_candidates": 60, "confirmation_n": int(confirmation_n), "confirmation_groups": int(confirmation_groups)},
        "adversary": {"permutations": 79, "pseudo_group_splits": 20, "baseline_advocate_candidates": 24},
        "provenance": {"discovery_license": "user-supplied", "confirmation_license": "user-supplied, sealed in the vault",
                       "notes": f"custom domain built from {source}; feature directions inferred on the discovery split only"},
    }


def write_files(
    root: str | Path, data: str | Path, outcome: str, baseline: str, *, group: str | None = None,
    unit: str | None = None, features: list[str] | None = None, positive: str | None = None,
    baseline_higher: bool = True, effect: float = 0.05, confirm_fraction: float = 0.4, seed: int = 20261115,
    if_underpowered: str = "refuse", chosen_by: str = "",
) -> dict[str, Any]:
    root, data = Path(root), Path(data)
    df = pd.read_csv(data, sep=None, engine="python")
    for col in (outcome, baseline, *([group] if group else []), *([unit] if unit else [])):
        if col not in df.columns:
            raise ContractError("missing_column", f"Column '{col}' is not in {data.name}. Columns: {list(df.columns)[:20]}")
    df = df.dropna(subset=[outcome]).reset_index(drop=True)
    df[outcome] = _binary(df[outcome], positive)
    if unit is None:
        unit = "unit_id"
        if unit not in df.columns:
            df.insert(0, unit, [f"u{i:06d}" for i in range(len(df))])
    reserved = {outcome, unit, group}
    if features is None:
        features = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    features = [f for f in features if f in df.columns and f not in reserved]
    if baseline not in features:
        features = [baseline] + features
    if len(features) < 2:
        raise ContractError("too_few_features", "Need the baseline plus at least one other numeric feature column.")
    if len(features) > MAX_FEATURES:
        raise ContractError("too_many_features", f"{len(features)} numeric features; pass --features to choose at most {MAX_FEATURES}.")
    keep = [unit] + ([group] if group else []) + features + [outcome]
    df = df[keep].dropna().reset_index(drop=True)

    is_conf = _split(df, outcome, group, confirm_fraction, seed)
    disc, conf = df[~is_conf], df[is_conf]
    if min(disc[outcome].sum(), conf[outcome].sum(), (1 - disc[outcome]).sum(), (1 - conf[outcome]).sum()) < 5:
        raise ContractError("too_few_events", "Each split needs at least 5 positive and 5 negative outcomes.")

    data_dir, examples = ensure_dir(root / "data"), ensure_dir(root / "examples" / "custom")
    disc_path, conf_path = data_dir / "custom_discovery.csv", data_dir / "custom_confirmation.csv"
    disc.to_csv(disc_path, index=False)
    conf.to_csv(conf_path, index=False)
    raw = make_contract_dict(
        disc, unit=unit, group=group, outcome=outcome, baseline=baseline, features=features,
        baseline_higher=baseline_higher, effect=effect, confirmation_n=len(conf),
        confirmation_groups=int(conf[group].nunique()) if group else len(conf),
        if_underpowered=if_underpowered, chosen_by=chosen_by, source=data.name,
    )
    contract_path = examples / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")

    vault = Vault(ensure_dir(root / "vault"))
    extra = {"group_column": group} if group else {}
    disc_meta = vault.ingest(disc_path, role="discovery", name="custom discovery split", source=data.name,
                             license="user-supplied", outcome_column=outcome, **extra)
    key_path = vault.make_key("custom-confirmation")
    conf_meta = vault.ingest(conf_path, role="sealed", name="custom sealed confirmation split", source=data.name,
                             license="user-supplied", sealed=True, outcome_column=outcome, budget=40, key_path=key_path, **extra)
    conf_path.unlink()  # the plaintext confirmation rows must not stay readable

    contract = load_contract(contract_path, disc_path)
    Registry(ensure_dir(root / "state")).register_contract(contract, registered_by="custom-bootstrap")
    return {
        "domain": "custom", "root": str(root), "contract_path": str(contract_path), "discovery_path": str(disc_path),
        "discovery_meta": disc_meta, "confirmation_meta": conf_meta, "key_path": str(key_path),
        "contract_hash": contract.hash, "n_discovery": int(len(disc)), "n_confirmation": int(len(conf)),
        "n_features": len(features),
        "limitations": [
            f"Custom domain built from {data.name}. Discovery and confirmation are a random split of one file, "
            "not an independent later dataset.",
            "Feature directions were inferred from the discovery split, not committed from mechanism.",
            "The gate's operating-characteristics table was not calibrated on this data.",
        ],
        "brief": f"Does a readable rule over {len(features)} features rank '{outcome}' better than '{baseline}' by at least {effect} AUC?",
    }
