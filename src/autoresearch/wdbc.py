"""Second domain (milestone 8): real data, no groups, signs pre-committed.

Wisconsin Diagnostic Breast Cancer (Wolberg, Street and Mangasarian, 1995):
569 fine-needle-aspirate images with 30 nuclear measurements and a biopsy
diagnosis. Bundled with scikit-learn, so it needs no download.

The question: does a readable multi-feature rule rank malignancy better than
the deployed triage criterion, tumour size (mean radius), by at least 0.02
AUC? The structure deliberately differs from the immunotherapy demo: real
measurements instead of a generator, no group column (independent units), a
different deployed baseline, and feature directions pre-committed as sign
predictions before any label is seen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .contract import load_contract
from .registry import Registry
from .utils import ensure_dir, rng_for
from .vault import Vault

# Directions are committed from mechanism before looking at outcomes: larger,
# more irregular, more textured nuclei indicate malignancy. Fractal dimension
# is a genuine risky prediction (irregular boundary -> higher), recorded as is.
SIGN_PREDICTIONS = {
    "mean_radius": "higher_better",
    "mean_texture": "higher_better",
    "mean_perimeter": "higher_better",
    "mean_area": "higher_better",
    "mean_smoothness": "higher_better",
    "mean_compactness": "higher_better",
    "mean_concavity": "higher_better",
    "mean_concave_points": "higher_better",
    "mean_symmetry": "higher_better",
    "mean_fractal_dimension": "higher_better",
}

LIMITATIONS = [
    "Real data: Wisconsin Diagnostic Breast Cancer (UCI, 1995), 569 tumours from one institution. "
    "It is a public benchmark, so any LLM proposer could have seen it; this run uses operator families only.",
    "Discovery and confirmation are a stratified random split of one cohort, not an independent later dataset.",
    "The gate's operating-characteristics table was calibrated on the immunotherapy synthetic substrate, not on this data.",
]


def make_contract_dict(confirmation_n: int, effect: float = 0.02) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "unit": "tumour_id",
        "group": None,
        "outcome": {
            "column": "malignant",
            "type": "binary",
            "measured_by": "histopathology of the biopsied mass (malignant vs benign)",
        },
        "candidates": {
            "grammar": "weighted_rank_fusion",
            "max_terms": None,
            "features": [{"name": n, "direction": d, "transform": "rank"} for n, d in SIGN_PREDICTIONS.items()],
        },
        "baseline": {
            "name": "tumour_size_triage",
            "column": "mean_radius",
            "higher_is_better": True,
            "description": "Size alone: the classical triage criterion for a suspicious mass",
        },
        "metric": {
            "primary": "paired_delta_auc",
            "secondary": ["auc", "top3_enrichment", "frac_improved", "mean_pct_rank", "rank_coef"],
            "top_k": 3,
        },
        "contexts": [],
        "constraints": {"interpretable": True, "max_terms": None},
        "decision": {
            "action": "replace size-only triage with the multi-feature rule in the reading protocol",
            "smallest_effect_that_matters": effect,
            "alpha": 0.05,
            "target_power": 0.8,
            "max_runs": 1,
            "alpha_chosen_by": "contract author: conventional 0.05 kept; a wrong switch is cheap to reverse",
        },
        "commitments": {"signs": True},
        "power": {"k_candidates": 60, "confirmation_n": confirmation_n, "confirmation_groups": confirmation_n},
        "adversary": {"permutations": 79, "pseudo_group_splits": 20, "baseline_advocate_candidates": 24},
        "provenance": {
            "discovery_license": "CC BY 4.0 (UCI Machine Learning Repository), via scikit-learn",
            "confirmation_license": "CC BY 4.0 (UCI Machine Learning Repository), sealed in the vault",
            "data_public_since": "1995-11-01",
            "notes": "Wolberg, Street, Mangasarian (1995). Real measurements; single institution.",
        },
    }


def load_frame():
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer(as_frame=True).frame
    df = raw.rename(columns={c: c.replace(" ", "_") for c in raw.columns})
    df.insert(0, "tumour_id", [f"T{i:03d}" for i in range(len(df))])
    df["malignant"] = (df.pop("target") == 0).astype(int)
    return df


def write_files(root: str | Path, seed: int = 20261114, confirm_fraction: float = 0.4) -> dict[str, Any]:
    root = Path(root)
    data_dir = ensure_dir(root / "data")
    examples = ensure_dir(root / "examples" / "wdbc")
    df = load_frame()

    # Stratified split by unit (no groups): deterministic from the seed.
    rng = rng_for("wdbc-split", seed)
    conf_idx = []
    for label in (0, 1):
        idx = np.where(df["malignant"].to_numpy() == label)[0]
        conf_idx.extend(rng.permutation(idx)[: int(round(len(idx) * confirm_fraction))].tolist())
    is_conf = np.zeros(len(df), dtype=bool)
    is_conf[conf_idx] = True
    disc, conf = df[~is_conf], df[is_conf]
    disc_path, conf_path = data_dir / "wdbc_discovery.csv", data_dir / "wdbc_confirmation.csv"
    disc.to_csv(disc_path, index=False)
    conf.to_csv(conf_path, index=False)

    contract_path = examples / "contract.yaml"
    contract_path.write_text(
        yaml.safe_dump(make_contract_dict(confirmation_n=int(len(conf))), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    vault = Vault(ensure_dir(root / "vault"))
    disc_meta = vault.ingest(disc_path, role="discovery", name="WDBC discovery split", source="sklearn load_breast_cancer",
                             license="CC BY 4.0", outcome_column="malignant")
    key_path = vault.make_key("wdbc-confirmation")
    conf_meta = vault.ingest(conf_path, role="sealed", name="WDBC sealed confirmation split", source="sklearn load_breast_cancer",
                             license="CC BY 4.0", sealed=True, outcome_column="malignant", budget=40, key_path=key_path)
    # The plaintext confirmation file must not stay where the search can read it.
    conf_path.unlink()

    contract = load_contract(contract_path, disc_path)
    Registry(ensure_dir(root / "state")).register_contract(contract, registered_by="wdbc-bootstrap")
    return {
        "domain": "wdbc",
        "root": str(root),
        "contract_path": str(contract_path),
        "discovery_path": str(disc_path),
        "discovery_meta": disc_meta,
        "confirmation_meta": conf_meta,
        "key_path": str(key_path),
        "contract_hash": contract.hash,
        "n_discovery": int(len(disc)),
        "n_confirmation": int(len(conf)),
        "limitations": LIMITATIONS,
        "brief": __doc__,
    }
