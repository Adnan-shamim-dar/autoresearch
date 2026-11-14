from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .certificate import build_certificate, load_or_create_key, save_certificate, sign_certificate
from .contract import Contract, DecisionSpec, load_contract
from .power import assess_continuous
from .registry import Registry
from .utils import atomic_write_json, ensure_dir, now_iso, sha256_file, sha256_obj
from .vault import SealedLabelService, Vault

FEATURES = [
    {
        "name": "binding_rank",
        "direction": "lower_better",
        "transform": "rank",
        "description": "Predicted HLA binding percentile rank (lower is a stronger binder)",
    },
    {
        "name": "stability",
        "direction": "higher_better",
        "transform": "rank",
        "description": "Predicted pMHC stability half-life (hours)",
    },
    {
        "name": "expression",
        "direction": "higher_better",
        "transform": "log_rank",
        "description": "Tumor transcript abundance (TPM, log-rank)",
    },
    {
        "name": "foreignness",
        "direction": "higher_better",
        "transform": "rank",
        "description": "Distance from the self-proteome / non-self novelty score",
    },
    {
        "name": "presentation",
        "direction": "higher_better",
        "transform": "rank",
        "description": "MS immunopeptidome presentation score",
    },
    {
        "name": "self_similarity",
        "direction": "lower_better",
        "transform": "rank",
        "description": "Similarity to a tolerated self-peptide (higher is riskier)",
    },
    {
        "name": "processing",
        "direction": "higher_better",
        "transform": "rank",
        "description": "Proteasomal cleavage / antigen-processing score",
    },
    {
        "name": "publicness",
        "direction": "higher_better",
        "transform": "rank",
        "description": "Public TCR clonotype frequency in the repertoire",
    },
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


def generate_immunotherapy_data(
    seed: int = 20261114,
    n_patients: int = 480,
    target_prevalence: float = 0.19,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    indications = ["melanoma", "NSCLC", "PDAC"]
    sites = ["S1", "S2", "S3", "S4"]
    hlas = ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"]
    for p in range(n_patients):
        pid = f"P{p+1:03d}"
        indication = indications[p % len(indications)]
        site = sites[(p // 3) % len(sites)]
        hla = hlas[p % len(hlas)]
        n_pep = int(rng.integers(18, 31))
        binding_rank = 10 ** rng.uniform(-2.2, -0.08, size=n_pep)
        stability = np.exp(rng.normal(1.2, 0.8, size=n_pep))
        expression = np.exp(rng.normal(4.0, 1.3, size=n_pep))
        foreignness = rng.beta(2.0, 5.0, size=n_pep)
        presentation = np.clip(0.65 * (1 - binding_rank) + rng.beta(4, 3, size=n_pep) * 0.35, 0, 1)
        self_similarity = rng.beta(2.2, 5.0, size=n_pep)
        processing = np.clip(0.5 * (1 - binding_rank) + rng.beta(5, 3, size=n_pep) * 0.5, 0, 1)
        publicness = rng.beta(2.0, 6.0, size=n_pep)
        hydrophobicity = rng.normal(0.0, 1.0, size=n_pep)
        for j in range(n_pep):
            rows.append(
                {
                    "neoepitope_id": f"{pid}-N{j+1:02d}",
                    "patient_id": pid,
                    "indication": indication,
                    "site": site,
                    "hla": hla,
                    "binding_rank": float(binding_rank[j]),
                    "stability": float(stability[j]),
                    "expression": float(expression[j]),
                    "foreignness": float(foreignness[j]),
                    "presentation": float(presentation[j]),
                    "self_similarity": float(self_similarity[j]),
                    "processing": float(processing[j]),
                    "publicness": float(publicness[j]),
                    "hydrophobicity": float(hydrophobicity[j]),
                    "source": "synthetic_demo_TESLA_like",
                }
            )
    df = pd.DataFrame(rows)

    # A plausible latent immunogenicity model, scaled so that a multi-feature
    # panel has a clear advantage over binding-only. This is a planted-signal
    # test fixture, not a claim about real biology: the platform's job is to
    # certify whatever signal is present, or refuse when it cannot.
    signal = 1.25 * (
        0.60 * _z(-np.log10(df["binding_rank"].to_numpy()))
        + 1.00 * _z(np.log1p(df["expression"].to_numpy()))
        + 0.80 * _z(np.log1p(df["stability"].to_numpy()))
        + 0.75 * _z(df["foreignness"].to_numpy())
        + 0.50 * _z(df["presentation"].to_numpy())
        - 0.50 * _z(df["self_similarity"].to_numpy())
        + 0.25 * _z(df["processing"].to_numpy())
        + 0.15 * _z(df["publicness"].to_numpy())
        + 0.05 * _z(df["hydrophobicity"].to_numpy())
    )
    # Add patient-level heterogeneity without erasing the mechanism.
    patient_shift = {pid: rng.normal(0, 0.18) for pid in df["patient_id"].unique()}
    signal = signal + df["patient_id"].map(patient_shift).to_numpy()

    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        prev = _sigmoid(mid + 0.95 * signal).mean()
        if prev < target_prevalence:
            lo = mid
        else:
            hi = mid
    intercept = (lo + hi) / 2
    p = _sigmoid(intercept + 0.95 * signal)
    df["immunogenic"] = rng.binomial(1, p).astype(int)
    # Deployed baseline: binding affinity alone, with measurement noise.
    df["baseline_score"] = -np.log10(df["binding_rank"].to_numpy()) + rng.normal(0, 0.18, len(df))
    return df


def make_contract_dict(
    effect: float = 0.04,
    k_candidates: int = 150,
    confirmation_n: int = 2400,
    confirmation_groups: int = 160,
) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "unit": "neoepitope_id",
        "group": "patient_id",
        "outcome": {
            "column": "immunogenic",
            "type": "binary",
            "measured_by": "validated T-cell recognition (multimer, ELISpot or cytotoxicity)",
        },
        "candidates": {
            "grammar": "weighted_rank_fusion",
            "max_terms": None,
            "features": FEATURES,
        },
        "baseline": {
            "name": "netmhcpan_binding_only",
            "column": "baseline_score",
            "higher_is_better": True,
            "description": "Binding affinity alone — the deployed NetMHCpan-style baseline",
        },
        "metric": {
            "primary": "paired_delta_auc",
            "secondary": ["auc", "top3_enrichment", "frac_improved", "mean_pct_rank", "rank_coef"],
            "top_k": 3,
        },
        "contexts": ["site", "indication"],
        "constraints": {"interpretable": True, "max_terms": None, "max_latency_ms": 50},
        "decision": {
            "action": "advance the multi-feature prioritization panel to prospective validation",
            "smallest_effect_that_matters": effect,
            "alpha": 0.05,
            "target_power": 0.8,
            "max_runs": 1,
        },
        "power": {
            "k_candidates": k_candidates,
            "confirmation_n": confirmation_n,
            "confirmation_groups": confirmation_groups,
        },
        "adversary": {
            "permutations": 79,
            "pseudo_group_splits": 20,
            "baseline_advocate_candidates": 24,
        },
        "provenance": {
            "discovery_license": "synthetic demo substrate (CC0); replace with real data",
            "confirmation_license": "synthetic demo substrate (CC0); sealed in the vault",
            "notes": "TESLA/CEDAR-style schema; not a clinical dataset.",
        },
    }


def write_demo_files(
    root: str | Path,
    seed: int = 20261114,
    reset: bool = True,
    n_patients: int = 480,
    n_confirmation_groups: int = 160,
) -> dict[str, Any]:
    root = Path(root)
    data_dir = ensure_dir(root / "data")
    examples = ensure_dir(root / "examples" / "immunotherapy")
    vault_root = ensure_dir(root / "vault")
    if reset:
        import shutil

        # The calibration OC table describes the gate, not this substrate, and
        # takes minutes to rebuild, so a substrate reset keeps it.
        for sub in ("vault", "state", "runs"):
            p = root / sub
            if p.exists():
                shutil.rmtree(p)
        ensure_dir(vault_root)
        ensure_dir(root / "state")
        ensure_dir(root / "runs")
        ensure_dir(root / "calibration")

    df = generate_immunotherapy_data(seed=seed, n_patients=n_patients)
    full_path = data_dir / "immunotherapy_demo.csv"
    df.to_csv(full_path, index=False)

    rng = np.random.default_rng(seed + 1)
    pids = sorted(df["patient_id"].unique())
    perm = list(rng.permutation(pids))
    n_conf = int(n_confirmation_groups)
    if n_conf >= len(perm) - 4:
        raise ValueError("Confirmation split leaves too few patients for discovery.")
    conf_pids = sorted(perm[:n_conf])
    disc_pids = sorted(perm[n_conf:])
    disc = df[df["patient_id"].isin(disc_pids)].copy()
    conf = df[df["patient_id"].isin(conf_pids)].copy()

    # Guarantee both classes on each side; otherwise re-split deterministically.
    if disc["immunogenic"].nunique() < 2 or conf["immunogenic"].nunique() < 2:
        raise RuntimeError("Demo split lost a class; adjust the seed.")
    disc_path = data_dir / "immunotherapy_discovery.csv"
    conf_path = data_dir / "immunotherapy_confirmation.csv"
    disc.to_csv(disc_path, index=False)
    conf.to_csv(conf_path, index=False)

    contract_dict = make_contract_dict(
        effect=0.04,
        k_candidates=150,
        confirmation_n=int(len(conf)),
        confirmation_groups=int(conf["patient_id"].nunique()),
    )
    contract_path = examples / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract_dict, sort_keys=False, allow_unicode=True), encoding="utf-8")

    vault = Vault(vault_root)
    disc_meta = vault.ingest(
        disc_path,
        role="discovery",
        name="Immunotherapy neoantigen discovery cohort (synthetic demo)",
        source="demo generator; TESLA/CEDAR-style schema",
        license="CC0 synthetic",
        outcome_column="immunogenic",
        group_column="patient_id",
    )
    key_path = vault.make_key("demo-confirmation")
    conf_meta = vault.ingest(
        conf_path,
        role="sealed",
        name="Immunotherapy neoantigen sealed confirmation cohort (synthetic demo)",
        source="demo generator; sealed in the vault",
        license="CC0 synthetic",
        sealed=True,
        outcome_column="immunogenic",
        group_column="patient_id",
        budget=180,
        key_path=key_path,
    )

    # Register the contract at creation, not at success.
    contract = load_contract(contract_path, disc_path)
    registry = Registry(root / "state")
    registry.register_contract(contract, registered_by="demo-bootstrap")

    from .metrics import auc as _auc

    info = {
        "root": str(root),
        "contract_path": str(contract_path),
        "discovery_path": str(disc_path),
        "confirmation_path": str(conf_path),
        "full_path": str(full_path),
        "discovery_meta": disc_meta,
        "confirmation_meta": conf_meta,
        "key_path": str(key_path),
        "contract_hash": contract.hash,
        "generator": {
            "seed": int(seed),
            "n_patients": int(n_patients),
            "n_confirmation_groups": int(n_conf),
            "effect": 0.04,
            "k_candidates": 150,
        },
        "n_discovery": int(len(disc)),
        "n_confirmation": int(len(conf)),
        "prevalence_discovery": float(disc["immunogenic"].mean()),
        "prevalence_confirmation": float(conf["immunogenic"].mean()),
        "baseline_auc_discovery": float(_auc(disc["immunogenic"].to_numpy(), disc["baseline_score"].to_numpy())),
    }
    atomic_write_json(data_dir / "demo_info.json", info)
    return info


def run_demo(
    root: str | Path,
    n_permutations: int = 79,
    candidate_budget: int = 150,
    generations: int = 4,
    event_sink=None,
    llm=None,
) -> dict[str, Any]:
    from .engine import Engine, RunConfig

    root = Path(root)
    info = json.loads((root / "data" / "demo_info.json").read_text(encoding="utf-8")) if (root / "data" / "demo_info.json").exists() else write_demo_files(root)
    contract_path = Path(info["contract_path"])
    disc_path = Path(info["discovery_path"])
    key_path = Path(info["key_path"])
    contract = load_contract(contract_path, disc_path)
    vault = Vault(root / "vault")
    service = SealedLabelService(
        vault,
        info["confirmation_meta"]["dataset_hash"],
        contract,
        key_path,
        budget=info["confirmation_meta"].get("budget_total", 180),
    )
    calibration = []
    cal_path = root / "calibration" / "oc_table.json"
    if cal_path.exists():
        try:
            calibration = json.loads(cal_path.read_text(encoding="utf-8")).get("rows", [])
        except Exception:
            calibration = []
    run_dir = root / "runs" / f"ar-{contract.hash[:8]}-12345"
    cfg = RunConfig(
        contract_path=contract_path,
        discovery_path=disc_path,
        run_dir=run_dir,
        vault_root=root / "vault",
        registry_root=root / "state",
        sealed=service,
        sealed_dataset_hash=info["confirmation_meta"]["dataset_hash"],
        run_seed=12345,
        n_permutations=n_permutations,
        n_splits=5,
        candidate_budget=candidate_budget,
        generations=generations,
        llm=llm,
        domain_brief=(Path(__file__).parent / "domains" / "immunotherapy" / "brief.md").read_text(encoding="utf-8")
        if (Path(__file__).parent / "domains" / "immunotherapy" / "brief.md").exists()
        else "",
        calibration_table=calibration,
        key_path=root / "state" / "certificate.key",
        event_sink=event_sink,
        mode="in_process_label_service",
        limitations=[
            "Substrate is a synthetic TESLA/CEDAR-style demo, not the real TESLA or CEDAR data. The same contract runs on the real tables after access is granted.",
            "The gate's operating-characteristics table in this build is calibrated on the synthetic substrate; treat it as a demonstration, not a validated instrument.",
        ],
    )
    engine = Engine(cfg)
    result = engine.run()
    atomic_write_json(root / "data" / "demo_info.json", info)
    return {
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "certificate": result.certificate,
        "power": result.power.to_dict() if result.power else None,
        "events": result.events,
    }


def build_economics_refusal(root: str | Path) -> dict[str, Any]:
    """A first-class refusal built from the report's two usable margin observations.

    This is deliberately not an end-to-end row-level run: the report does not
    contain enough margin observations to run one. The power oracle refuses the
    question as posed and prescribes what would fix it.
    """
    root = Path(root)
    registry = Registry(root / "state")
    power = assess_continuous(
        n_per_group=1,
        sd=0.12,
        effect=0.10,
        alpha=0.05,
        target_power=0.8,
    )
    raw = {
        "contract_version": 1,
        "unit": "modality",
        "group": "modality_class",
        "outcome": {
            "column": "gross_margin",
            "type": "continuous",
            "measured_by": "reported GAAP product gross margin",
        },
        "candidates": {
            "grammar": "linear_score",
            "features": [
                {"name": "personalization", "direction": "lower_better", "description": "Autologous=N=1 vs off-the-shelf"},
                {"name": "turnaround_days", "direction": "lower_better", "description": "Vein-to-vein or batch release time"},
                {"name": "price", "direction": "higher_better", "description": "List price per course"},
            ],
        },
        "baseline": {"name": "autologous_TIL_lifileucel", "column": "gross_margin", "higher_is_better": True},
        "metric": {"primary": "mean_difference"},
        "contexts": ["disease"],
        "decision": {
            "action": "choose an off-the-shelf modality over autologous manufacturing on margin",
            "smallest_effect_that_matters": 0.10,
            "alpha": 0.05,
            "target_power": 0.8,
        },
        "provenance": {
            "notes": "Two usable margin observations in the report: TIL 56.1%, Immunocore 99.6%. No row-level test is possible.",
        },
    }
    contract = Contract(raw, path=None)
    # Populate the attributes needed by the certificate builder without running
    # the binary validator (this is a power-only refusal).
    contract.contract_version = 1
    contract.unit = "modality"
    contract.group = "modality_class"
    contract.outcome = raw["outcome"]
    contract.metric_raw = raw["metric"]
    contract.baseline = __import__("autoresearch.contract", fromlist=["BaselineSpec"]).BaselineSpec(
        name="autologous_TIL_lifileucel", column="gross_margin"
    )
    contract.decision = DecisionSpec(
        action=raw["decision"]["action"],
        smallest_effect_that_matters=0.10,
        alpha=0.05,
        target_power=0.8,
        max_runs=1,
    )
    registry.register_contract(contract, registered_by="economics-refusal")
    run_id = "ar-econ-refusal"
    run_dir = ensure_dir(root / "runs" / run_id)
    key = load_or_create_key(root / "state" / "certificate.key")
    cert = build_certificate(
        contract=contract,
        run_id=run_id,
        seed=0,
        claim=f"Refused: {power['prescription']}",
        claim_type="refusal",
        status="refused",
        tier=None,
        verdict="refused",
        provenance={
            "datasets": [
                {
                    "hash": sha256_obj({"source": "Personalized Immunotherapy Datasets and Economics.txt", "rows": ["TIL 56.1%", "Immunocore 99.6%"]}),
                    "role": "support_only",
                    "source": "provided report, Q2/Q1 2026 SEC filings as cited",
                    "license": "analysis of public filings",
                }
            ],
            "access_log_hash": sha256_obj({"calls": 0}),
            "access_calls": 0,
            "code_hash": None,
            "split": None,
        },
        search_price={
            "proposed": 0,
            "effective_independent": 0,
            "alpha_allocated": 0.0,
            "alpha_spent": 0.0,
            "selection_null": {"n_permutations": 0, "p_value": None, "max_stat_p95": None},
        },
        gate_oc=None,
        adversary=[],
        commitments={"registered": True, "prediction_hash": None, "confirmed": False},
        interpretability_cost={"terms": None, "max_terms": None, "gap_vs_simple": None},
        refusals=[
            {
                "reason_code": "not_answerable",
                "message": power["prescription"],
                "observed_margin_observations": 2,
            }
        ],
        reproduce="python -m autoresearch power --demo economics",
        evidence={
            "continuous_power": power,
            "observed": {"autologous_TIL_gross_margin": 0.561, "off_the_shelf_Immunocore_gross_margin": 0.996},
            "note": "The two observed margins are consistent with the thesis, but two observations cannot certify it. This refusal is the honest output.",
        },
        limitations=[
            "This is a power-only refusal. No row-level economics test was run because the report does not contain enough observations.",
            "Reported margins are for different products and disease settings; they are not a like-for-like sample.",
        ],
        key=None,
    )
    cert = sign_certificate(cert, key)
    save_certificate(cert, run_dir, key)
    registry.issue_certificate(cert, run_id, str(run_dir))
    atomic_write_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "created_at": now_iso(),
            "status": "refused",
            "reason_code": "not_answerable",
            "mode": "power_only",
            "contract_hash": contract.hash,
            "files": {
                p.name: sha256_file(p)
                for p in sorted(run_dir.glob("*"))
                if p.is_file() and p.name != "run.json"
            },
        },
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "certificate": cert,
        "power": power,
    }
