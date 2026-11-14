"""Library entry points.

``verify(contract, data, candidate) -> certificate`` is the agent tool of
spec §11: another system brings a candidate it already has, and gets back a
signed certificate or a refusal, priced at K=1 and confirmed once on the
sealed set through the out-of-process label service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .contract import load_contract
from .engine import Engine, RunConfig
from .utils import read_json, sha256_obj
from .vault import SERVICE_MODE, LabelServiceProcess, Vault


def _calibration_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "calibration" / "oc_table.json"
    if not path.exists():
        return []
    try:
        return list(read_json(path).get("rows", []))
    except (OSError, ValueError):
        return []


def verify(
    contract_path: str | Path,
    discovery_path: str | Path,
    candidate: dict[str, Any] | str | Path,
    *,
    root: str | Path = ".",
    sealed_dataset: str | None = None,
    vault_root: str | Path | None = None,
    vault_key: str | Path | None = None,
    seed: int = 12345,
    n_permutations: int = 99,
    n_splits: int = 5,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Certify (or refuse) one pre-specified candidate under a contract.

    ``candidate`` is a candidate dict (as found in a certificate's evidence), a
    short spec ``{"weights": {...}, "transforms": {...}, "mechanism": "..."}``,
    or a path to a JSON file holding either. With no ``sealed_dataset`` the
    demo substrate under ``root`` is used.
    """
    root = Path(root)
    if isinstance(candidate, (str, Path)):
        candidate = json.loads(Path(candidate).read_text(encoding="utf-8"))
    if sealed_dataset is None:
        info_path = root / "data" / "demo_info.json"
        if not info_path.exists():
            raise FileNotFoundError("No sealed dataset given and no demo substrate under root; run `autoresearch init`")
        info = read_json(info_path)
        sealed_dataset = info["confirmation_meta"]["dataset_hash"]
        vault_key = vault_key or info["key_path"]
    if vault_key is None:
        raise ValueError("vault_key is required with an explicit sealed_dataset")
    vault = Vault(vault_root or (root / "vault"))
    contract = load_contract(contract_path, discovery_path)
    run_dir = root / "runs" / f"ar-{contract.hash[:8]}-{seed}-v{sha256_obj(candidate)[:8]}"
    cfg = RunConfig(
        contract_path=contract_path,
        discovery_path=discovery_path,
        run_dir=run_dir,
        vault_root=vault.root,
        registry_root=root / "state",
        sealed=None,
        sealed_dataset_hash=sealed_dataset,
        run_seed=seed,
        n_permutations=n_permutations,
        n_splits=n_splits,
        calibration_table=_calibration_rows(root),
        key_path=root / "state" / "certificate.key",
        event_sink=event_sink,
        mode=SERVICE_MODE,
        fixed_candidate=candidate,
    )
    budget = vault.meta(sealed_dataset).get("budget_total")
    with LabelServiceProcess(vault.root, sealed_dataset, contract_path, vault_key, budget=budget) as client:
        cfg.sealed = client
        result = Engine(cfg).run()
    return result.certificate
