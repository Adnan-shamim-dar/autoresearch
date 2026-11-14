from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .contract import Contract
from .utils import (
    append_jsonl,
    atomic_write_text,
    canonical_json,
    chain_hash,
    ensure_dir,
    now_iso,
    read_jsonl,
    sha256_obj,
)


class RegistryError(Exception):
    pass


class Registry:
    """Procedure registration, prediction commitments, and a conserved alpha ledger.

    Alpha is accounted per dataset lifetime, not per run. The registry exposes the
    denominator: every contract registered at creation, including the ones that
    never produced a certificate.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        ensure_dir(self.root)
        self.ledger_path = self.root / "registry.jsonl"
        self.contracts_dir = ensure_dir(self.root / "contracts")
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def entries(self, kind: str | None = None) -> list[dict]:
        rows = read_jsonl(self.ledger_path)
        if kind:
            rows = [r for r in rows if r.get("type") == kind]
        return rows

    def _append(self, payload: dict[str, Any]) -> dict:
        with self._lock:
            rows = self.entries()
            prev = rows[-1]["hash"] if rows else "GENESIS"
            body = dict(payload)
            body["ts"] = now_iso()
            body["prev_hash"] = prev
            body["hash"] = chain_hash(prev, body)
            append_jsonl(self.ledger_path, body)
            return body

    def verify(self) -> tuple[bool, str | None, int]:
        rows = self.entries()
        prev = "GENESIS"
        for i, row in enumerate(rows):
            body = dict(row)
            claimed = body.pop("hash", None)
            if claimed is None:
                return False, f"entry {i} has no hash", len(rows)
            expected = chain_hash(prev, body)
            if claimed != expected:
                return False, f"entry {i} hash mismatch", len(rows)
            prev = claimed
        return True, None, len(rows)

    # ------------------------------------------------------------------
    def register_contract(self, contract: Contract, registered_by: str = "engine") -> dict:
        existing = [e for e in self.entries("contract_registered") if e.get("contract_hash") == contract.hash]
        if existing:
            return existing[0]
        snapshot = self.contracts_dir / f"{contract.hash}.yaml"
        atomic_write_text(snapshot, contract.canonical_yaml())
        entry = self._append(
            {
                "type": "contract_registered",
                "contract_hash": contract.hash,
                "contract_version": contract.contract_version,
                "unit": contract.unit,
                "group": contract.group,
                "outcome": contract.outcome,
                "baseline": contract.baseline.name,
                "smallest_effect_that_matters": contract.decision.smallest_effect_that_matters,
                "contract_yaml": contract.canonical_yaml(),
                "registered_by": registered_by,
            }
        )
        return entry

    def commit_prediction(self, contract_hash: str, run_id: str, prediction: dict) -> dict:
        prediction_hash = sha256_obj(prediction)
        existing = [
            e
            for e in self.entries("prediction_committed")
            if e.get("contract_hash") == contract_hash and e.get("run_id") == run_id
        ]
        if existing:
            return existing[0]
        return self._append(
            {
                "type": "prediction_committed",
                "contract_hash": contract_hash,
                "run_id": run_id,
                "prediction": prediction,
                "prediction_hash": prediction_hash,
            }
        )

    def register_procedure(self, contract_hash: str, run_id: str, plan: dict) -> dict:
        """Register the procedure and selection rule before the search runs (spec §8).

        The plan is machine-authored by the engine and hashed; the winner is a
        consequence of it. First write wins, so a plan cannot be edited after
        the fact for the same run id.
        """
        plan_hash = sha256_obj(plan)
        existing = [
            e
            for e in self.entries("procedure_registered")
            if e.get("contract_hash") == contract_hash and e.get("run_id") == run_id
        ]
        if existing:
            if existing[0].get("plan_hash") != plan_hash:
                raise RegistryError(
                    f"Run {run_id} already registered a different procedure ({existing[0].get('plan_hash')[:12]}); "
                    "a changed procedure needs a new run id"
                )
            return existing[0]
        return self._append(
            {
                "type": "procedure_registered",
                "contract_hash": contract_hash,
                "run_id": run_id,
                "plan": plan,
                "plan_hash": plan_hash,
                "authored_by": "engine",
            }
        )

    def prediction(self, contract_hash: str, run_id: str) -> dict | None:
        for e in self.entries("prediction_committed"):
            if e.get("contract_hash") == contract_hash and e.get("run_id") == run_id:
                return e
        return None

    # ------------------------------------------------------------------
    def alpha_state(self, dataset_id: str) -> dict[str, Any]:
        allocated = 0.0
        spent = 0.0
        for e in self.entries():
            if e.get("dataset_id") != dataset_id:
                continue
            if e.get("type") == "alpha_allocated":
                allocated += float(e.get("amount", 0.0))
            elif e.get("type") == "alpha_spent":
                spent += float(e.get("amount", 0.0))
        return {
            "dataset_id": dataset_id,
            "allocated": allocated,
            "spent": spent,
            "remaining": max(0.0, allocated - spent),
        }

    def ensure_alpha(self, dataset_id: str, total: float = 0.05) -> dict[str, Any]:
        state = self.alpha_state(dataset_id)
        if state["allocated"] <= 0:
            self._append(
                {
                    "type": "alpha_allocated",
                    "dataset_id": dataset_id,
                    "amount": float(total),
                    "reason": "lifetime budget for this sealed dataset",
                }
            )
            state = self.alpha_state(dataset_id)
        return state

    def can_spend_alpha(self, dataset_id: str, amount: float) -> bool:
        return self.alpha_state(dataset_id)["remaining"] + 1e-12 >= amount

    def spend_alpha(self, dataset_id: str, amount: float, run_id: str, reason: str) -> dict[str, Any]:
        state = self.alpha_state(dataset_id)
        if state["remaining"] + 1e-12 < amount:
            raise RegistryError(
                f"Alpha budget exhausted for dataset {dataset_id}: remaining {state['remaining']:.4f}, requested {amount:.4f}"
            )
        self._append(
            {
                "type": "alpha_spent",
                "dataset_id": dataset_id,
                "amount": float(amount),
                "run_id": run_id,
                "reason": reason,
            }
        )
        return self.alpha_state(dataset_id)

    # ------------------------------------------------------------------
    def issue_certificate(self, certificate: dict, run_id: str, path: str = "") -> dict:
        cert_hash = certificate.get("certificate_hash") or sha256_obj(certificate)
        existing = [e for e in self.entries("certificate_issued") if e.get("certificate_hash") == cert_hash]
        if existing:
            return existing[0]
        return self._append(
            {
                "type": "certificate_issued",
                "certificate_hash": cert_hash,
                "run_id": run_id,
                "tier": certificate.get("tier"),
                "status": certificate.get("status"),
                "claim_type": certificate.get("claim_type"),
                "contract_hash": certificate.get("contract_hash"),
                "claim": certificate.get("claim"),
                "path": path,
            }
        )

    def certificate_entries(self) -> list[dict]:
        return self.entries("certificate_issued")

    def summary(self) -> dict[str, Any]:
        contracts = self.entries("contract_registered")
        certs = self.entries("certificate_issued")
        predictions = self.entries("prediction_committed")
        procedures = self.entries("procedure_registered")
        alpha: dict[str, dict] = {}
        for e in self.entries():
            ds = e.get("dataset_id")
            if ds:
                alpha[ds] = self.alpha_state(ds)
        tiers: dict[str, int] = {}
        for c in certs:
            key = str(c.get("tier") or c.get("status") or "unknown")
            tiers[key] = tiers.get(key, 0) + 1
        return {
            "contracts_registered": len({c.get("contract_hash") for c in contracts}),
            "contract_events": len(contracts),
            "predictions_committed": len(predictions),
            "procedures_registered": len(procedures),
            "certificates_issued": len(certs),
            "denominator": len({c.get("contract_hash") for c in contracts}),
            "tiers": tiers,
            "alpha": alpha,
            "ledger_head": self.entries()[-1]["hash"] if self.entries() else "GENESIS",
        }
