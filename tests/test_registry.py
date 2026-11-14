"""Registry: hash-chained ledger, contract registration, and the alpha ledger."""

from __future__ import annotations

import json

import pytest

from autoresearch.registry import Registry, RegistryError


@pytest.fixture()
def registry(tmp_path):
    return Registry(tmp_path / "state")


def test_empty_registry_verifies(registry):
    ok, reason, n = registry.verify()
    assert ok and reason is None and n == 0


def test_contract_registration_is_idempotent_and_snapshotted(registry, contract):
    first = registry.register_contract(contract)
    second = registry.register_contract(contract)
    assert first["hash"] == second["hash"]
    assert len(registry.entries("contract_registered")) == 1
    assert (registry.contracts_dir / f"{contract.hash}.yaml").exists()


def test_alpha_is_conserved_across_runs(registry):
    ds = "dataset-1"
    state = registry.ensure_alpha(ds, total=0.10)
    assert state["allocated"] == pytest.approx(0.10)
    # A second ensure must not top the budget up.
    registry.ensure_alpha(ds, total=0.10)
    assert registry.alpha_state(ds)["allocated"] == pytest.approx(0.10)

    registry.spend_alpha(ds, 0.05, "run-a", "first test")
    registry.spend_alpha(ds, 0.05, "run-b", "second test")
    assert registry.alpha_state(ds)["remaining"] == pytest.approx(0.0)
    assert not registry.can_spend_alpha(ds, 0.05)
    with pytest.raises(RegistryError):
        registry.spend_alpha(ds, 0.05, "run-c", "third test")


def test_alpha_is_tracked_per_dataset(registry):
    registry.ensure_alpha("a", 0.05)
    registry.ensure_alpha("b", 0.05)
    registry.spend_alpha("a", 0.05, "run", "spend a")
    assert registry.alpha_state("a")["remaining"] == pytest.approx(0.0)
    assert registry.alpha_state("b")["remaining"] == pytest.approx(0.05)


def test_prediction_commitment_is_first_write_wins(registry):
    first = registry.commit_prediction("c1", "run-1", {"predicted_failure": "overfit"})
    second = registry.commit_prediction("c1", "run-1", {"predicted_failure": "changed my mind"})
    assert first["prediction_hash"] == second["prediction_hash"]
    assert registry.prediction("c1", "run-1")["prediction"]["predicted_failure"] == "overfit"


def test_certificate_issue_and_summary(registry, contract):
    registry.register_contract(contract)
    registry.issue_certificate({"certificate_hash": "h1", "tier": "confirmed"}, "run-1")
    registry.issue_certificate({"certificate_hash": "h1", "tier": "confirmed"}, "run-1")
    registry.issue_certificate({"certificate_hash": "h2", "status": "refused"}, "run-2")
    summary = registry.summary()
    assert summary["certificates_issued"] == 2
    assert summary["denominator"] == 1
    assert summary["tiers"] == {"confirmed": 1, "refused": 1}
    assert summary["ledger_head"] == registry.entries()[-1]["hash"]


@pytest.mark.parametrize("field,value", [("amount", 0.0001), ("dataset_id", "other"), ("prev_hash", "GENESIS")])
def test_ledger_tamper_is_detected(registry, field, value):
    registry.ensure_alpha("ds", 0.05)
    registry.spend_alpha("ds", 0.05, "run-1", "spend")
    registry.issue_certificate({"certificate_hash": "h"}, "run-1")
    assert registry.verify()[0]

    rows = [json.loads(line) for line in registry.ledger_path.read_text(encoding="utf-8").splitlines()]
    rows[1][field] = value
    registry.ledger_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    ok, reason, _ = registry.verify()
    assert not ok
    assert reason == "entry 1 hash mismatch"


def test_deleting_an_entry_breaks_the_chain(registry):
    registry.ensure_alpha("ds", 0.05)
    registry.spend_alpha("ds", 0.02, "run-1", "spend")
    registry.spend_alpha("ds", 0.02, "run-2", "spend")
    lines = registry.ledger_path.read_text(encoding="utf-8").splitlines()
    # Erasing a spend would refund alpha; the chain must catch it.
    registry.ledger_path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    ok, reason, _ = registry.verify()
    assert not ok
    assert reason == "entry 1 hash mismatch"


def test_procedure_registration_is_first_write_wins(registry):
    plan = {"selection_rule": "max cv delta", "seed": 1}
    first = registry.register_procedure("c1", "run-1", plan)
    assert registry.register_procedure("c1", "run-1", dict(plan))["hash"] == first["hash"]
    with pytest.raises(RegistryError):
        registry.register_procedure("c1", "run-1", {**plan, "seed": 2})
    assert registry.summary()["procedures_registered"] == 1


def test_certificates_start_private_and_publication_is_logged(registry):
    """Spec §12.6."""
    registry.issue_certificate({"certificate_hash": "r1", "status": "refused", "refusals": [{"reason_code": "underpowered"}]}, "run-1")
    assert registry.entries("certificate_issued")[0]["visibility"] == "private"
    assert not registry.is_published("r1")
    entry = registry.publish_certificate("r1", published_by="lab lead", note="for the paper")
    assert registry.is_published("r1")
    assert entry["published_by"] == "lab lead"
    with pytest.raises(RegistryError):
        registry.publish_certificate("never-issued", published_by="x")


def test_refusal_taxonomy_aggregates_reasons(registry):
    """Spec §12.7."""
    registry.issue_certificate({"certificate_hash": "a", "status": "refused", "refusals": [{"reason_code": "not_answerable"}]}, "r1")
    registry.issue_certificate({"certificate_hash": "b", "status": "refused", "refusals": [{"reason_code": "not_answerable"}]}, "r2")
    registry.issue_certificate({"certificate_hash": "c", "status": "issued", "tier": "exploratory", "verdict": "not_affirmed"}, "r3")
    registry.issue_certificate({"certificate_hash": "d", "status": "issued", "tier": "confirmed", "verdict": "affirmed"}, "r4")
    tax = registry.refusal_taxonomy()
    assert tax["runs"] == 4
    assert tax["outcomes"] == {"refused": 2, "not_affirmed:exploratory": 1, "affirmed": 1}
    assert tax["reasons"]["not_answerable"] == 2
    assert tax["reasons"]["not_affirmed:exploratory"] == 1


def test_operator_decisions_are_chained(registry):
    """Spec §12.2."""
    registry.log_operator_decision("demo", {"reset": True}, "ok")
    assert registry.entries("operator_decision")[0]["command"] == "demo"
    assert registry.summary()["operator_decisions"] == 1
    assert registry.verify()[0]
