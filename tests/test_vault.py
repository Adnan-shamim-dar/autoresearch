from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

from autoresearch.scoring import make_candidate
from autoresearch.vault import (
    SERVICE_MODE,
    BudgetExhausted,
    LabelServiceProcess,
    PermissionDenied,
    SealedClient,
    SealedLabelService,
    Vault,
    VaultError,
    is_os_sealed,
)


@pytest.fixture()
def sealed(tmp_path, small_frame, contract):
    vault = Vault(tmp_path / "vault")
    key_path = vault.make_key("confirmation")
    meta = vault.ingest_dataframe(
        small_frame,
        role="sealed",
        name="confirmation",
        sealed=True,
        outcome_column="immunogenic",
        group_column="patient_id",
        budget=2,
        key_path=key_path,
    )
    return vault, key_path, meta


def test_direct_read_of_sealed_data_is_denied_and_logged(sealed):
    vault, key_path, meta = sealed
    with pytest.raises(PermissionDenied):
        vault.read(meta["dataset_hash"], actor="search", capability="search")
    entries = vault.ledger_entries()
    assert any(e["type"] == "read_denied" and e["dataset"] == meta["dataset_hash"] for e in entries)
    with pytest.raises(PermissionDenied):
        vault.read(meta["dataset_hash"], actor="label_service", capability="confirm")


def test_only_label_service_can_decrypt(sealed):
    vault, key_path, meta = sealed
    with pytest.raises(PermissionDenied):
        vault.decrypt_sealed(meta["dataset_hash"], key_path, actor="search")
    df = vault.decrypt_sealed(meta["dataset_hash"], key_path, actor="label_service")
    assert len(df) == meta["n_rows"]


def test_confirm_decrements_budget_and_then_refuses(sealed, contract):
    vault, key_path, meta = sealed
    service = SealedLabelService(vault, meta["dataset_hash"], contract, key_path, budget=2)
    candidate = make_candidate(
        {"expression": 1.0},
        mechanism="test mechanism",
        family="single",
        cid="c-test",
    )

    first = service.confirm(candidate, seed=3, bootstrap=80)
    assert first["budget_remaining"] == 1
    assert first["n"] == meta["n_rows"]
    assert "ci_low" in first and "ci_high" in first

    second = service.confirm(candidate, seed=4, bootstrap=80)
    assert second["budget_remaining"] == 0

    with pytest.raises(BudgetExhausted):
        service.confirm(candidate)
    assert any(e["type"] == "confirm_denied" for e in vault.ledger_entries())


def test_twin_exists_and_shares_structure(sealed, small_frame):
    vault, key_path, meta = sealed
    twin = vault.twin(meta["dataset_hash"])
    assert len(twin) == len(small_frame)
    assert list(twin["neoepitope_id"]) == list(small_frame["neoepitope_id"])
    assert list(twin["patient_id"]) == list(small_frame["patient_id"])


def test_ledger_hash_chain_detects_tampering(sealed):
    vault, key_path, meta = sealed
    ok, err, n = vault.verify_ledger()
    assert ok and err is None and n > 0

    rows = vault.ledger_entries()
    rows[0]["role"] = "benchmark_only"
    vault.ledger_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    ok, err, n = vault.verify_ledger()
    assert ok is False
    assert "hash mismatch" in (err or "")


def test_discovery_role_can_read(tmp_path, small_frame):
    vault = Vault(tmp_path / "discovery-vault")
    discovery = vault.ingest_dataframe(
        small_frame,
        role="discovery",
        name="discovery",
        outcome_column="immunogenic",
        group_column="patient_id",
    )
    df = vault.read(discovery["dataset_hash"], actor="search", capability="search", purpose="unit test")
    assert len(df) == len(small_frame)
    assert any(e["type"] == "read" for e in vault.ledger_entries())


def test_same_content_cannot_be_relabelled(sealed, small_frame):
    vault, key_path, meta = sealed
    with pytest.raises(VaultError) as exc:
        vault.ingest_dataframe(
            small_frame,
            role="discovery",
            name="discovery",
            outcome_column="immunogenic",
            group_column="patient_id",
        )
    assert "refusing to relabel" in str(exc.value)


# ---------------------------------------------------------------------------
# Spec §2: sealing is physical, not conventional
# ---------------------------------------------------------------------------

def _candidate():
    return make_candidate({"expression": 1.0}, mechanism="test mechanism", family="single", cid="c-test")


def test_direct_read_parquet_of_sealed_file_raises_permission_error(sealed):
    """Acceptance test 1: the OS, not this library, refuses the read."""
    vault, key_path, meta = sealed
    blob = vault.root / meta["dataset_hash"] / meta["data_file"]
    if not is_os_sealed(blob):
        pytest.skip("OS lock not enforceable for this account (e.g. running as root)")
    with pytest.raises(PermissionError):
        pd.read_parquet(blob)
    with pytest.raises(PermissionError):
        key_path.read_bytes()


def test_search_process_cannot_read_sealed_file_or_key(sealed):
    """Same check from a separate process, as the search process would be."""
    vault, key_path, meta = sealed
    blob = vault.root / meta["dataset_hash"] / meta["data_file"]
    if not is_os_sealed(blob):
        pytest.skip("OS lock not enforceable for this account")
    code = (
        "import sys, pandas as pd\n"
        "for fn in (lambda: pd.read_parquet(sys.argv[1]), lambda: open(sys.argv[2], 'rb').read()):\n"
        "    try:\n"
        "        fn(); print('READ')\n"
        "    except PermissionError:\n"
        "        print('DENIED')\n"
    )
    out = subprocess.run([sys.executable, "-c", code, str(blob), str(key_path)], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["DENIED", "DENIED"]


def test_lock_is_restored_after_the_label_service_reads(sealed):
    vault, key_path, meta = sealed
    blob = vault.root / meta["dataset_hash"] / meta["data_file"]
    if not is_os_sealed(blob):
        pytest.skip("OS lock not enforceable for this account")
    df = vault.decrypt_sealed(meta["dataset_hash"], key_path, actor="label_service")
    assert len(df) == meta["n_rows"]
    assert is_os_sealed(blob)
    assert is_os_sealed(key_path)


@pytest.fixture()
def service_process(sealed, contract, tmp_path):
    vault, key_path, meta = sealed
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(contract.canonical_yaml(), encoding="utf-8")
    with LabelServiceProcess(vault.root, meta["dataset_hash"], contract_path, key_path, budget=2) as client:
        yield vault, meta, client


@pytest.mark.integration
def test_label_service_runs_in_a_separate_process(service_process):
    vault, meta, client = service_process
    info = client.meta()
    assert info["mode"] == SERVICE_MODE
    assert info["service_pid"] != os.getpid()
    result = client.confirm(_candidate(), seed=3, bootstrap=80)
    assert result["n"] == meta["n_rows"]
    # Only aggregates cross the boundary.
    assert not {"rows", "labels", "y", meta["outcome_column"]} & set(result)


@pytest.mark.integration
def test_budget_exhaustion_across_the_boundary_raises_and_is_logged(service_process):
    """Acceptance test 2, through the process boundary."""
    vault, meta, client = service_process
    client.confirm(_candidate(), seed=3, bootstrap=40)
    client.confirm(_candidate(), seed=4, bootstrap=40)
    with pytest.raises(BudgetExhausted):
        client.confirm(_candidate(), seed=5, bootstrap=40)
    assert any(e["type"] == "confirm_denied" for e in vault.ledger_entries())
    assert vault.verify_ledger()[0]


@pytest.mark.integration
def test_wrong_token_is_rejected(service_process):
    _vault, _meta, client = service_process
    forged = SealedClient(client.base_url, "not-the-token")
    with pytest.raises(PermissionDenied):
        forged.meta()
