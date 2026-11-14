"""End-to-end: substrate -> engine -> certificate -> CLI verify.

These use a deliberately small substrate and few permutations so the suite
stays fast. With 9 permutations the smallest attainable selection-null p is
0.1, so the issued run tops out at the ``exploratory`` tier by construction;
the point here is the plumbing, not the tier (that is what the OC table is for).
"""

from __future__ import annotations

import json

import pytest

from autoresearch.certificate import verify_certificate
from autoresearch.cli import main as cli_main
from autoresearch.demo import run_demo, write_demo_files
from autoresearch.registry import Registry
from autoresearch.utils import read_json, read_jsonl
from autoresearch.vault import Vault

pytestmark = pytest.mark.integration

FAST = dict(n_permutations=9, candidate_budget=12, generations=1)


@pytest.fixture(scope="module")
def issued(tmp_path_factory):
    root = tmp_path_factory.mktemp("issued")
    write_demo_files(root, seed=11, reset=True, n_patients=200, n_confirmation_groups=160)
    result = run_demo(root, **FAST)
    return root, result


@pytest.fixture()
def refused(tmp_path):
    root = tmp_path / "refused"
    write_demo_files(root, seed=11, reset=True, n_patients=120, n_confirmation_groups=40)
    return root


def test_issued_certificate_is_signed_and_verifies(issued):
    root, result = issued
    cert = result["certificate"]
    assert cert["status"] == "issued"
    assert cert["tier"] in {"confirmed", "supported", "exploratory"}
    key = (root / "state" / "certificate.key").read_bytes()
    assert verify_certificate(cert, key)["valid"]
    assert read_json(root / "runs" / result["run_id"] / "certificate.json") == cert


def test_run_manifest_matches_artifacts(issued):
    root, result = issued
    assert cli_main(["--root", str(root), "verify", "--run", result["run_id"]]) == 0


def test_ledgers_verify_and_alpha_was_spent(issued):
    root, result = issued
    registry = Registry(root / "state")
    assert registry.verify()[0]
    assert Vault(root / "vault").verify_ledger()[0]
    sealed = [d for d in result["certificate"]["provenance"]["datasets"] if d["role"] == "sealed"]
    ds = sealed[0]["hash"]
    assert registry.alpha_state(ds)["spent"] > 0


def test_event_log_is_one_run_with_unique_sequence(issued):
    root, result = issued
    events = read_jsonl(root / "runs" / result["run_id"] / "events.jsonl")
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(1, len(seqs) + 1))
    assert events[-1]["stage"] == "certificate"


def test_refusal_is_first_class_and_signed(refused):
    result = run_demo(refused, **FAST)
    cert = result["certificate"]
    assert cert["status"] == "refused"
    assert cert["claim"].startswith("Refused:")
    key = (refused / "state" / "certificate.key").read_bytes()
    assert verify_certificate(cert, key)["valid"]
    # The manifest must cover the final state of every artifact, events included.
    assert cli_main(["--root", str(refused), "verify", "--run", result["run_id"]]) == 0


def test_rerun_into_same_dir_does_not_mix_event_logs(refused):
    first = run_demo(refused, **FAST)
    run_dir = refused / "runs" / first["run_id"]
    n_first = len(read_jsonl(run_dir / "events.jsonl"))
    (run_dir / "user_notes.txt").write_text("keep me", encoding="utf-8")

    second = run_demo(refused, **FAST)

    events = read_jsonl(run_dir / "events.jsonl")
    assert len(events) == len(second["events"])
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert n_first > 0
    # Files the engine did not write are never removed.
    assert (run_dir / "user_notes.txt").read_text(encoding="utf-8") == "keep me"
    assert cli_main(["--root", str(refused), "verify", "--run", second["run_id"]]) == 0


def test_verify_fails_on_tampered_artifact(refused):
    result = run_demo(refused, **FAST)
    run_dir = refused / "runs" / result["run_id"]
    assert cli_main(["--root", str(refused), "verify", "--run", result["run_id"]]) == 0
    with open(run_dir / "events.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"seq": 999, "stage": "forged"}) + "\n")
    assert cli_main(["--root", str(refused), "verify", "--run", result["run_id"]]) == 3


def test_verify_fails_on_tampered_certificate(refused):
    result = run_demo(refused, **FAST)
    path = refused / "runs" / result["run_id"] / "certificate.json"
    cert = read_json(path)
    cert["claim"] = "Confirmed: everything works"
    path.write_text(json.dumps(cert), encoding="utf-8")
    assert cli_main(["--root", str(refused), "verify", "--run", result["run_id"]]) == 2


def test_economics_refusal_manifest_covers_its_artifacts(refused):
    from autoresearch.demo import build_economics_refusal

    result = build_economics_refusal(refused)
    run_json = read_json(refused / "runs" / result["run_id"] / "run.json")
    assert {"certificate.json", "certificate.md"} <= set(run_json["files"])
    assert cli_main(["--root", str(refused), "verify", "--run", result["run_id"]]) == 0


def test_substrate_reset_keeps_calibration(tmp_path):
    cal = tmp_path / "calibration"
    cal.mkdir()
    (cal / "oc_table.json").write_text("{}", encoding="utf-8")
    write_demo_files(tmp_path, seed=3, reset=True, n_patients=120, n_confirmation_groups=40)
    write_demo_files(tmp_path, seed=3, reset=True, n_patients=120, n_confirmation_groups=40)
    assert (cal / "oc_table.json").exists()
