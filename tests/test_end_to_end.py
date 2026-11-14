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


def test_sealed_set_sees_exactly_one_candidate(issued):
    """Spec §6: the number of things confirmation data ever sees is one."""
    root, result = issued
    confirms = [e for e in Vault(root / "vault").ledger_entries() if e.get("type") == "confirm"]
    assert len(confirms) == 1
    assert confirms[0]["purpose"] == "primary confirmation"
    price = result["certificate"]["search_price"]
    assert price["sealed_queries"] == 1
    assert price["selection_null"]["computed_on"].startswith("discovery")


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


def test_repeat_run_never_overwrites_a_finished_run(refused):
    first = run_demo(refused, **FAST)
    first_dir = refused / "runs" / first["run_id"]
    before = {p.name: p.read_bytes() for p in first_dir.iterdir()}

    second = run_demo(refused, **FAST)

    assert second["run_id"] == first["run_id"] + "-r2"
    assert {p.name: p.read_bytes() for p in first_dir.iterdir()} == before
    second_dir = refused / "runs" / second["run_id"]
    events = read_jsonl(second_dir / "events.jsonl")
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    for run_id in (first["run_id"], second["run_id"]):
        assert cli_main(["--root", str(refused), "verify", "--run", run_id]) == 0


def test_interrupted_run_dir_is_reused_without_touching_unknown_files(refused):
    from autoresearch.contract import load_contract

    info = read_json(refused / "data" / "demo_info.json")
    contract = load_contract(info["contract_path"], info["discovery_path"])
    run_dir = refused / "runs" / f"ar-{contract.hash[:8]}-12345"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(json.dumps({"seq": 1, "stage": "stale"}) + "\n", encoding="utf-8")
    (run_dir / "user_notes.txt").write_text("keep me", encoding="utf-8")

    result = run_demo(refused, **FAST)

    assert result["run_id"] == run_dir.name
    assert read_jsonl(run_dir / "events.jsonl")[0]["stage"] != "stale"
    assert (run_dir / "user_notes.txt").read_text(encoding="utf-8") == "keep me"


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


WALL_CLOCK_LOGS = {"run.json", "events.jsonl"}


def test_two_processes_produce_byte_identical_artifacts(tmp_path):
    """Spec §6 acceptance test 4: two runs, two processes, byte-identical artifacts.

    Each run is a fresh interpreter with a different PYTHONHASHSEED. Only the
    two logs that carry wall-clock time (run.json, events.jsonl) may differ, and
    the event log must match once timestamps are removed.
    """
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from autoresearch.certificate import load_or_create_key

    src = str(Path(__file__).resolve().parents[1] / "src")
    roots = [tmp_path / "a", tmp_path / "b"]
    for r in roots:
        write_demo_files(r, seed=11, reset=True, n_patients=200, n_confirmation_groups=160)
    load_or_create_key(roots[0] / "state" / "certificate.key")
    shutil.copy2(roots[0] / "state" / "certificate.key", roots[1] / "state" / "certificate.key")

    code = (
        "import sys\n"
        "from autoresearch.demo import run_demo\n"
        "print(run_demo(sys.argv[1], n_permutations=9, candidate_budget=12, generations=1)['run_id'])\n"
    )
    run_ids = []
    for r, hashseed in zip(roots, ("1", "98765")):
        env = dict(os.environ, PYTHONPATH=src, PYTHONHASHSEED=hashseed)
        out = subprocess.run([sys.executable, "-c", code, str(r)], env=env, capture_output=True, text=True, check=True)
        run_ids.append(out.stdout.strip())
    assert run_ids[0] == run_ids[1]

    da, db = (r / "runs" / run_ids[0] for r in roots)
    names = sorted(p.name for p in da.iterdir())
    assert names == sorted(p.name for p in db.iterdir())
    assert "certificate.json" in names
    for name in names:
        if name in WALL_CLOCK_LOGS:
            continue
        assert (da / name).read_bytes() == (db / name).read_bytes(), name

    def strip(events):
        return [{k: v for k, v in e.items() if k != "ts"} for e in events]

    assert strip(read_jsonl(da / "events.jsonl")) == strip(read_jsonl(db / "events.jsonl"))


def test_procedure_registered_before_search_and_bound_to_certificate(issued):
    """Spec §8: register the procedure and selection rule; the winner is its consequence."""
    root, result = issued
    entries = Registry(root / "state").entries()
    kinds = [e["type"] for e in entries]
    assert kinds.index("procedure_registered") < kinds.index("prediction_committed")
    proc = next(e for e in entries if e["type"] == "procedure_registered")
    assert proc["authored_by"] == "engine"
    plan = read_json(root / "runs" / result["run_id"] / "plan.json")
    from autoresearch.utils import sha256_obj

    assert sha256_obj(plan) == proc["plan_hash"]
    commitments = result["certificate"]["commitments"]
    assert commitments["procedure_hash"] == proc["plan_hash"]
    assert commitments["procedure_registered_before_search"] is True


def test_certificate_prints_interpretability_bill_expiry_and_operating_point(issued):
    """Spec §10 / §12.4 / §12.5."""
    _root, result = issued
    cert = result["certificate"]
    ic = cert["interpretability_cost"]
    assert ic["gap_vs_unconstrained"] == pytest.approx(ic["unconstrained_cv_delta"] - ic["rule_cv_delta"])
    assert cert["expires"] and cert["expires"] > "2026-01-01"
    assert cert["operating_point"]["alpha"] == 0.05
    assert "chosen_by" in cert["operating_point"]


def test_private_refusals_and_operator_log_through_the_cli(refused, capsys):
    """Spec §12.2 and §12.6 at the CLI."""
    result = run_demo(refused, **FAST)
    assert cli_main(["--root", str(refused), "certificates"]) == 0
    listing = capsys.readouterr().out
    assert "[private refusal:" in listing
    assert result["certificate"]["claim"][:40] not in listing

    assert cli_main(["--root", str(refused), "publish", "--run", result["run_id"], "--by", "tester"]) == 0
    capsys.readouterr()
    assert cli_main(["--root", str(refused), "certificates"]) == 0
    assert result["certificate"]["claim"][:40] in capsys.readouterr().out

    decisions = Registry(refused / "state").entries("operator_decision")
    assert [d["command"] for d in decisions] == ["publish"]
    assert decisions[0]["detail"]["by"] == "tester"


def test_selected_candidate_carries_its_proposal_tier(issued):
    """Spec §12.3: operator-family candidates cannot be recall."""
    _root, result = issued
    prov = result["certificate"]["evidence"]["proposal_provenance"]
    assert prov["tier"] == "operator"
    assert prov["recall_risk"] is False


def test_demo_substrate_leaves_no_plaintext_sealed_rows(refused):
    """Spec §2: after sealing, the confirmation rows exist only inside the vault."""
    data_files = {p.name for p in (refused / "data").iterdir()}
    assert data_files == {"immunotherapy_discovery.csv", "demo_info.json"}
