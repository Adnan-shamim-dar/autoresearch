"""Milestone 8: a second domain on real data, no groups, signs pre-committed."""

from __future__ import annotations

import pytest
import yaml

from autoresearch import wdbc
from autoresearch.api import verify
from autoresearch.certificate import verify_certificate
from autoresearch.demo import run_demo
from autoresearch.utils import atomic_write_json
from autoresearch.vault import is_os_sealed

pytestmark = pytest.mark.integration


@pytest.fixture()
def wdbc_root(tmp_path):
    info = wdbc.write_files(tmp_path)
    atomic_write_json(tmp_path / "data" / "demo_info.json", info)
    return tmp_path, info


def test_plaintext_confirmation_is_not_left_readable(wdbc_root):
    root, info = wdbc_root
    assert not (root / "data" / "wdbc_confirmation.csv").exists()
    blob = root / "vault" / info["confirmation_meta"]["dataset_hash"] / "sealed.enc"
    assert blob.exists()
    if is_os_sealed(blob):
        with pytest.raises(PermissionError):
            blob.read_bytes()


def test_underpowered_magnitude_is_refused_and_signs_are_tested(wdbc_root):
    """Real data: the 0.02 question needs thousands of tumours; signs are a powered test."""
    root, _info = wdbc_root
    result = run_demo(root, n_permutations=19, candidate_budget=60, generations=1)
    cert = result["certificate"]
    assert cert["status"] == "refused"
    assert cert["refusals"][0]["reason_code"] == "not_answerable"
    signs = cert["evidence"]["sign_commitments"]
    assert signs["n_signs"] == 10
    assert signs["agree"] >= 8 and signs["affirmed"] is True
    assert "Direction aggregation" in cert["claim"]
    assert verify_certificate(cert, (root / "state" / "certificate.key").read_bytes())["valid"]


def test_full_pipeline_runs_without_groups(wdbc_root, tmp_path):
    """Folds, bootstrap and stability must work when units are independent."""
    root, info = wdbc_root
    contract = yaml.safe_load(open(info["contract_path"], encoding="utf-8"))
    contract["decision"]["smallest_effect_that_matters"] = 0.08  # test-only: answerable at K=1
    contract["commitments"] = {"signs": False}
    path = tmp_path / "answerable.yaml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    candidate = {"weights": {"mean_concave_points": 1.0, "mean_texture": 0.6, "mean_area": 0.6}}
    cert = verify(path, info["discovery_path"], candidate, root=root, seed=7, n_permutations=19)
    assert cert["status"] == "issued"
    ev = cert["evidence"]
    assert ev["n_groups_confirmation"] == info["n_confirmation"]
    assert ev["ci_low"] is not None and ev["ci_low"] < ev["delta"] < ev["ci_high"]
    assert 0.0 <= ev["confirmation_stability"] <= 1.0
    assert ev["delta"] > 0  # the rule does beat size alone on real data
