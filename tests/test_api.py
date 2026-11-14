"""Spec §11: verify(contract, data, candidate) -> certificate."""

from __future__ import annotations

import pytest

from autoresearch.api import verify
from autoresearch.certificate import verify_certificate
from autoresearch.demo import write_demo_files
from autoresearch.utils import read_json

pytestmark = pytest.mark.integration

GOOD = {
    "weights": {"expression": 1.0, "stability": 0.8, "foreignness": 0.75, "binding_rank": 0.6, "presentation": 0.5},
    "mechanism": "Presented, expressed, stable and foreign peptides are the ones T cells see.",
}
USELESS = {"weights": {"publicness": 1.0}, "mechanism": "Publicness alone, a weak signal (expected to fail against the baseline)."}


@pytest.fixture()
def substrate(tmp_path):
    """A fresh sealed dataset per test: the demo contract allows one run per dataset lifetime."""
    info = write_demo_files(tmp_path, seed=11, reset=True, n_patients=200, n_confirmation_groups=160)
    return tmp_path, info


def _call(substrate, candidate, seed):
    root, info = substrate
    return verify(info["contract_path"], info["discovery_path"], candidate, root=root, seed=seed, n_permutations=19)


def test_good_candidate_gets_a_signed_affirmative_certificate(substrate):
    root, _ = substrate
    cert = _call(substrate, GOOD, seed=1)
    assert cert["verdict"] == "affirmed", cert["claim"]
    assert cert["search_price"]["effective_independent"] == 1
    assert cert["search_price"]["sealed_queries"] == 1
    assert cert["evidence"]["candidate"]["family"] == "prespecified"
    key = (root / "state" / "certificate.key").read_bytes()
    assert verify_certificate(cert, key)["valid"]
    plan = read_json(root / "runs" / cert["run_id"] / "plan.json")
    assert plan["search"]["proposers"] == ["prespecified"]


def test_useless_candidate_is_not_affirmed(substrate):
    cert = _call(substrate, USELESS, seed=2)
    assert cert["status"] == "issued"  # tested and not affirmed, not refused for another reason
    assert cert["verdict"] != "affirmed"


def test_second_verification_on_the_same_dataset_spends_no_extra_alpha(substrate):
    """Spec §8: alpha is conserved per dataset lifetime, across sessions and callers."""
    _call(substrate, GOOD, seed=1)
    cert = _call(substrate, USELESS, seed=2)
    assert cert["status"] == "refused"
    assert cert["refusals"][0]["reason_code"] == "alpha_exhausted"


def test_unknown_feature_is_refused_with_a_named_reason(substrate):
    from autoresearch.contract import ContractError

    with pytest.raises(ContractError) as exc:
        _call(substrate, {"weights": {"no_such_feature": 1.0}}, seed=3)
    assert exc.value.reason_code == "invalid_candidate"
    # Nothing was charged: a valid candidate can still use the dataset's alpha.
    assert _call(substrate, GOOD, seed=4)["status"] == "issued"
