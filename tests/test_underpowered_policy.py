"""decision.if_underpowered: refuse (default) or proceed, always attributed."""

from __future__ import annotations

import copy

import pytest

from autoresearch.calibration_real import contract_dict
from autoresearch.contract import Contract, ContractError


def _raw(**decision):
    raw = copy.deepcopy(contract_dict(confirmation_n=228, k_candidates=32))
    raw["decision"].update(decision)
    return raw


def test_default_is_refuse():
    c = Contract(_raw())
    c.validate_static()
    assert c.decision.if_underpowered == "refuse"


def test_proceed_must_be_attributed():
    with pytest.raises(ContractError) as exc:
        Contract(_raw(if_underpowered="proceed")).validate_static()
    assert exc.value.reason_code == "if_underpowered_unattributed"


def test_unknown_policy_is_rejected():
    with pytest.raises(ContractError) as exc:
        Contract(_raw(if_underpowered="maybe")).validate_static()
    assert exc.value.reason_code == "invalid_if_underpowered"


def test_proceed_is_recorded():
    c = Contract(_raw(if_underpowered="proceed", if_underpowered_chosen_by="tester"))
    c.validate_static()
    assert (c.decision.if_underpowered, c.decision.if_underpowered_chosen_by) == ("proceed", "tester")


@pytest.mark.integration
def test_underpowered_question_runs_under_proceed_and_is_refused_by_default():
    from autoresearch.calibration import evaluate_replicate
    from autoresearch.calibration_real import plant, split
    from autoresearch.wdbc import load_frame

    full, realised = plant(load_frame(), 0.0, seed=5)
    disc, conf = split(full, seed=5)
    payload = {"seed": 5, "target_effect": 0.0, "candidates": 16, "permutations": 19}
    refused = evaluate_replicate(payload, _raw(smallest_effect_that_matters=0.01), disc, conf, None, realised)
    assert refused["tier"] == "power_refusal"
    ran = evaluate_replicate(
        payload,
        _raw(smallest_effect_that_matters=0.01, if_underpowered="proceed", if_underpowered_chosen_by="tester"),
        disc, conf, None, realised,
    )
    assert ran["power_verdict"] == "NOT_ANSWERABLE" and ran["tier"] != "power_refusal"
    assert not ran["detected_affirmed"]  # a null task is still not promoted
