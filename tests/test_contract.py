from __future__ import annotations

import copy

import pandas as pd
import pytest
import yaml

from autoresearch.contract import Contract, ContractError, load_contract
from autoresearch.demo import make_contract_dict


def _raw(**overrides):
    raw = copy.deepcopy(make_contract_dict(effect=0.04, k_candidates=16))
    raw.update(overrides)
    return raw


def test_missing_smallest_effect_is_rejected():
    raw = _raw()
    raw["decision"].pop("smallest_effect_that_matters")
    with pytest.raises(ContractError) as exc:
        Contract(raw).validate_static()
    assert exc.value.reason_code == "missing_smallest_effect"


def test_unknown_transform_is_rejected_with_named_reason():
    raw = _raw()
    raw["candidates"]["features"][0]["transform"] = "magic"
    with pytest.raises(ContractError) as exc:
        Contract(raw).validate_static()
    assert exc.value.reason_code == "invalid_transform"


def test_wrong_group_column_fails_with_named_reason(small_frame, contract):
    raw = copy.deepcopy(contract.raw)
    raw["group"] = "cohort_that_does_not_exist"
    bad = Contract(raw)
    bad.validate_static()
    with pytest.raises(ContractError) as exc:
        bad.attach_data(small_frame)
    assert exc.value.reason_code == "missing_group_column"
    assert bad.data is None


def test_duplicate_units_are_rejected(small_frame, contract):
    df = small_frame.copy()
    df.loc[df.index[1], "neoepitope_id"] = df.loc[df.index[0], "neoepitope_id"]
    with pytest.raises(ContractError) as exc:
        contract.validate_data(df)
    assert exc.value.reason_code == "duplicate_units"


def test_outcome_cannot_be_a_feature(small_frame, contract):
    raw = copy.deepcopy(contract.raw)
    raw["candidates"]["features"].append(
        {"name": "immunogenic", "direction": "higher_better", "transform": "rank"}
    )
    bad = Contract(raw)
    bad.validate_static()
    with pytest.raises(ContractError) as exc:
        bad.attach_data(small_frame)
    assert exc.value.reason_code == "outcome_as_feature"


def test_contract_hash_changes_when_decision_changes():
    a = Contract(_raw())
    b = Contract(_raw())
    assert a.hash == b.hash
    b.raw["decision"]["smallest_effect_that_matters"] = 0.05
    assert Contract(b.raw).hash != a.hash


def test_yaml_roundtrip_preserves_hash(tmp_path, small_frame):
    contract = Contract(_raw())
    contract.validate_static()
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(contract.raw, sort_keys=True), encoding="utf-8")
    data_path = tmp_path / "data.csv"
    small_frame.to_csv(data_path, index=False)

    loaded = load_contract(path, data_path)
    assert loaded.hash == contract.hash
    assert loaded.validation is not None and loaded.validation.ok
    assert loaded.data is not None and len(loaded.data) == len(small_frame)


def test_confirmation_schema_check_reports_missing_columns(contract, small_frame):
    meta = {
        "n_rows": len(small_frame),
        "n_groups": int(small_frame["patient_id"].nunique()),
        "columns": [c for c in small_frame.columns if c != "expression"],
    }
    with pytest.raises(ContractError) as exc:
        contract.attach_confirmation_meta(meta)
    assert exc.value.reason_code == "confirmation_schema_mismatch"
