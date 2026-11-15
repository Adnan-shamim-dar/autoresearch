"""Bring-your-own-CSV domain: contract, split and sealing from one command."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoresearch import custom
from autoresearch.cli import main as cli_main
from autoresearch.contract import ContractError


def _csv(tmp_path, n=600, groups=False):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"score": rng.normal(size=n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["label"] = np.where(df["score"] + df["a"] + rng.normal(size=n) > 0, "good", "bad")
    df["note"] = "text"
    if groups:
        df["site"] = [f"s{i % 30}" for i in range(n)]
    path = tmp_path / "mydata.csv"
    df.to_csv(path, index=False)
    return path


def test_custom_domain_builds_contract_and_seals(tmp_path):
    info = custom.write_files(tmp_path / "root", _csv(tmp_path), "label", "score", positive="good")
    assert info["n_discovery"] + info["n_confirmation"] == 600
    assert info["n_features"] == 3  # numeric columns only; text column ignored
    data_files = {p.name for p in (tmp_path / "root" / "data").iterdir()}
    assert "custom_confirmation.csv" not in data_files  # sealed rows are not left in plaintext


def test_group_split_keeps_groups_whole(tmp_path):
    info = custom.write_files(tmp_path / "root", _csv(tmp_path, groups=True), "label", "score", positive="good", group="site")
    disc = pd.read_csv(info["discovery_path"])
    assert disc["site"].nunique() < 30  # some sites are entirely sealed
    assert info["n_discovery"] == len(disc)


def test_non_binary_outcome_needs_positive(tmp_path):
    with pytest.raises(ContractError) as exc:
        custom.write_files(tmp_path / "root", _csv(tmp_path), "label", "score")
    assert exc.value.reason_code == "outcome_not_binary"


def test_missing_column_is_named(tmp_path):
    with pytest.raises(ContractError) as exc:
        custom.write_files(tmp_path / "root", _csv(tmp_path), "label", "nope", positive="good")
    assert exc.value.reason_code == "missing_column"


@pytest.mark.integration
def test_cli_init_custom_then_run_issues_a_verifiable_certificate(tmp_path):
    root = tmp_path / "root"
    assert cli_main(["--root", str(root), "init", "--domain", "custom", "--data", str(_csv(tmp_path, groups=True)),
                     "--outcome", "label", "--positive", "good", "--baseline", "score", "--group", "site",
                     "--if-underpowered", "proceed", "--by", "tester"]) == 0
    from autoresearch.demo import run_demo

    result = run_demo(root, n_permutations=9, candidate_budget=12, generations=1)
    assert result["certificate"]["operating_point"]["if_underpowered"] == "proceed"
    assert cli_main(["--root", str(root), "verify", "--run", result["run_id"]]) == 0
