"""Replication back-test mechanics (pre-registered rules), on synthetic rows."""

from __future__ import annotations

import pandas as pd
import pytest

from autoresearch.backtest import (
    INPUT_COLS,
    PRIMARY,
    SECONDARY,
    decide,
    include,
    required_n,
    run_backtest,
    to_markdown,
    wilson,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "T.N.O": [40, 80, 120, 30, 60, 50, 25],
            "T.pval.USE.O": [0.0001, 0.001, 0.004, 0.03, 0.04, 0.2, None],
            "T.r.O": [0.6, 0.4, 0.3, 0.35, 0.25, 0.1, 0.3],
            "T.pval.USE.R": [0.001, 0.01, 0.2, 0.5, 0.01, 0.01, 0.01],
            "Direction.R": ["same", "same", "same", "same", "opposite", "same", "same"],
        }
    )


def test_inclusion_drops_missing_and_non_significant_originals():
    kept = include(_frame())
    assert len(kept) == 5
    assert (kept["T.pval.USE.O"] < 0.05).all()


def test_decide_refuses_to_see_outcome_columns():
    kept = include(_frame())
    with pytest.raises(ValueError, match="non-input"):
        decide(kept, PRIMARY)


def test_primary_rule_is_search_priced_threshold():
    kept = include(_frame())
    d = decide(kept[list(INPUT_COLS)], PRIMARY)
    assert d["accept"].tolist() == [True, True, True, False, False]
    assert d.loc[~d["accept"], "prescribed_n"].notna().all()


def test_secondary_rule_is_stricter():
    kept = include(_frame())
    d = decide(kept[list(INPUT_COLS)], SECONDARY)
    assert SECONDARY.z_threshold == pytest.approx(3.649, abs=1e-3)
    assert d["accept"].tolist() == [True, False, False, False, False]


def test_required_n_matches_fisher_z_formula():
    # r = 0.3, alpha 0.005, 80% power: ((2.807 + 0.842) / atanh(0.3))^2 + 3 = 141.99
    assert required_n(0.3, 0.005) == 142
    assert required_n(0.0, 0.005) is None


def test_wilson_interval_contains_estimate():
    lo, hi = wilson(3, 10)
    assert lo < 0.3 < hi
    assert wilson(0, 0) == (None, None)


def test_run_backtest_counts_and_headline():
    result = run_backtest(_frame())
    p = result["primary"]
    assert (p["accepted"], p["refused"]) == (3, 2)
    assert (p["accepted_replicated"], p["refused_replicated"]) == (2, 0)
    assert len(result["sensitivity"]) == 5
    md = to_markdown(result)
    assert "accepted **3** and refused **2**" in md
