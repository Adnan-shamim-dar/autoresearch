"""Shared fixtures and import path for the acceptance tests.

The package lives under ``src/`` and is not installed in the test environment, so
the suite puts it on ``sys.path`` itself. That keeps ``python -m pytest`` working
from a clean checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.contract import Contract  # noqa: E402
from autoresearch.demo import generate_immunotherapy_data, make_contract_dict  # noqa: E402


@pytest.fixture(scope="session")
def small_frame():
    """A small but structurally faithful copy of the immunotherapy fixture."""
    return generate_immunotherapy_data(seed=7, n_patients=24)


@pytest.fixture()
def contract(small_frame):
    raw = make_contract_dict(
        effect=0.04,
        k_candidates=16,
        confirmation_n=len(small_frame),
        confirmation_groups=int(small_frame["patient_id"].nunique()),
    )
    c = Contract(raw)
    c.validate_static()
    report = c.attach_data(small_frame)
    assert report.ok, report.errors
    return c
