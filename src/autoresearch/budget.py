from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .contract import Contract
from .scoring import Candidate, candidate_complexity


@dataclass
class ComplexityBudget:
    n_rows: int
    n_groups: int
    n_positive: int
    n_negative: int
    effective_pairs: float
    design_effect: float
    effective_n: float
    max_terms: int
    max_features: int
    max_candidates: int
    max_generations: int
    max_gates: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def check(self, candidate: Candidate) -> tuple[bool, str]:
        if candidate.complexity > self.max_terms:
            return False, "rejected_complexity"
        weights = candidate.params.get("weights", {}) or {}
        if len(weights) > self.max_features:
            return False, "rejected_too_many_features"
        gates = candidate.params.get("gates", []) or []
        if len(gates) > self.max_gates:
            return False, "rejected_too_many_gates"
        if not weights and "column" not in candidate.params:
            return False, "rejected_empty"
        return True, "ok"


def compute_budget(contract: Contract, data) -> ComplexityBudget:
    y = data[contract.outcome_column].to_numpy()
    n = len(data)
    groups = data[contract.group_column].astype(str).to_numpy() if contract.group_column else np.array(["__all__"] * n)
    n_groups = int(len(set(groups)))
    pos = int(y.sum())
    neg = int(n - pos)
    effective_pairs = (pos * neg / (pos + neg)) if (pos + neg) else 0.0

    sizes = []
    for g in set(groups):
        sizes.append(int(np.sum(groups == g)))
    mean_size = float(np.mean(sizes)) if sizes else 1.0
    # A conservative design effect: clustered observations buy less than their
    # nominal count. 0.05 is a floor, not a fitted intraclass correlation.
    design_effect = max(1.0, 1.0 + 0.05 * max(0.0, mean_size - 1.0)) if contract.group_column else 1.0
    effective_n = max(4.0, effective_pairs / design_effect)

    max_terms = max(1, min(len(contract.features), int(math.floor(math.log2(max(8.0, effective_n))))))
    max_features = max(1, min(len(contract.features), max_terms))
    max_candidates = int(np.clip(20.0 * effective_n / (max_terms + 1.0), 24, 4000))
    max_generations = 3 if effective_n < 120 else 4 if effective_n < 400 else 5
    max_gates = 1 if effective_n < 250 else 2
    rationale = (
        f"effective_pairs={effective_pairs:.1f}, design_effect={design_effect:.2f}, "
        f"effective_n={effective_n:.1f}; capacity grows logarithmically, not with the search budget"
    )
    return ComplexityBudget(
        n_rows=n,
        n_groups=n_groups,
        n_positive=pos,
        n_negative=neg,
        effective_pairs=effective_pairs,
        design_effect=design_effect,
        effective_n=effective_n,
        max_terms=max_terms,
        max_features=max_features,
        max_candidates=max_candidates,
        max_generations=max_generations,
        max_gates=max_gates,
        rationale=rationale,
    )
