from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import metrics
from .budget import ComplexityBudget
from .contract import Contract
from .scoring import Candidate, FeatureMatrix, make_candidate
from .utils import normalize_weights, rng_for


FEATURE_SYNONYMS = {
    "binding": ["binding", "affinity", "netmhc", "hla", "presentation"],
    "stability": ["stability", "stab", "half"],
    "expression": ["expression", "tpm", "abundance", "rna", "transcript"],
    "foreignness": ["foreign", "non_self", "nonself", "novel", "mutant"],
    "hydrophobicity": ["hydrophobic", "gravy", "hydro"],
    "self_similarity": ["self", "similar", "tolerance", "auto"],
    "publicness": ["public", "clonal", "frequency", "prevalence"],
    "processing": ["processing", "cleavage", "proteasom", "tap"],
}


# ---------------------------------------------------------------------------
# The feedback channel (spec §5)
# ---------------------------------------------------------------------------
# Feedback to any proposer is {kept | rejected, reason_code} and nothing else:
# never a score, p-value or margin. A proposer given numbers optimises against
# the gate, which is p-hacking performed by a machine at scale. Every feedback
# message the engine sends passes through sanitize_feedback; the leak test in
# feedback_leak.py measures what happens when it does not.

FEEDBACK_STATUSES = frozenset({"kept", "rejected"})
_REASON_CODE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz_")


def sanitize_feedback(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce feedback to ``{id, status, reason_code}``.

    Any other field is dropped. A reason code that is not a plain lowercase
    identifier is replaced, because free text or digits can smuggle a number.
    """
    out: list[dict[str, Any]] = []
    for item in items:
        status = item.get("status")
        status = status if status in FEEDBACK_STATUSES else "rejected"
        code = str(item.get("reason_code") or status)
        if not code or not set(code) <= _REASON_CODE_CHARS:
            code = f"{status}_unspecified"
        out.append({"id": str(item.get("id", "")), "status": status, "reason_code": code})
    return out


def _match_feature(contract: Contract, keywords: list[str]) -> str | None:
    for f in contract.features:
        text = (f.name + " " + f.description).lower()
        if any(k in text for k in keywords):
            return f.name
    return None


def _mechanism_for(weights: dict[str, float], family: str, contract: Contract) -> str:
    parts = []
    for name, w in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        f = next((x for x in contract.features if x.name == name), None)
        desc = f.description if f and f.description else name
        direction = "enriches" if w >= 0 else "suppresses"
        parts.append(f"{desc} ({w:+.2f} {direction})")
    return f"Hypothesis [{family}]: a weighted within-group rank fusion of " + "; ".join(parts) + "."


@dataclass
class Proposer:
    contract: Contract
    data: Any
    fm: FeatureMatrix
    budget: ComplexityBudget
    seed: int = 0
    domain_brief: str = ""
    llm: Callable[[str], str] | None = None
    _seen: set[str] = field(default_factory=set, init=False)
    _counter: int = field(default=0, init=False)
    _family_stats: dict[str, dict[str, int]] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------
    def _fit(self, weights: dict[str, float]) -> dict[str, Any]:
        return self.fm.export_fit_stats(list(weights))

    def _candidate(
        self,
        weights: dict[str, float],
        family: str,
        mechanism: str | None = None,
        transforms: dict[str, str] | None = None,
        gates: list[dict[str, Any]] | None = None,
        parents: list[str] | None = None,
        generation: int = 0,
        source: str = "operator",
    ) -> Candidate | None:
        weights = {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-9}
        if not weights:
            return None
        # Respect the complexity cap before the evaluator ever sees the candidate.
        if len(weights) > self.budget.max_features:
            items = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[: self.budget.max_features]
            weights = dict(items)
        self._counter += 1
        cid = f"c{generation:02d}-{self._counter:04d}"
        c = make_candidate(
            weights=weights,
            mechanism=mechanism or _mechanism_for(weights, family, self.contract),
            family=family,
            cid=cid,
            name=f"{family}-{self._counter:04d}",
            transforms=transforms,
            gates=gates,
            fit_stats=self._fit(weights),
            parents=parents,
            generation=generation,
            source=source,
        )
        sig = json.dumps({"w": normalize_weights(weights), "t": transforms or {}, "g": gates or []}, sort_keys=True)
        if sig in self._seen:
            return None
        self._seen.add(sig)
        return c

    # ------------------------------------------------------------------
    def seed_candidates(self, n: int = 48) -> list[Candidate]:
        y = self.data[self.contract.outcome_column].to_numpy().astype(int)
        out: list[Candidate] = []

        # 1. Baseline clone / nearest feature.
        base = self.contract.baseline
        if base.weights:
            c = self._candidate(
                base.weights,
                family="baseline_clone",
                mechanism=f"Clone of the deployed baseline ({base.name}); the search must beat this, not chance.",
                transforms=base.transforms or None,
                generation=0,
            )
            if c:
                out.append(c)
        else:
            # Pick the feature with the strongest univariate relationship to the baseline score.
            base_scores = self.fm.score_candidate(self.contract.baseline_candidate())
            best_f, best_r = None, 0.0
            for f in self.contract.features:
                try:
                    r = abs(float(np.corrcoef(self.fm.array(f), base_scores)[0, 1]))
                except Exception:
                    r = 0.0
                if r > best_r:
                    best_f, best_r = f.name, r
            if best_f:
                c = self._candidate(
                    {best_f: 1.0},
                    family="baseline_clone",
                    mechanism=f"Single-feature proxy for the deployed baseline {base.name} ({best_f}).",
                    generation=0,
                )
                if c:
                    out.append(c)

        # 2. Singles: every feature gets a fair shot.
        for f in self.contract.features:
            if f.name in self.data.columns:
                c = self._candidate({f.name: 1.0}, family="single", generation=0)
                if c:
                    out.append(c)

        # 3. Domain templates.
        binding = _match_feature(self.contract, FEATURE_SYNONYMS["binding"])
        stability = _match_feature(self.contract, FEATURE_SYNONYMS["stability"])
        expression = _match_feature(self.contract, FEATURE_SYNONYMS["expression"])
        foreign = _match_feature(self.contract, FEATURE_SYNONYMS["foreignness"])
        hydro = _match_feature(self.contract, FEATURE_SYNONYMS["hydrophobicity"])
        selfsim = _match_feature(self.contract, FEATURE_SYNONYMS["self_similarity"])
        public = _match_feature(self.contract, FEATURE_SYNONYMS["publicness"])
        processing = _match_feature(self.contract, FEATURE_SYNONYMS["processing"])

        templates = [
            ("binding+expression", {binding: 0.65, expression: 0.35}),
            ("binding+stability", {binding: 0.6, stability: 0.4}),
            ("binding+foreignness", {binding: 0.65, foreign: 0.35}),
            ("binding+expression+foreignness", {binding: 0.5, expression: 0.3, foreign: 0.2}),
            ("binding+stability+expression", {binding: 0.45, stability: 0.25, expression: 0.25, foreign: 0.05}),
            ("presentation+processing", {binding: 0.45, processing: 0.3, expression: 0.25}),
            ("publicness+foreignness", {public: 0.5, foreign: 0.5}),
            ("hydrophobicity+self_similarity", {hydro: 0.5, selfsim: -0.5}),
        ]
        for name, w in templates:
            w = {k: v for k, v in w.items() if k}
            if len(w) >= 2:
                c = self._candidate(w, family="domain_template", generation=0)
                if c:
                    out.append(c)

        # 4. Importance-derived: univariate AUCs on discovery. This is a proposal
        #    mechanism, not a confirmation test; confirmation remains sealed.
        weights: dict[str, float] = {}
        for f in self.contract.features:
            if f.name not in self.data.columns:
                continue
            a = metrics.auc(y, self.fm.array(f))
            if not np.isnan(a):
                w = 2.0 * (a - 0.5)
                if abs(w) > 0.01:
                    weights[f.name] = w
        if weights:
            top = dict(sorted(weights.items(), key=lambda kv: -abs(kv[1]))[: self.budget.max_terms])
            out.append(
                self._candidate(
                    top,
                    family="importance",
                    mechanism="Univariate importance rule: rank-fuse the features whose individual AUCs are farthest from chance.",
                    generation=0,
                )
            )

        # 5. Random sparse blends.
        rng = rng_for("proposer-seed", self.contract.hash, self.seed)
        feature_names = [f.name for f in self.contract.features if f.name in self.data.columns]
        attempts = 0
        while len(out) < n and feature_names and attempts < n * 4:
            attempts += 1
            k = int(rng.integers(2, min(len(feature_names), self.budget.max_terms) + 1))
            chosen = list(rng.choice(feature_names, size=k, replace=False))
            w = rng.dirichlet(np.ones(k))
            if rng.random() < 0.2 and k > 1:
                w[0] *= -1
            weights = {name: float(val) for name, val in zip(chosen, w)}
            transforms = {name: str(rng.choice(["rank", "log_rank", "z"])) for name in chosen}
            c = self._candidate(weights, family="random_sparse", transforms=transforms, generation=0)
            if c:
                out.append(c)
        return out[:n]

    # ------------------------------------------------------------------
    def next_generation(
        self,
        kept: list[Candidate],
        feedback: list[dict[str, Any]],
        n: int = 40,
        generation: int = 1,
    ) -> list[Candidate]:
        if not kept:
            return self.seed_candidates(n)
        rng = rng_for("proposer-gen", self.contract.hash, self.seed, generation)
        self._update_family_stats(feedback)
        out: list[Candidate] = []

        # High-keep families get more children; low-keep families are not starved.
        families = ["mutation", "crossover", "random_sparse", "gate", "domain_template"]
        family_weights = []
        for fam in families:
            st = self._family_stats.get(fam, {"kept": 0, "rejected": 0})
            total = st["kept"] + st["rejected"]
            rate = (st["kept"] + 1) / (total + 2) if total else 0.5
            family_weights.append(0.4 + rate)
        family_weights = np.asarray(family_weights, dtype=float)
        family_weights /= family_weights.sum()
        counts = rng.multinomial(n, family_weights)
        feature_names = [f.name for f in self.contract.features if f.name in self.data.columns]

        for fam, count in zip(families, counts):
            for _ in range(int(count)):
                if fam == "mutation":
                    parent = kept[int(rng.integers(0, len(kept)))]
                    c = self._mutate(parent, generation)
                elif fam == "crossover":
                    if len(kept) >= 2:
                        a, b = rng.choice(len(kept), size=2, replace=False)
                        c = self._crossover(kept[int(a)], kept[int(b)], generation)
                    else:
                        c = self._mutate(kept[0], generation)
                elif fam == "random_sparse":
                    k = int(rng.integers(2, min(len(feature_names), self.budget.max_terms) + 1)) if feature_names else 0
                    if k:
                        chosen = list(rng.choice(feature_names, size=k, replace=False))
                        w = rng.dirichlet(np.ones(k))
                        c = self._candidate(
                            {name: float(val) for name, val in zip(chosen, w)},
                            family="random_sparse",
                            transforms={name: str(rng.choice(["rank", "log_rank", "z"])) for name in chosen},
                            generation=generation,
                        )
                    else:
                        c = None
                elif fam == "gate":
                    parent = kept[int(rng.integers(0, len(kept)))]
                    c = self._gate(parent, generation, rng)
                else:
                    parent = kept[int(rng.integers(0, len(kept)))]
                    c = self._mutate(parent, generation, family="domain_template")
                if c and c.id not in {x.id for x in out}:
                    out.append(c)
        # Fill if the stochastic families under-produced.
        attempts = 0
        while len(out) < n and feature_names and attempts < n * 4:
            attempts += 1
            k = int(rng.integers(2, min(len(feature_names), self.budget.max_terms) + 1))
            chosen = list(rng.choice(feature_names, size=k, replace=False))
            w = rng.dirichlet(np.ones(k))
            c = self._candidate({name: float(val) for name, val in zip(chosen, w)}, family="random_sparse", generation=generation)
            if c:
                out.append(c)
        return out[:n]

    def _update_family_stats(self, feedback: list[dict[str, Any]]) -> None:
        by_id = {c.id: c for c in getattr(self, "_last_candidates", [])}
        for f in feedback:
            c = by_id.get(f.get("id"))
            if not c:
                continue
            st = self._family_stats.setdefault(c.family, {"kept": 0, "rejected": 0})
            if f.get("status") == "kept":
                st["kept"] += 1
            else:
                st["rejected"] += 1

    def remember(self, candidates: list[Candidate]) -> None:
        self._last_candidates = candidates

    def _mutate(self, parent: Candidate, generation: int, family: str = "mutation") -> Candidate | None:
        rng = rng_for("mutation", parent.id, generation, self.seed)
        weights = dict(parent.params.get("weights", {}) or {})
        if not weights:
            return None
        out: dict[str, float] = {}
        for k, v in weights.items():
            if rng.random() < 0.15:
                continue
            out[k] = float(v) * float(np.exp(rng.normal(0, 0.45)))
        feature_names = [f.name for f in self.contract.features if f.name in self.data.columns]
        if feature_names and rng.random() < 0.5:
            add = str(rng.choice(feature_names))
            if add not in out:
                out[add] = float(rng.normal(0, 0.25))
        if len(out) > self.budget.max_features:
            out = dict(sorted(out.items(), key=lambda kv: -abs(kv[1]))[: self.budget.max_features])
        transforms = dict(parent.params.get("transforms", {}) or {})
        if transforms and rng.random() < 0.3:
            k = str(rng.choice(list(transforms.keys())))
            transforms[k] = str(rng.choice(["rank", "log_rank", "z"]))
        mechanism = (
            f"Hypothesis [mutation of {parent.name}]: perturb the weights of a kept mechanism; "
            "if the improvement disappears under a small perturbation, it was a knife-edge fit."
        )
        return self._candidate(out, family=family, mechanism=mechanism, transforms=transforms, parents=[parent.id], generation=generation)

    def _crossover(self, a: Candidate, b: Candidate, generation: int) -> Candidate | None:
        rng = rng_for("crossover", a.id, b.id, self.seed)
        wa = a.params.get("weights", {}) or {}
        wb = b.params.get("weights", {}) or {}
        keys = sorted(set(wa) | set(wb))
        out = {}
        for k in keys:
            if k in wa and k in wb:
                out[k] = float(wa[k]) * (1 - 0.5) + float(wb[k]) * 0.5
            else:
                out[k] = float(wa.get(k, wb.get(k, 0.0))) * 0.7
        if len(out) > self.budget.max_features:
            out = dict(sorted(out.items(), key=lambda kv: -abs(kv[1]))[: self.budget.max_features])
        mechanism = (
            f"Hypothesis [crossover {a.name} × {b.name}]: recombine two mechanistically distinct kept rules; "
            "the parts should be complementary, not merely correlated."
        )
        return self._candidate(out, family="crossover", mechanism=mechanism, parents=[a.id, b.id], generation=generation)

    def _gate(self, parent: Candidate, generation: int, rng: np.random.Generator) -> Candidate | None:
        if self.budget.max_gates < 1:
            return None
        weights = dict(parent.params.get("weights", {}) or {})
        feature_names = [f.name for f in self.contract.features if f.name in self.data.columns]
        if not feature_names:
            return None
        gate_feature = str(rng.choice(feature_names))
        # Gate at the median of this feature's within-group rank (0.5).
        gate = {"feature": gate_feature, "op": ">=", "threshold": 0.5, "weights": dict(weights)}
        mechanism = (
            f"Hypothesis [gate on {gate_feature}]: the parent rule should work mainly where {gate_feature} "
            "is high; a conditional rule is more interpretable than a larger weight vector."
        )
        return self._candidate(
            weights,
            family="gate",
            mechanism=mechanism,
            transforms=dict(parent.params.get("transforms", {}) or {}),
            gates=[gate],
            parents=[parent.id],
            generation=generation,
        )

    # ------------------------------------------------------------------
    def llm_candidates(self, n: int = 12, rejection_history: list[dict[str, Any]] | None = None) -> list[Candidate]:
        if not self.llm:
            return []
        feature_text = "\n".join(
            f"- {f.name} ({f.direction}, {f.transform}): {f.description}" for f in self.contract.features
        )
        prompt = f"""You are a scientific hypothesis proposer inside a certification engine.
You never see scores, p-values, or confirmation outcomes. You only see the contract,
the domain brief, and which mechanisms were kept or rejected (status + reason code).

CONTRACT
unit: {self.contract.unit}
group: {self.contract.group}
outcome: {self.contract.outcome}
baseline: {self.contract.baseline.name}
effect that matters: {self.contract.decision.smallest_effect_that_matters}
features:
{feature_text}

DOMAIN BRIEF
{self.domain_brief[:4000]}

REJECTION HISTORY (categorical only)
{json.dumps(rejection_history or [], indent=2)[:4000]}

Propose up to {n} mechanistically distinct candidates. Each must be a JSON object:
{{"weights": {{"feature": number, ...}}, "transforms": {{"feature": "rank|log_rank|z"}}, "mechanism": "one sentence"}}
Return ONLY a JSON array. Do not propose more than {self.budget.max_terms} non-zero weights per candidate.
"""
        try:
            text = self.llm(prompt)
            start = text.find("[")
            end = text.rfind("]")
            if start < 0 or end < 0:
                return []
            items = json.loads(text[start : end + 1])
        except Exception:
            return []
        out = []
        for item in items:
            if not isinstance(item, dict):
                continue
            weights = item.get("weights", {})
            if not isinstance(weights, dict) or not weights:
                continue
            c = self._candidate(
                {k: float(v) for k, v in weights.items() if k in {f.name for f in self.contract.features}},
                family="llm",
                mechanism=str(item.get("mechanism", "LLM proposal without stated mechanism")),
                transforms=item.get("transforms", {}),
                generation=99,
                source="llm",
            )
            if c:
                out.append(c)
        return out[:n]


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible chat client, configured entirely by env vars.

    AUTORESEARCH_LLM_URL   e.g. http://localhost:11434/v1/chat/completions
    AUTORESEARCH_LLM_KEY   optional bearer token
    AUTORESEARCH_LLM_MODEL e.g. a local or hosted model name
    """

    def __init__(self, url: str | None = None, key: str | None = None, model: str | None = None):
        self.url = url or os.environ.get("AUTORESEARCH_LLM_URL")
        self.key = key or os.environ.get("AUTORESEARCH_LLM_KEY", "")
        self.model = model or os.environ.get("AUTORESEARCH_LLM_MODEL", "")

    @property
    def available(self) -> bool:
        return bool(self.url and self.model)

    def __call__(self, prompt: str) -> str:
        if not self.available:
            raise RuntimeError("LLM proposer not configured")
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1200,
            }
        ).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
