from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import metrics
from .adversary import run_adversary
from .budget import compute_budget
from .certificate import build_certificate, load_or_create_key, save_certificate, sign_certificate
from .contract import Contract, ContractError, load_contract
from .evaluator import (
    SearchEvaluation,
    assign_folds,
    effective_independent,
    evaluate_candidates,
    leave_one_group_out_delta,
    permute_within_groups,
    select_on_permuted_outcomes,
    interpretability_bill,
    selection_aware_null,
)
from .gate import GateDecision, assign_tier
from .power import Assessment, assess
from .proposer import Proposer, sanitize_feedback
from .registry import Registry, RegistryError
from .scoring import Candidate, FeatureMatrix
from .utils import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    json_safe,
    now_iso,
    rng_for,
    sha256_file,
    sha256_obj,
)
from .vault import BudgetExhausted


CODE_FILES = [
    Path(__file__).with_name("scoring.py"),
    Path(__file__).with_name("metrics.py"),
    Path(__file__).with_name("evaluator.py"),
    Path(__file__).with_name("gate.py"),
    Path(__file__).with_name("power.py"),
]


@dataclass
class RunConfig:
    contract_path: str | Path
    discovery_path: str | Path
    run_dir: str | Path
    vault_root: str | Path
    registry_root: str | Path
    sealed: Any
    sealed_dataset_hash: str
    run_seed: int = 12345
    n_permutations: int = 99
    n_splits: int = 5
    candidate_budget: int | None = None
    generations: int | None = None
    llm: Callable[[str], str] | None = None
    domain_brief: str = ""
    calibration_table: list[dict[str, Any]] | None = None
    key_path: str | Path | None = None
    event_sink: Callable[[dict[str, Any]], None] | None = None
    mode: str = "separate_process_label_service"
    limitations: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    certificate: dict[str, Any]
    power: Assessment | None
    events: list[dict[str, Any]]


class Engine:
    def __init__(self, config: RunConfig):
        self.config = config
        self.run_dir = self._fresh_run_dir(Path(config.run_dir))
        self._suffix = self.run_dir.name[len(Path(config.run_dir).name):]
        self.registry = Registry(config.registry_root)
        self.events: list[dict[str, Any]] = []
        self._seq = 0

    # ------------------------------------------------------------------
    def emit(self, stage: str, status: str, message: str, **data: Any) -> None:
        self._seq += 1
        ev = {
            "seq": self._seq,
            "ts": now_iso(),
            "stage": stage,
            "status": status,
            "message": message,
            "data": json_safe(data),
        }
        self.events.append(ev)
        append_jsonl(self.run_dir / "events.jsonl", ev)
        if self.config.event_sink:
            try:
                self.config.event_sink(ev)
            except Exception:
                pass

    # ------------------------------------------------------------------
    @staticmethod
    def _fresh_run_dir(base: Path) -> Path:
        """Never overwrite a finished run.

        Run ids are deterministic (contract hash + seed), so a repeat lands on
        the same name. A finished run (it has run.json) keeps its artifacts; the
        repeat goes to ``<name>-r2``, ``-r3``, ... and is charged to the ledger
        like any other run. A directory without run.json is an interrupted run
        and is reused.
        """
        candidate, n = base, 1
        while (candidate / "run.json").exists():
            n += 1
            candidate = base.with_name(f"{base.name}-r{n}")
        return ensure_dir(candidate)

    def _run_id(self, contract: Contract) -> str:
        return f"ar-{contract.hash[:8]}-{self.config.run_seed}{self._suffix}"

    def _provenance(self, contract: Contract, meta: dict[str, Any], access_trace: list[dict[str, Any]]) -> dict[str, Any]:
        discovery_hash = sha256_file(self.config.discovery_path)
        return {
            "datasets": [
                {
                    "hash": discovery_hash,
                    "role": "discovery",
                    "source": Path(self.config.discovery_path).name,
                    "license": contract.provenance_raw.get("discovery_license", "unknown"),
                },
                {
                    "hash": meta.get("dataset_hash", self.config.sealed_dataset_hash),
                    "role": "sealed",
                    "source": meta.get("name", "sealed"),
                    "license": contract.provenance_raw.get("confirmation_license", "unknown"),
                    "budget_total": meta.get("budget_total"),
                    "budget_remaining": meta.get("budget_remaining"),
                },
            ],
            "access_log_hash": sha256_obj(access_trace),
            "access_calls": len(access_trace),
            "code_hash": sha256_obj(
                {p.name: sha256_file(p) for p in CODE_FILES if Path(p).exists()}
            ),
            "split": {
                "discovery_groups": int(len(set(self._groups(contract.data, contract)))),
                "confirmation_groups": int(meta.get("n_groups", 0)),
                "group_overlap": "checked",
            },
        }

    @staticmethod
    def _groups(df: pd.DataFrame, contract: Contract) -> np.ndarray:
        if contract.group_column and contract.group_column in df.columns:
            return df[contract.group_column].astype(str).to_numpy()
        return np.array(["__all__"] * len(df))

    def _write_power(self, power: Assessment) -> None:
        atomic_write_json(self.run_dir / "power.json", power.to_dict())

    def _refusal_certificate(
        self,
        contract: Contract,
        run_id: str,
        power: Assessment | None,
        reason_code: str,
        message: str,
        meta: dict[str, Any],
        refusals_extra: list[Any] | None = None,
        limitations: list[str] | None = None,
    ) -> dict[str, Any]:
        access_trace: list[dict[str, Any]] = []
        provenance = self._provenance(contract, meta, access_trace)
        cert = build_certificate(
            contract=contract,
            run_id=run_id,
            seed=self.config.run_seed,
            claim=f"Refused: {message}",
            claim_type="refusal",
            status="refused",
            tier=None,
            verdict="refused",
            provenance=provenance,
            search_price={
                "proposed": 0,
                "effective_independent": 0,
                "alpha_allocated": 0.0,
                "alpha_spent": 0.0,
                "selection_null": {"n_permutations": 0, "p_value": None, "max_stat_p95": None},
            },
            gate_oc=None,
            adversary=[],
            commitments={"registered": True, "prediction_hash": None, "confirmed": False},
            interpretability_cost={"terms": None, "max_terms": None, "gap_vs_unconstrained": None},
            refusals=[{"reason_code": reason_code, "message": message}] + (refusals_extra or []),
            reproduce=f"python -m autoresearch power --contract {self.config.contract_path}",
            evidence={"power": power.to_dict() if power else None},
            limitations=(limitations or []) + list(self.config.limitations),
            expires=self._expires(contract, meta),
            key=None,
        )
        key = load_or_create_key(self.config.key_path or (Path(self.config.registry_root) / "certificate.key"))
        cert = sign_certificate(cert, key)
        save_certificate(cert, self.run_dir, key)
        self.registry.register_contract(contract)
        self.registry.issue_certificate(cert, run_id, str(self.run_dir))
        self.emit("certificate", "refused", message, reason_code=reason_code, run_id=run_id)
        atomic_write_json(
            self.run_dir / "run.json",
            {
                "run_id": run_id,
                "created_at": now_iso(),
                "status": "refused",
                "reason_code": reason_code,
                "mode": self.config.mode,
                "contract_hash": contract.hash,
                "seed": self.config.run_seed,
                "config": self._config_summary(),
                "vault_ledger_head": self.vault_head(),
                "files": self._manifest(),
            },
        )
        # Nothing may be written to the run directory after the manifest above.
        return cert

    def vault_head(self) -> str:
        try:
            from .vault import Vault

            return Vault(self.config.vault_root).ledger_head()
        except Exception:
            return "unknown"

    def _manifest(self) -> dict[str, Any]:
        files = {}
        for p in sorted(self.run_dir.glob("*")):
            if p.is_file():
                files[p.name] = sha256_file(p)
        return files

    @staticmethod
    def _expires(contract: Contract, meta: dict[str, Any]) -> str | None:
        """Claims age with their data: valid N days from when confirmation was sealed."""
        import datetime as _dt

        sealed_at = meta.get("sealed_at")
        if not sealed_at:
            return None
        day = _dt.date.fromisoformat(str(sealed_at)[:10])
        return (day + _dt.timedelta(days=contract.decision.certificate_valid_days)).isoformat()

    def _procedure_plan(self, contract: Contract, budget: Any, alpha_run: float) -> dict[str, Any]:
        """The registered procedure: everything that determines the winner."""
        cfg = self.config
        package = Path(__file__).parent
        return {
            "schema": "autoresearch.procedure/v1",
            "contract_hash": contract.hash,
            "discovery_sha256": sha256_file(cfg.discovery_path),
            "sealed_dataset": cfg.sealed_dataset_hash,
            "seed": int(cfg.run_seed),
            "complexity_budget": budget.to_dict(),
            "search": {
                "candidate_budget": cfg.candidate_budget or budget.max_candidates,
                "generations": int(cfg.generations or budget.max_generations),
                "proposers": ["operator_families"] + (["llm"] if cfg.llm else []),
                "feedback": "categorical {status, reason_code} via sanitize_feedback",
                "cv": {"scheme": "group-aware stratified k-fold", "n_splits": int(cfg.n_splits)},
            },
            "selection_rule": (
                "max cross-validated mean paired delta over every evaluated candidate; "
                "tie-break lower complexity, then candidate id"
            ),
            "selection_null": {
                "statistic": "max cross-validated paired delta over the candidate pool",
                "permutations": int(min(cfg.n_permutations, contract.adversary.permutations)),
                "permuted_within": contract.group_column,
                "data": "discovery only",
            },
            "confirmation": {"candidates": 1, "estimand": contract.metric_raw.get("primary"), "alpha": float(alpha_run)},
            "code_sha256": sha256_obj({p.name: sha256_file(p) for p in sorted(package.glob("*.py"))}),
        }

    def _config_summary(self) -> dict[str, Any]:
        return {
            "run_seed": int(self.config.run_seed),
            "n_permutations": int(self.config.n_permutations),
            "n_splits": int(self.config.n_splits),
            "candidate_budget": self.config.candidate_budget,
            "generations": self.config.generations,
            "mode": self.config.mode,
        }

    # ------------------------------------------------------------------
    def _reset_run_dir(self) -> None:
        """Remove artifacts left by an earlier run into the same directory.

        Run ids are deterministic (contract hash + seed), so a re-run lands in the
        same directory. Without this, ``events.jsonl`` would interleave both runs
        and stale artifacts from the earlier run would enter the new manifest.

        Only files the engine itself wrote are removed: those named in the prior
        ``run.json`` manifest, plus the manifest and the event log. ``--run-dir``
        may point anywhere, so unknown files are never touched.
        """
        owned = {"run.json", "events.jsonl"}
        prior = self.run_dir / "run.json"
        if prior.exists():
            try:
                owned |= set((json.loads(prior.read_text(encoding="utf-8")).get("files") or {}).keys())
            except (OSError, ValueError):
                pass
        for name in owned:
            p = self.run_dir / name
            if p.parent == self.run_dir and p.is_file():
                p.unlink()

    def run(self) -> RunResult:
        cfg = self.config
        self._reset_run_dir()
        self.events = []
        self._seq = 0
        self.emit("contract", "start", "Loading and validating the contract against discovery data.")
        contract = load_contract(cfg.contract_path, cfg.discovery_path)
        self.emit(
            "contract",
            "ok",
            "Contract valid. The question is explicit.",
            contract_hash=contract.hash,
            n_rows=len(contract.data),
            n_groups=len(set(self._groups(contract.data, contract))),
            baseline=contract.baseline.name,
            effect=contract.decision.smallest_effect_that_matters,
        )
        atomic_write_text(self.run_dir / "contract.yaml", contract.canonical_yaml())
        atomic_write_json(self.run_dir / "validation.json", contract.validation.to_dict() if contract.validation else {})

        meta = cfg.sealed.meta()
        contract.attach_confirmation_meta(meta)
        discovery_groups = sorted(set(map(str, self._groups(contract.data, contract))))
        contract.check_group_overlap(discovery_groups, meta.get("group_ids_hash"))
        self.emit(
            "vault",
            "ok",
            "Sealed confirmation set is reachable through the label service only.",
            dataset_hash=meta.get("dataset_hash"),
            n_rows=meta.get("n_rows"),
            n_groups=meta.get("n_groups"),
            budget_remaining=meta.get("budget_remaining"),
        )

        budget = compute_budget(contract, contract.data)
        atomic_write_json(self.run_dir / "budget.json", budget.to_dict())
        self.emit("budget", "ok", "Complexity budget computed before any candidate is proposed.", **budget.to_dict())

        alpha_total = contract.decision.alpha * contract.decision.max_runs
        alpha_run = contract.decision.alpha
        self.registry.register_contract(contract)
        dataset_id = str(meta.get("dataset_hash") or cfg.sealed_dataset_hash)
        self.registry.ensure_alpha(dataset_id, alpha_total)
        if not self.registry.can_spend_alpha(dataset_id, alpha_run):
            state = self.registry.alpha_state(dataset_id)
            self.emit("registry", "refused", "Alpha budget exhausted for this dataset lifetime.", **state)
            msg = (
                f"Alpha budget exhausted for dataset {dataset_id}: {state['spent']:.4f} of {state['allocated']:.4f} spent. "
                "A new test would not be a pre-registered spend. Register a new dataset lifetime or stop."
            )
            cert = self._refusal_certificate(contract, self._run_id(contract), None, "alpha_exhausted", msg, meta)
            return RunResult(self._run_id(contract), self.run_dir, cert, None, self.events)

        self.emit("power", "start", "Pricing the search before any hypothesis is proposed.")
        power = assess(
            contract,
            contract.data,
            confirmation_n=int(meta["n_rows"]),
            confirmation_groups=int(meta["n_groups"]),
            confirmation_prevalence=meta.get("prevalence"),
            k_candidates=cfg.candidate_budget or contract.power.k_candidates,
            alpha=alpha_run,
            target_power=contract.decision.target_power,
            seed=cfg.run_seed,
        )
        self._write_power(power)
        self.emit(
            "power",
            power.verdict,
            power.prescription,
            mde_plain=power.mde_plain,
            mde_search=power.mde_search,
            effect=power.effect_that_matters,
            required_n=power.required_n_search,
            max_k=power.max_k_answerable,
        )
        if power.verdict == "NOT_ANSWERABLE":
            cert = self._refusal_certificate(
                contract,
                self._run_id(contract),
                power,
                "not_answerable",
                power.prescription,
                meta,
                refusals_extra=[{"reason_code": "search_not_priced", "message": "No search was run; searching would not make the question answerable."}],
            )
            return RunResult(self._run_id(contract), self.run_dir, cert, power, self.events)

        # ------------------------------------------------------------------
        # Register the procedure and selection rule before any candidate
        # exists (spec §8). The plan is written by the engine, not a person.
        # ------------------------------------------------------------------
        run_id = self._run_id(contract)
        plan = self._procedure_plan(contract, budget, alpha_run)
        plan_entry = self.registry.register_procedure(contract.hash, run_id, plan)
        atomic_write_json(self.run_dir / "plan.json", plan)
        self.emit(
            "registry",
            "registered",
            "Procedure and selection rule registered before the search. The winner is a consequence of this plan.",
            plan_hash=plan_entry["plan_hash"],
        )

        # ------------------------------------------------------------------
        # Search (discovery labels only; confirmation stays sealed)
        # ------------------------------------------------------------------
        fm = FeatureMatrix(contract.data, contract, fit=True)
        baseline_scores = fm.score_candidate(contract.baseline_candidate())
        proposer = Proposer(
            contract=contract,
            data=contract.data,
            fm=fm,
            budget=budget,
            seed=cfg.run_seed,
            domain_brief=cfg.domain_brief,
            llm=cfg.llm,
        )
        n_generations = int(cfg.generations or budget.max_generations)
        initial_n = min(48, budget.max_candidates)
        all_candidates: dict[str, Candidate] = {}
        all_results: dict[str, dict[str, Any]] = {}
        proposed_count = 0
        kept: list[Candidate] = []
        feedback: list[dict[str, Any]] = []
        discovery_selected: Candidate | None = None
        discovery_selected_result: dict[str, Any] | None = None

        for gen in range(n_generations):
            remaining = (cfg.candidate_budget or budget.max_candidates) - proposed_count
            if remaining <= 0:
                break
            if gen == 0:
                batch = proposer.seed_candidates(n=min(initial_n, remaining))
            else:
                batch = proposer.next_generation(kept, feedback, n=min(40, remaining), generation=gen)
            if cfg.llm:
                try:
                    batch.extend(proposer.llm_candidates(n=min(8, remaining - len(batch)), rejection_history=feedback))
                except Exception as exc:
                    self.emit("proposer", "warn", f"LLM proposer unavailable: {exc}")
            # Deduplicate within this batch.
            seen = set()
            unique_batch = []
            for c in batch:
                if c is None:
                    continue
                if c.id not in seen:
                    seen.add(c.id)
                    unique_batch.append(c)
            batch = unique_batch[:remaining]
            if not batch:
                break
            proposed_count += len(batch)
            proposer.remember(batch)
            for c in batch:
                all_candidates[c.id] = c
                self.emit(
                    "proposal",
                    "proposed",
                    c.mechanism,
                    candidate_id=c.id,
                    family=c.family,
                    source=c.source,
                    complexity=c.complexity,
                )
            ev = evaluate_candidates(
                contract.data,
                contract,
                batch,
                fm,
                baseline_scores,
                budget,
                n_splits=cfg.n_splits,
                seed=cfg.run_seed + gen,
                keep_fraction=0.4,
            )
            for cid, r in ev.results.items():
                all_results.setdefault(cid, r)
                fb = r.get("feedback", {})
                self.emit(
                    "feedback",
                    fb.get("status", "rejected"),
                    f"{cid}: {fb.get('reason_code', 'rejected')}",
                    candidate_id=cid,
                    reason_code=fb.get("reason_code"),
                    feedback_status=fb.get("status"),
                )
            feedback = sanitize_feedback(
                [{"id": cid, **r.get("feedback", {})} for cid, r in ev.results.items()]
            )
            kept = [all_candidates[cid] for cid, r in ev.results.items() if r.get("feedback", {}).get("status") == "kept"]
            if ev.selected_candidate:
                discovery_selected = ev.selected_candidate
                discovery_selected_result = ev.selected_result

        if discovery_selected is None or discovery_selected_result is None:
            msg = "The search produced no scorable candidate within the complexity budget; no claim can be made."
            self.emit("search", "refused", msg)
            cert = self._refusal_certificate(contract, self._run_id(contract), power, "no_scorable_candidate", msg, meta)
            return RunResult(self._run_id(contract), self.run_dir, cert, power, self.events)

        # Global deterministic selection rule across every generation.
        evaluated = [(cid, r) for cid, r in all_results.items() if r.get("status") == "evaluated" and np.isfinite(r.get("mean_delta", float("nan")))]
        evaluated.sort(key=lambda kv: (-kv[1]["mean_delta"], kv[1].get("complexity", 99), kv[0]))
        best_id, best_result = evaluated[0][0], evaluated[0][1]
        selected = all_candidates[best_id]
        self.emit(
            "selection",
            "selected",
            selected.mechanism,
            candidate_id=selected.id,
            discovery_delta=best_result["mean_delta"],
            complexity=selected.complexity,
            proposed=proposed_count,
        )

        # Effective independent hypotheses across all generations.
        eval_ids = [cid for cid, r in evaluated]
        score_cols = []
        score_ids = []
        comp_map = {}
        for cid in eval_ids:
            c = all_candidates[cid]
            try:
                score_cols.append(fm.score_candidate(c))
                score_ids.append(cid)
                comp_map[cid] = c.complexity
            except Exception:
                continue
        if score_cols:
            sm = np.column_stack(score_cols)
            eff_k, clusters = effective_independent(sm, score_ids)
        else:
            eff_k, clusters = 0, []

        # Discovery stability.
        selected_scores = fm.score_candidate(selected)
        loo = leave_one_group_out_delta(
            contract.data[contract.outcome_column].to_numpy().astype(int),
            self._groups(contract.data, contract),
            selected_scores,
            baseline_scores,
        )
        fold_deltas = best_result.get("fold_deltas", [])
        discovery_stability = {
            "fold_fraction_positive": float(np.mean([d > 0 for d in fold_deltas])) if fold_deltas else float("nan"),
            "fold_deltas": fold_deltas,
            "leave_one_group_out": loo,
        }
        atomic_write_json(self.run_dir / "selection.json", {
            "candidate": selected.to_dict(),
            "result": best_result,
            "proposed": proposed_count,
            "effective_independent": eff_k,
            "clusters": clusters,
            "discovery_stability": discovery_stability,
        })

        # ------------------------------------------------------------------
        # Pre-commit the prediction before the sealed set is touched.
        # ------------------------------------------------------------------
        prediction = {
            "run_id": run_id,
            "contract_hash": contract.hash,
            "candidate_id": selected.id,
            "candidate_hash": sha256_obj(selected.to_dict()),
            "procedure_hash": plan_entry["plan_hash"],
            "selection_rule": "max cross-validated mean paired delta; tie-break lower complexity, then candidate id",
            "predicted_direction": "positive",
            "predicted_ci_excludes_zero": True,
            "predicted_effect_at_least": contract.decision.smallest_effect_that_matters,
            "predicted_failure": "if confirmation delta < effect_that_matters, the claim is refused",
            "proposed": proposed_count,
            "effective_independent": eff_k,
        }
        pred_entry = self.registry.commit_prediction(contract.hash, run_id, prediction)
        atomic_write_json(self.run_dir / "prediction.json", prediction)
        self.emit(
            "commitment",
            "committed",
            "Prediction hashed and registered before confirmation. No retrofitting.",
            prediction_hash=pred_entry["prediction_hash"],
            candidate_id=selected.id,
        )

        # ------------------------------------------------------------------
        # Exactly one candidate touches the sealed set for the primary test.
        # ------------------------------------------------------------------
        self.emit("confirm", "start", "Submitting exactly one candidate to the sealed label service.")
        access_trace: list[dict[str, Any]] = []
        try:
            confirmation = cfg.sealed.confirm(
                selected.to_dict(),
                purpose="primary confirmation",
                cost=1,
                detail="full",
                seed=cfg.run_seed,
            )
        except BudgetExhausted as exc:
            msg = f"Confirmation budget exhausted before the primary test: {exc}"
            self.emit("confirm", "refused", msg)
            cert = self._refusal_certificate(contract, run_id, power, "confirmation_budget_exhausted", msg, meta)
            return RunResult(run_id, self.run_dir, cert, power, self.events)
        access_trace.append(
            {
                "purpose": "primary confirmation",
                "candidate_id": selected.id,
                "delta": confirmation.get("delta"),
                "n": confirmation.get("n"),
            }
        )
        # The vault ledger head carries wall-clock time; it belongs in the run
        # log, not in an artifact that must be byte-identical across processes.
        sealed_access_entry = confirmation.pop("access_entry_hash", None)
        atomic_write_json(self.run_dir / "confirmation.json", confirmation)
        self.emit(
            "confirm",
            "ok",
            "Sealed confirmation returned aggregates only.",
            delta=confirmation.get("delta"),
            ci_low=confirmation.get("ci_low"),
            ci_high=confirmation.get("ci_high"),
            p=confirmation.get("p_one_sided"),
            n=confirmation.get("n"),
            n_groups=confirmation.get("n_groups"),
        )

        # ------------------------------------------------------------------
        # Selection-aware null (spec §6): the best score the whole search reaches
        # on outcomes permuted within groups. Discovery data only, so the sealed
        # set still sees exactly one candidate.
        # ------------------------------------------------------------------
        n_perm = int(min(cfg.n_permutations, contract.adversary.permutations))
        if score_cols:
            y = contract.data[contract.outcome_column].to_numpy().astype(int)
            groups = self._groups(contract.data, contract)
            folds = assign_folds(groups, y, n_splits=cfg.n_splits, seed=cfg.run_seed)
            null_summary = selection_aware_null(
                np.column_stack(score_cols),
                baseline_scores,
                y,
                groups,
                folds,
                score_ids,
                comp_map,
                n_permutations=n_perm,
                rng_parts=("selection-null", contract.hash, cfg.run_seed),
            )
        else:
            null_summary = {"n_permutations": 0, "p_value": None, "max_stat_p95": None, "null_mean": None, "null_min": None, "null_max": None}
        if null_summary["p_value"] is None:
            null_summary["note"] = "No permutation produced a finite search maximum; the gate cannot certify."
        atomic_write_json(self.run_dir / "selection_null.json", null_summary)
        self.emit(
            "null",
            "ok" if null_summary["p_value"] is not None else "warn",
            f"Selection-aware null: {null_summary['n_permutations']} full-pool searches on permuted outcomes, discovery data only.",
            p_value=null_summary["p_value"],
            p95=null_summary["max_stat_p95"],
        )

        # ------------------------------------------------------------------
        # Adversary on discovery, plus sealed split checks.
        # ------------------------------------------------------------------
        self.emit("adversary", "start", "Running the automated red team.")
        attacks = run_adversary(
            contract,
            contract.data,
            fm,
            selected,
            contract.data[contract.outcome_column].to_numpy().astype(int),
            self._groups(contract.data, contract),
            baseline_scores,
            float(best_result["mean_delta"]),
            budget,
            alpha=alpha_run,
            seed=cfg.run_seed,
            validation=contract.validation.to_dict() if contract.validation else None,
            discovery_groups=discovery_groups,
            confirmation_group_hash=meta.get("group_ids_hash"),
            code_files=CODE_FILES,
        )
        atomic_write_json(self.run_dir / "adversary.json", attacks)
        for a in attacks:
            self.emit("adversary", a.get("result", "info"), f"{a.get('attack')}: {a.get('result')}", **{k: v for k, v in a.items() if k not in ("attack", "result")})

        # ------------------------------------------------------------------
        # Gate and certificate.
        # ------------------------------------------------------------------
        decision = assign_tier(
            contract,
            selected.to_dict(),
            confirmation,
            {
                **null_summary,
                "effective_independent": eff_k,
                "proposed": proposed_count,
            },
            discovery_stability,
            attacks,
            power,
            alpha_run,
            calibration_table=cfg.calibration_table,
            budget_info=budget.to_dict(),
        )
        atomic_write_json(self.run_dir / "gate.json", decision.to_dict())
        self.emit("gate", decision.tier, decision.claim, checks=[c for c in decision.checks if not c["ok"]])

        # Interpretability bill (spec §12.5): the readable rule against an
        # unconstrained model on the same folds.
        bill = interpretability_bill(
            contract.data,
            contract,
            selected_scores,
            baseline_scores,
            assign_folds(self._groups(contract.data, contract), contract.data[contract.outcome_column].to_numpy().astype(int), n_splits=cfg.n_splits, seed=cfg.run_seed),
            seed=cfg.run_seed,
        )

        provenance = self._provenance(contract, meta, access_trace)
        cert = build_certificate(
            contract=contract,
            run_id=run_id,
            seed=cfg.run_seed,
            claim=decision.claim,
            claim_type=decision.claim_type,
            status="issued",
            tier=decision.tier,
            verdict=decision.verdict,
            provenance=provenance,
            search_price={
                "proposed": int(proposed_count),
                "effective_independent": int(eff_k),
                "alpha_allocated": float(alpha_total),
                "alpha_spent": float(alpha_run),
                "selection_null": {
                    "n_permutations": null_summary["n_permutations"],
                    "p_value": null_summary["p_value"],
                    "max_stat_p95": null_summary["max_stat_p95"],
                    "null_mean": null_summary["null_mean"],
                    "null_min": null_summary["null_min"],
                    "null_max": null_summary["null_max"],
                    "observed_max_stat": null_summary.get("observed_max_stat"),
                    "computed_on": null_summary.get("computed_on"),
                },
                "sealed_queries": len(access_trace),
                "clusters": len(clusters),
            },
            gate_oc=decision.gate_oc,
            adversary=attacks,
            commitments={
                "registered": True,
                "procedure_hash": plan_entry["plan_hash"],
                "procedure_registered_before_search": True,
                "prediction_hash": pred_entry["prediction_hash"],
                "registered_at": "registry ledger (procedure_registered, prediction_committed); kept out of the certificate so it stays byte-reproducible",
                "predicted_failure": prediction["predicted_failure"],
                "confirmed": decision.verdict == "affirmed",
            },
            interpretability_cost={
                "terms": int(selected.complexity),
                "max_terms": int(budget.max_terms),
                **bill,
                "note": (
                    "Unconstrained model minus the selected rule, paired delta AUC on the same discovery folds. "
                    "A readable rule buys trust and can cost accuracy; a simple wrong model is acted on without hesitation."
                ),
            },
            refusals=list(power.warnings) + [a for a in attacks if a.get("result") == "warn"],
            reproduce=f"python -m autoresearch reproduce --run {self.run_dir.name} --seed {cfg.run_seed}",
            evidence={
                "candidate": selected.to_dict(),
                "baseline": contract.baseline.name,
                "discovery_delta": float(best_result["mean_delta"]),
                "discovery_fold_deltas": best_result.get("fold_deltas"),
                "n_discovery": int(len(contract.data)),
                "n_groups_discovery": int(len(set(discovery_groups))),
                "n_confirmation": int(confirmation.get("n", 0)),
                "n_groups_confirmation": int(confirmation.get("n_groups", 0)),
                "delta": confirmation.get("delta"),
                "point_delta": confirmation.get("point_delta"),
                "ci_low": confirmation.get("ci_low"),
                "ci_high": confirmation.get("ci_high"),
                "p_one_sided": confirmation.get("p_one_sided"),
                "auc_candidate": confirmation.get("auc_candidate"),
                "auc_baseline": confirmation.get("auc_baseline"),
                "top_k_enrichment": confirmation.get("top_k_enrichment"),
                "frac_improved": confirmation.get("frac_improved"),
                "mean_pct_rank": confirmation.get("mean_pct_rank"),
                "rank_coef": confirmation.get("rank_coef"),
                "confirmation_stability": confirmation.get("stability_confirmation"),
                "discovery_stability": discovery_stability,
                "effective_independent": int(eff_k),
            },
            limitations=[
                "Confirmation labels were accessed only through the label service; no row-level labels entered the search process.",
                "The certificate is conditional on the registered contract and selection rule. Changing either invalidates it.",
                "A refusal is a first-class output; it is not evidence that the hypothesis is false.",
            ] + list(cfg.limitations),
            expires=self._expires(contract, meta),
            key=None,
        )
        key = load_or_create_key(cfg.key_path or (Path(cfg.registry_root) / "certificate.key"))
        cert = sign_certificate(cert, key)
        save_certificate(cert, self.run_dir, key)
        self.registry.issue_certificate(cert, run_id, str(self.run_dir))
        try:
            self.registry.spend_alpha(dataset_id, alpha_run, run_id, f"primary confirmation under {contract.hash}")
        except RegistryError as exc:
            self.emit("registry", "warn", str(exc))
        self.emit("certificate", "issued", decision.claim, certificate_hash=cert.get("certificate_hash"), tier=decision.tier, verdict=decision.verdict)
        atomic_write_json(
            self.run_dir / "run.json",
            {
                "run_id": run_id,
                "created_at": now_iso(),
                "status": "issued",
                "mode": cfg.mode,
                "contract_hash": contract.hash,
                "seed": cfg.run_seed,
                "tier": decision.tier,
                "verdict": decision.verdict,
                "config": self._config_summary(),
                "vault_ledger_head": self.vault_head(),
                "sealed_access_entry_hash": sealed_access_entry,
                "registry": self.registry.summary(),
                "files": self._manifest(),
            },
        )
        return RunResult(run_id, self.run_dir, cert, power, self.events)
