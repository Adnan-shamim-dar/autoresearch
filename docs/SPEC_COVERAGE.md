# Spec coverage

Status of this build against `Auto.md` (build specification v1), section by section. **Built** means implemented and covered by a test in `tests/`; **partial** means built with a stated gap; **owed** means not done, with the reason.

## Summary

| § | Module / requirement | Status |
|---|---|---|
| 1 | Contract + validator | Built |
| 2 | Vault, physical sealing, twins, ledger | Built (one deployment gap) |
| 3 | Power oracle + design advisor | Built |
| 4 | Complexity budget | Built |
| 5 | Proposer + feedback-leak test | Built |
| 6 | Evaluator, acceptance tests with broken guards | Built (reusable holdout owed) |
| 7 | Gate + published OC table | Built; synthetic and real-data (plasmode) tables |
| 8 | Registry, procedure registration, alpha ledger | Built |
| 9 | Adversary + its own false-alarm rate | Built |
| 10 | Certificate | Built |
| 11 | Interfaces | Partial: no public registry or DOIs |
| 12 | Non-obvious constraints | Mostly built; see below |
| 13 | Milestones | 1–10 met |
| 14 | The five proofs | 1 and 4 met; 5 has a one-command kit and awaits a third party; 2 and 3 owed |

## Section by section

### §1 Contract
- One YAML file; `contract_hash = sha256(canonical_yaml)` is carried by every artifact.
- `decision.smallest_effect_that_matters` is required, has no default and must be > 0.
- Validation checks types, group structure, missingness, metric computability and leakage (forbidden and post-outcome columns, near-perfect outcome proxies, duplicate units, discovery/confirmation group overlap). A run cannot start on a failed validation.
- **Acceptance test:** a wrong group column fails with the named reason `missing_group_column` (`tests/test_contract.py`).

### §2 Vault
- Roles and capability bits as specified; sealed datasets grant only `confirm`.
- **Physical sealing.** The sealed blob is encrypted (encrypt-then-MAC) *and* carries an OS deny-read lock (`icacls` on Windows, `chmod 000` on POSIX). The label service runs as a **separate process** (`autoresearch label-service`), and only it lifts the lock, reads and restores it. The engine holds an HTTP client with a bearer token and never receives the key path. No plaintext confirmation rows remain on disk after sealing.
- Synthetic twins (within-group outcome permutation) for every dataset.
- **Acceptance tests** (`tests/test_vault.py`):
  1. `pd.read_parquet` of the sealed file raises `PermissionError` from the OS, both in-process and from a separate process.
  2. Exceeding the budget raises, across the process boundary, and the ledger shows the attempt.
  3. Tampering with any ledger entry breaks the hash chain.
- **Gap:** the lock is against the same OS account, so it stops accidents (the spec's stated threat model), not a deliberate unlock. Full separation means running the label service under its own OS account that owns the key; that's a deployment step.

### §3 Power oracle and design advisor
- `MDE_plain`, `MDE_search(K)`, verdict (`ANSWERABLE` / `ANSWERABLE_IF_K_LE(n)` / `NOT_ANSWERABLE`), prescription, MDE(n, K) and power curves.
- The design advisor ranks partial pooling, direction aggregation, endpoint substitution, recovering discarded information, stratify/match and sequential design, each with power, what it establishes and what it does not.
- Direction aggregation is also **executable**: a contract with `commitments: {signs: true}` runs its pre-committed sign test as one aggregate sealed query.
- **Known defect (measured by the benchmark):** the oracle over-refuses. Its variance estimate comes from a label-free equal-weight combination of all declared features, which is noisier against the baseline than a realistic winning rule. On the benchmark's development set it refused 167 of 200 tasks, many with real effects well above the declared 0.05.
- **Acceptance test:** at a planted 0.04 with MDE about 0.09 the verdict is `NOT_ANSWERABLE`, and the prescribed sample size recovers the effect in simulation (`tests/test_power.py`).

### §4 Budget
- Capacity is computed from effective sample size before the search, printed in the certificate, and over-budget candidates are rejected unexamined. With no group column there's no design effect.

### §5 Proposer
- Operator families (seeds, mutation, recombination, gated rules, importance-derived) with keep-rate-adaptive budgets, plus an optional OpenAI-compatible LLM proposer that must state a mechanism.
- All feedback passes through `sanitize_feedback`, leaving `{id, status, reason_code}` and nothing else. Reason codes that could smuggle numbers are replaced.
- Effective independent hypotheses are tracked and feed the multiplicity correction.
- **Acceptance test** (`feedback_leak.py`, `calibration/feedback_leak.md`): one score-seeking proposer on pure-null data. A **leaky** channel promoted 36 of 100 null runs; the **guarded** channel promoted 1 of 100 (95% CI 0–5%).

### §6 Evaluator
- Group-aware CV; selection on discovery; the sealed set sees **exactly one** candidate (asserted per run).
- Paired per-group inference; all secondary metrics always computed.
- **Selection-aware null:** the best score the whole candidate pool reaches on outcomes permuted within groups, on discovery data only. Holding the pool fixed across permutations is a surrogate for re-running the adaptive proposer (§12.1); the maximum is taken over every evaluated candidate.
- With no group column, units are independent: unit-level folds and bootstrap.
- **Acceptance tests**, each with its broken-guard counterpart:
  1. A planted signal is recovered, and its search maximum clears the null's 95th percentile.
  2. Pure noise: held out, the CI contains zero and nothing is promoted. Tested on its own selection units (guard broken), noise is promoted.
  3. The paired per-group delta agrees with scikit-learn to 1e-9.
  4. Two runs in two processes with different `PYTHONHASHSEED` produce byte-identical artifacts. Only `run.json` and `events.jsonl` carry wall-clock time, and the event log matches once timestamps are removed.
- **Owed:** a reusable confirmation set via noise-added queries with a declared lifetime. The engine never reuses the sealed set, so nothing depends on it yet.

### §7 Gate
- Tiers `confirmed → supported → provisional → exploratory → artifact_risk` (plus confirmed negatives), assigned by a function. The gate can say yes, and a test shows it.
- **OC table** (`calibration/oc_table.md`): 20 replicates per effect, with the full adversary in every replicate.

  | True effect | Promoted | Confirmed |
  |---:|---:|---:|
  | 0.00 | 0.00 | 0.00 |
  | 0.03 | 0.00 | 0.00 |
  | 0.06 | 0.10 | 0.00 |
  | 0.10 | 0.85 | 0.00 |
  | 0.15 | 1.00 | 0.85 |

  False-positive rate: 0 of 20 nulls.
- **Real-data table** (`calibration/oc_table_real.md`, `autoresearch calibrate --real`): plasmode design on the 569 real WDBC tumours. Real measurements, correlations and sample size; only the outcome is re-drawn, so the true effect is known. 100 replicates per effect:

  | True effect | Promoted | 95% CI |
  |---:|---:|---:|
  | 0.00 | 0.00 | 0.00–0.04 |
  | 0.03 | 0.00 | 0.00–0.04 |
  | 0.06 | 0.00 | 0.00–0.04 |
  | 0.10 | 0.00 | 0.00–0.04 |
  | 0.15 | 0.44 | 0.35–0.54 |

  No false positives, and a gate that's much more conservative on real data of this size than on the synthetic substrate. The binding check is the selection-aware null: at a true 0.10 the sealed set confirms the effect (one-sided p ≤ 0.05) in 96 of 100 replicates, but the search-priced p clears α in only 6, because pricing a 32-candidate search on 341 discovery tumours is expensive. The adversary killed 0 of 44 honest promotions. The calibration contract asks a 0.10 question, because the power oracle refuses 0.04 at this n (MDE after search about 0.08–0.10).
- **Gap:** one real dataset, with no group structure. The group-aware machinery is calibrated only on synthetic data.

### §8 Registry
- Contracts are registered at creation; the denominator is exposed.
- **The procedure is registered before the search:** the engine writes and hashes its own plan (selection rule, budget, seed, null, code hash). The certificate binds `procedure_hash` and `prediction_hash`.
- Predictions are committed before confirmation, first write wins.
- Alpha is conserved per dataset lifetime, across runs and callers (a repeated run is refused with `alpha_exhausted`). Repeat runs never overwrite a finished run.

### §9 Adversary
- Runs on every promotion: pseudo-group control, permutation of outcomes, features within strata and group labels, leave-one-site-out, leakage scan with a code hash, baseline advocate, prespecified failure, and context heterogeneity per declared context.
- **Pseudo-group control as specified.** Random halves of one homogeneous context are labelled as two contexts and run through the context-heterogeneity test. The earlier version could not fail by construction. It's now shown failing when that test's bootstrap ignores clustering.
- **Adversary calibration:** of 39 honest promotions, it killed 0 (95% CI 0–9%). Per-attack warn/fail rates are published. One finding: `group_label_permutation` warns on most real effects and carries little information.

### §10 Certificate
- Signed JSON (HMAC-SHA256) with a Markdown face. Fields: claim, tier, contract hash, provenance and access-log hash, search price (including sealed query count and null statistics), gate OC at this effect, adversary results, commitments, `interpretability_cost.gap_vs_unconstrained` (gradient boosting on every feature over the same folds), refusals, reproduce command, `expires` (sealing date + `decision.certificate_valid_days`), operating point and who chose it.
- Negative certificates have the same structure.
- **Deviation:** `registered_at` timestamps live in the registry ledger rather than the certificate, so certificates stay byte-reproducible.

### §11 Interfaces
- Library: `autoresearch.api.verify`.
- CLI: `autoresearch --help`.
- **Agent tool:** MCP `autoresearch_verify(contract, data, candidate) -> certificate`, priced at K = 1 and confirmed through the label-service process.
- Reviewer view in the console (paste a certificate and check it); `reproduce --fresh` re-runs from the certificate.
- Live run view over server-sent events.
- **Owed:** a public certificate registry with DOIs, which needs hosting and an account.

### §12 Non-obvious constraints
1. **Compute.** Vectorised ranks and permutations (adversary 168 s → 20 s per run); discovery-only null. *Partial:* no cached nulls across contracts yet.
2. **Human operator.** State-changing CLI commands and MCP verify calls are logged as `operator_decision` ledger entries; any re-test is charged to alpha by the run it triggers.
3. **LLM recall.** Model-proposed candidates are a separate provenance tier, checked against `AUTORESEARCH_LLM_CUTOFF` and `provenance.data_public_since` (`possible_recall`).
4. **Operating point.** The power oracle prints an α dial in decision units (wrong actions per 100 null decisions, missed effects per 100); the contract records `alpha_chosen_by`.
5. **Interpretability bill.** Printed in every issued certificate.
6. **Private refusals.** Every certificate starts private; `autoresearch publish --by` is an attributed ledger event; listings hide unpublished refusal text.
7. **Refusal taxonomy.** `autoresearch refusals`.
8. **Run it on itself.** *Partial:* the engine's own calibration and leak test are published with their parameters, but design choices are not yet charged to an alpha ledger of their own.
9. **Certified vs uncertified.** There is now a leaderboard: the Discovery Validity Benchmark (`docs/BENCHMARK.md`) scores any discovery agent's false-claim rate and power against hidden, re-drawn ground truth on four real datasets. It also measures this engine. No false claims in 200 tasks, but the power oracle refuses 167 of them, so only 9% of real effects are found (42% with the refusal skipped). The oracle's variance estimate, from an equal-weight reference candidate, is too pessimistic. That is an open §3 calibration defect, measured rather than assumed.

### §13 Milestones
- **1–7:** met.
- **8 (second domain, signs pre-committed):** met. `autoresearch init --domain wdbc` runs on **real data** (Wisconsin Diagnostic Breast Cancer, bundled with scikit-learn): no groups, tumour size as the deployed baseline, ten feature signs committed in advance. The 0.02-AUC question is refused before any search (it needs about 3,000 sealed tumours; there are 228). Direction aggregation runs instead: **9 of 10 signs held (p = 0.011)**. Building it exposed and fixed real generality bugs: the gate could not say yes without groups, and plaintext sealed rows were left readable.
- **9 (replication back-test):** met, on summary statistics (see §14.4).
- **10 (agent tool + registry):** met.

### §14 The five proofs
1. **Real-data OC table: met** (§7), on one real dataset without groups.
2. **A correct yes agreed with by later data: owed.** Needs a later dataset.
3. **A correct no where a conventional analysis would have looked publishable: owed.** Candidate: the WDBC 0.02 question.
4. **Replication back-test: met.** Pre-registered in `docs/backtest/PREREGISTRATION.md` (commit 0a91747), committed before outcomes were joined; results in `docs/backtest/RESULTS.md`. Run blind on 92 published findings from the Reproducibility Project: Psychology, the search-priced rule (K = 10) accepted 47 and refused 45. Its accepts replicated at 49% (95% CI 35–63%) and its refusals at 24% (14–39%); Fisher p = 0.018. The stricter MDE rule: 65% vs 28%, p = 0.002. At K = 5 the difference isn't significant (40% vs 31%, p = 0.51), so the result depends on the implicit search size. It applies the gate's pricing to summary statistics, not the full engine, and the link between small original p-values and replication was already known.
5. **Third-party reproduction: kit ready, not yet done.** `python scripts/reproduce_proofs.py` recomputes the back-test, the WDBC result and the test suite from public data and writes a report (`docs/REPRODUCE.md`). It passes from a clean directory on the author's machine. Someone else has to run it.

## Architecture note
§0 asks that modules communicate only through files on disk with declared schemas. They currently call each other in-process. The run's artifacts are files with `schema` fields, and the label service crosses a real process boundary, but the remaining modules are not separated. That would be a refactor, not a fix.
