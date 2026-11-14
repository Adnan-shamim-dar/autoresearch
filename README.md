# AutoResearch

A discovery engine that **prices the search before it runs**, and then issues either a signed **claim certificate** or a first-class **refusal with a design prescription**.

The engine is domain-general and ships with two domains:

- **Immunotherapy** (neoantigen immunogenicity ranking) on a **synthetic** TESLA/CEDAR-shaped substrate.
- **WDBC**: real breast-cancer data with no groups and pre-committed signs.

Nothing here is clinical evidence. Section-by-section status against the build specification is in [docs/SPEC_COVERAGE.md](docs/SPEC_COVERAGE.md).

## What it does

1. **Contract.** You state the question up front: unit, group, outcome, a deployed baseline (never chance), the smallest effect that matters and the decision it changes. The contract is validated against the data and hashed.
2. **Power oracle.** Before any hypothesis exists, it computes the minimum detectable effect for one candidate and for the best of K. An unanswerable question is refused with the sample size or alternative design that would fix it, plus an α dial priced in the decision's own units.
3. **Registration.** The engine writes and hashes its own plan (selection rule, budget, seed, null, code hash) and registers it before the search.
4. **Search.** Operator families, plus an optional LLM proposer, generate candidates under a complexity budget. Every proposer hears only `{kept | rejected, reason_code}`, never a number.
5. **Selection-aware null.** The whole candidate pool is re-scored on outcomes permuted within groups, using discovery data only, so the p-value prices the fact that the best of K was picked.
6. **Sealed confirmation.** Exactly one candidate goes to the sealed set. The label service runs as a **separate process**; sealed files are encrypted and carry an **OS read lock**; only aggregates come back, and every access is written to a hash-chained ledger.
7. **Gate and adversary.** Tiers are assigned by a function calibrated by the published operating-characteristics table. A red team (pseudo-group control, permutations at several levels, leave-one-site-out, leakage scan, baseline advocate, context heterogeneity) attacks every promotion, and its own false-alarm rate is published.
8. **Certificate.** HMAC-signed and byte-reproducible across processes. It records the search price, alpha spent, provenance, adversary results, the interpretability bill (rule vs an unconstrained model), expiry, and who chose the operating point. Every artifact is covered by a manifest.

Alpha is conserved **per dataset lifetime**, across runs, sessions and callers. Every contract is registered at creation, so the denominator is visible, and refusals stay private until someone publishes them.

## Quickstart

Requires Python 3.11+.

```bash
pip install -e ".[dev]"

autoresearch init                     # synthetic immunotherapy substrate; seals the confirmation set
autoresearch demo                     # end-to-end run (~4 min) + economics refusal
autoresearch certificates             # list certificates (unpublished refusals stay private)
autoresearch verify --run ar-<hash>-12345
autoresearch serve                    # console on http://127.0.0.1:8787

autoresearch --root wdbc init --domain wdbc   # second domain: real data
autoresearch --root wdbc demo --skip-economics
```

Without installing, run commands as `python -m autoresearch ...` with `PYTHONPATH=src`. Every command takes `--root <dir>` first (default: the current directory).

| Command | Purpose |
|---|---|
| `init [--domain immunotherapy\|wdbc]` | Build a substrate and seal its confirmation set |
| `demo [--reset]` | End-to-end run; `--reset` rebuilds the substrate (the calibration table is kept) |
| `run --contract C --discovery D --dataset H --vault-root V --key K` | Engine on your own contract and sealed dataset |
| `verify-candidate --contract C --discovery D --candidate JSON` | `verify(contract, data, candidate) -> certificate`, priced at K = 1 |
| `validate`, `power` | Check a contract; price a search without running it |
| `certificates`, `certificate --run R` | List or show certificates |
| `verify --run R` | Hash, signature and artifact manifest. Exit codes: 0 ok, 2 certificate invalid, 3 manifest mismatch |
| `reproduce --run R [--fresh]` | Verify, or rebuild the substrate and re-run to compare |
| `publish --run R --by NAME` | Make a certificate public (an attributed ledger event) |
| `refusals` | Refusal taxonomy across every run |
| `ledger` | Verify the vault and registry hash chains |
| `calibrate [--real] [--publish-only]`, `leak-test` | Rebuild the synthetic or real-data OC table; run the feedback-leak test |
| `backtest --data RPPdataConverted.xlsx` | Pre-registered replication back-test |
| `bench [--agent NAME] [--command CMD] [--salt S]` | Discovery Validity Benchmark; `--command` runs any external agent (an LLM analyst: `examples/bench/llm_agent.py`) |
| `public-key` | Print the Ed25519 key that verifies this root's certificates |
| `serve`, `mcp` | Web console; MCP stdio server (`autoresearch_verify`, `_check_certificate`, `_power`, `_validate`, `_catalog`, `_run`) |

## Published numbers

**Discovery Validity Benchmark** ([docs/BENCHMARK.md](docs/BENCHMARK.md)). This is the false-discovery rate of automated discovery agents on real data with hidden, re-drawn ground truth. Any agent can be run through it. Over 200 tasks, a search that doesn't price itself claims a discovery in **51% of pure-noise tasks**. A holdout tested once claims one in 7% of null tasks; an agent that re-tests the same holdout after each failure, in 14%. AutoResearch made **no false claims**. In a pre-registered run on 300 fresh tasks ([results](docs/bench_eval/RESULTS.md)) with the contract option `if_underpowered: proceed`, it found 41% of real effects (78% on 3,000-row data) with **0 false claims in 142 null tasks**. A holdout re-tested by an iterating agent: 10.6% false claims.

**Replication back-test, pre-registered** ([docs/backtest/RESULTS.md](docs/backtest/RESULTS.md)). Run blind on 92 published psychology findings with known replication outcomes (Reproducibility Project: Psychology), the gate's search-priced rule accepted 47 and refused 45. **Its accepts replicated at 49%, its refusals at 24%** (Fisher p = 0.018). The rule was committed to git before any outcome was joined.

**Gate operating characteristics on real data** ([calibration/oc_table_real.md](calibration/oc_table_real.md)). 569 real WDBC tumours with planted outcomes of known size, 100 replicates per effect: **0 false positives in 100 nulls** (95% CI 0–4%). Detection is 0% up to a true AUC gain of 0.10 and 44% at 0.15. On data this small the gate is conservative, and the search price is what binds.

**Gate operating characteristics, synthetic** ([calibration/oc_table.md](calibration/oc_table.md)). Synthetic planted-signal fixture, 20 replicates per effect, full adversary in every replicate.

| True effect | Promoted | Confirmed only | False positives |
|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.00 | 0 / 20 |
| 0.03 | 0.00 | 0.00 | |
| 0.06 | 0.10 | 0.00 | |
| 0.10 | 0.85 | 0.00 | |
| 0.15 | 1.00 | 0.85 | |

The adversary killed **0 of 39** honest promotions (95% CI 0–9%). The gate is conservative: it needs an effect of about 0.10 before it reliably promotes.

**Feedback-leak test** ([calibration/feedback_leak.md](calibration/feedback_leak.md)). Same proposer on pure-null data. When the channel leaks scores, **36 of 100** null runs are promoted; through `sanitize_feedback`, **1 of 100**.

**Second domain, real data (WDBC).** The question "does a multi-feature rule beat tumour size by 0.02 AUC" is refused before any search: it needs about 3,000 sealed tumours and there are 228. Direction aggregation runs instead, and **9 of 10 pre-committed feature signs held** on the sealed set (one-sided binomial p = 0.011).

## Layout

| Module | Role |
|---|---|
| `contract.py` | Contract schema, validation, hashing |
| `vault.py` | Sealed storage, OS lock, out-of-process label service, access ledger |
| `power.py` | MDE, search pricing, α dial, design advisor |
| `budget.py` | Complexity budget |
| `proposer.py`, `scoring.py` | Candidate generation, `sanitize_feedback`, feature scoring |
| `evaluator.py`, `metrics.py` | Group-aware CV, selection-aware null, interpretability bill, paired metrics |
| `gate.py` | Tier assignment from the OC table |
| `adversary.py` | Red team, context heterogeneity, pseudo-group control |
| `registry.py` | Contracts, procedures, predictions, alpha, publication, operator log, taxonomy |
| `certificate.py` | Signing, verification, Markdown face |
| `engine.py`, `api.py` | Orchestration; `verify(contract, data, candidate)` |
| `calibration.py`, `calibration_real.py`, `feedback_leak.py` | OC harness (synthetic and real-data plasmode); spec §5 leak test |
| `backtest.py` | Pre-registered replication back-test |
| `bench.py` | Discovery Validity Benchmark: tasks, reference agents, external-agent protocol, scoring |
| `demo.py`, `wdbc.py` | The two domains |
| `cli.py`, `server.py`, `mcp_server.py`, `web/` | Interfaces |

## Reproduce

`python scripts/reproduce_proofs.py` recomputes the back-test, the WDBC result and the fast test suite from public data in about five minutes. It writes `reproduction_report.json`. See [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Tests

```bash
python -m pytest                      # full suite, a few minutes
python -m pytest -m "not integration" # skip engine runs and subprocess tests
```

The spec's acceptance tests are each shown failing when their guard is broken: a leaky feedback channel, testing on the selection units, a context test that ignores clustering, a tampered ledger or artifact.

## Limitations

- **Refusing is the default.** By default the engine refuses questions whose declared effect is below the minimum detectable effect, which costs power when true effects are larger. `decision.if_underpowered: proceed` (attributed, printed on the certificate) runs them instead, with unchanged error control. A one-shot holdout that nobody peeks at still finds more effects; AutoResearch's edge is that the vault can't be re-tested.
- **Evidence is narrow.** The real-data OC table uses one dataset without groups. The back-test applies the gate's pricing to summary statistics, not the full engine, and depends on an implicit search size (K = 10; not significant at K = 5). No third party has reproduced the results yet. [docs/REPRODUCE.md](docs/REPRODUCE.md) gives the one command for that.
- **Same-account OS lock.** The lock stops accidental reads from the search process, not a deliberate unlock by the same OS account. Full separation means running the label service under its own account.
- **Signing.** Certificates carry an HMAC signature and, with the optional `cryptography` package, an Ed25519 public-key signature anyone can verify (`autoresearch public-key` prints the key). The Ed25519 key is derived from the secret key in `state/`, so that file must still be protected.
- **Shared code, not file-based modules.** The selection null holds the candidate pool fixed rather than re-running the adaptive proposer per permutation (a §12.1 surrogate). Modules call each other in-process rather than communicating only through files.
