# AutoResearch

A discovery engine that **prices the search before it runs**, and then either issues a signed **claim certificate** or a first-class **refusal with a design prescription**.

The engine is domain-general. The bundled application domain is personalised immunotherapy (neoantigen immunogenicity ranking), demonstrated on a **synthetic** TESLA/CEDAR-shaped substrate. Nothing in this repository is clinical evidence.

## What it does

1. **Contract.** You state the question up front: the unit, the group, the outcome, the baseline, the smallest effect that matters and the decision it would change. The contract is validated against the data and hashed.
2. **Power oracle.** Before any hypothesis is proposed, it computes the minimum detectable effect, both plain and after pricing *K* candidate hypotheses. If the question is underpowered, the run stops with a refusal that says how much more data, or which alternative design, would make it answerable.
3. **Search.** A proposer (a deterministic bank, optionally extended by an OpenAI-compatible LLM) generates candidate rules under a complexity budget. Candidates are scored with group-aware cross-validation on discovery data only.
4. **Selection-aware null.** The whole search is re-run on outcome permutations, so the p-value accounts for the fact that the best of *K* was picked.
5. **Sealed confirmation.** The selected rule is tested once on a sealed set. Labels are reachable only through an aggregate label service with a query budget, and every access is written to a hash-chained ledger.
6. **Gate and adversary.** A tiered gate (`confirmed` / `supported` / `exploratory` / refused) uses the published operating-characteristics table. Adversarial checks attack the result before it is issued.
7. **Certificate.** The result is HMAC-signed and lists the search price, alpha spent, provenance hashes, adversary results, limitations and a reproduce command. Every artifact in the run directory is covered by a manifest.

Alpha is conserved **per dataset lifetime** in the registry ledger, not per run, and the registry exposes the denominator: every registered contract, including the ones that never produced a claim.

## Quickstart

Requires Python 3.11+.

```bash
pip install -e ".[dev]"

autoresearch init          # build the synthetic substrate, seal the confirmation set
autoresearch demo          # end-to-end run (~4 min) + economics refusal
autoresearch certificates  # list issued certificates
autoresearch verify --run ar-<hash>-12345
autoresearch ledger        # verify vault and registry hash chains
autoresearch serve         # web console on http://127.0.0.1:8787
```

Without installing, run commands as `python -m autoresearch ...` with `PYTHONPATH=src`.

All commands take `--root <dir>` (default: the current directory) as the first argument.

| Command | Purpose |
|---|---|
| `init` | Build the synthetic substrate and seal the confirmation set |
| `demo [--reset]` | Run the end-to-end demo; `--reset` rebuilds the substrate (the calibration table is kept) |
| `run --contract C --discovery D ...` | Run the engine on your own contract and data |
| `validate --contract C --data D` | Check a contract against a table |
| `power --contract C --data D` | Price a search without running it |
| `economics` | Emit the power-only refusal on the economics margin question |
| `certificates`, `certificate --run R` | List or show certificates |
| `verify --run R` | Check the certificate hash, signature and artifact manifest. Exit codes: 0 ok, 2 certificate invalid, 3 manifest mismatch |
| `reproduce --run R [--fresh]` | Verify, or rebuild the substrate and re-run to compare hashes |
| `ledger` | Verify the vault and registry hash chains |
| `datasets` | Show the immunotherapy dataset atlas |
| `calibrate [--publish-only]` | Run the calibration harness, or re-publish the existing table |
| `serve` | Web console |
| `mcp` | MCP stdio server (tools: `autoresearch_power`, `_validate`, `_verify`, `_catalog`, `_run`) |

## Operating characteristics

Published in [calibration/oc_table.md](calibration/oc_table.md): 20 replicates per effect, 32 candidates, 39 permutations, 480 discovery patients, 160 confirmation groups, on the synthetic planted-signal fixture.

| True effect | Promoted (confirmed or supported) | Confirmed only | False-positive rate |
|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.00 | 0.00 |
| 0.03 | 0.00 | 0.00 | 0.00 |
| 0.06 | 0.10 | 0.00 | 0.00 |
| 0.10 | 0.85 | 0.00 | 0.00 |
| 0.15 | 1.00 | 0.85 | 0.00 |

The gate is conservative. It never promoted a null in 20 replicates (95% CI upper bound 0.16), but it needs an effect of about 0.10 before it reliably promotes. Twenty replicates per cell is small, so treat these as demonstration numbers, not a validated instrument.

## Layout

| Module | Role |
|---|---|
| `contract.py` | Contract schema, validation, hashing |
| `vault.py` | Sealed storage, aggregate label service, query budget, access ledger |
| `power.py` | MDE, search pricing, design advisor |
| `budget.py` | Complexity budget |
| `proposer.py`, `scoring.py` | Candidate generation and feature scoring |
| `evaluator.py`, `metrics.py` | Group-aware CV, selection-aware null, paired metrics |
| `gate.py` | Tier assignment from the OC table |
| `adversary.py` | Attacks on the result before issue |
| `registry.py` | Contract registry, prediction commitments, alpha ledger |
| `certificate.py` | Signing, verification, Markdown rendering |
| `engine.py` | Orchestration |
| `calibration.py` | OC harness |
| `demo.py` | Synthetic substrate and demo runs |
| `cli.py`, `server.py`, `mcp_server.py` | Interfaces |
| `domains/immunotherapy/` | Dataset catalog, modalities, opportunities, domain brief |

## Tests

```bash
python -m pytest                      # full suite (~1 min)
python -m pytest -m "not integration" # skip end-to-end engine runs
```

## Limitations

- The substrate is synthetic. The same contract is meant to run on real TESLA/CEDAR tables once access is granted, but that has not been done.
- The vault's separation between search and labels is enforced in-process (capabilities, encrypt-then-MAC, ledger). Production-grade isolation needs OS-level separation, for example a separate user or service holding the key.
- Certificates use a symmetric HMAC key stored under `state/`. Anyone holding the key can verify and also sign. Public verifiability would need asymmetric signatures.
- The web console's front-end script (`web/app.js`) has not been written yet. The HTML/CSS shell and JSON API exist.
