# Reproduce the evidence

Everything AutoResearch claims about itself can be recomputed from this repository with one command. If you are an outside reviewer, this is the request: run it and send back the report, pass or fail.

## Quick (about 5 minutes)

```bash
git clone https://github.com/Adnan-shamim-dar/autoresearch && cd autoresearch
pip install -e ".[dev]"
python scripts/reproduce_proofs.py
```

| Check | What it recomputes | Expected |
|---|---|---|
| `unit_tests` | The fast test suite, including every acceptance test with its broken-guard counterpart | all pass |
| `replication_backtest_rpp` | Downloads the Reproducibility Project: Psychology data (SHA-256 checked) and re-runs the pre-registered back-test | 47 accepted / 45 refused; 23 and 11 replicated |
| `benchmark_sample` | Re-runs 20 Discovery Validity Benchmark tasks for the three fast reference agents | 60/60 rows match |
| `wdbc_real_data_refusal_and_signs` | Builds the WDBC domain from scikit-learn, seals it, runs the engine | refused as underpowered; 9 of 10 pre-committed signs held; signature and manifest verify |

## Full (about an hour, mostly the synthetic table)

```bash
python scripts/reproduce_proofs.py --full
```

This adds both operating-characteristics tables (`calibration/oc_table.json`, `calibration/oc_table_real.json`) and compares every row's promotion and confirmation rate.

## What to send back

The script writes `reproduction_report.json`, which holds the git commit, Python version, platform and every observed-versus-expected value. It contains no personal data. Send it, or open an issue with it attached, whether or not the checks passed. A failure on another platform is the more useful report.

## What a pass does and does not show

A pass shows the published numbers follow from the published code and public data. It does not show that the method is right. For that, read [backtest/PREREGISTRATION.md](backtest/PREREGISTRATION.md) and check that its commit precedes [backtest/RESULTS.md](backtest/RESULTS.md) (`git log --format='%h %cI %s' -- docs/backtest`).
