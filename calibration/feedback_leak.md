# Feedback-leak test (spec §5)

Generated: 2026-11-14T11:21:18Z

Pure-null data; one score-seeking proposer; two feedback channels. Every promotion is a false positive.

| Channel | Replicates | False positives | FPR | 95% CI |
|---|---:|---:|---:|---:|
| Leaky (scores on gate data) | 100 | 36 | 0.36 | 0.27-0.46 |
| Guarded (`sanitize_feedback`) | 100 | 1 | 0.01 | 0.00-0.05 |

Parameters: `{'replicates': 100, 'seed0': 5000}`
