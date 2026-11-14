# Operating characteristics of the certification gate

Generated: 2026-11-14T12:43:37Z
Substrate: immunotherapy synthetic planted-signal fixture

`Detected` is the promotion rate: the fraction of replicates the gate affirmed at either the **confirmed** or **supported** tier. `Confirmed` is the stricter top-tier-only rate. `FPR (alpha)` is the fraction of pure-null replicates with a positive confirmation delta and one-sided p <= alpha; `FPR (decision)` is the fraction the gate actually promoted.

| True effect | n | Mean realised effect | Detected | 95% CI | Confirmed | Without adversary | Adversary kills | FPR (alpha) | FPR (decision) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 20 | 0.001 | 0.00 | 0.00-0.16 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 0.030 | 20 | 0.030 | 0.00 | 0.00-0.16 | 0.00 | 0.00 | 0 | 0.00 | 0.00 |
| 0.060 | 20 | 0.060 | 0.10 | 0.03-0.30 | 0.00 | 0.10 | 0 | 0.00 | 0.00 |
| 0.100 | 20 | 0.100 | 0.85 | 0.64-0.95 | 0.00 | 0.85 | 0 | 0.00 | 0.00 |
| 0.150 | 20 | 0.150 | 1.00 | 0.84-1.00 | 0.85 | 1.00 | 0 | 0.00 | 0.00 |

## Adversary false-alarm rate (spec §9)

Of 39 replicates with a real planted effect that the gate would promote without the adversary, the adversary killed 0: false-alarm rate 0.00 (95% CI 0.00-0.09).

| Attack | Runs on real effects | Fail | Warn |
|---|---:|---:|---:|
| baseline_advocate | 80 | 0 | 0 |
| context_heterogeneity:indication | 80 | 0 | 5 |
| context_heterogeneity:site | 80 | 0 | 3 |
| feature_permutation:binding_rank | 78 | 0 | 19 |
| feature_permutation:expression | 65 | 0 | 0 |
| feature_permutation:foreignness | 7 | 0 | 0 |
| feature_permutation:presentation | 62 | 0 | 24 |
| feature_permutation:processing | 19 | 0 | 10 |
| feature_permutation:publicness | 1 | 0 | 0 |
| feature_permutation:stability | 8 | 0 | 0 |
| group_label_permutation | 80 | 0 | 60 |
| leakage_scan | 80 | 0 | 0 |
| leave_one_site_out | 80 | 0 | 0 |
| outcome_permutation | 80 | 0 | 0 |
| pseudo_group | 80 | 0 | 0 |

## Setup

- Replicates per effect: 20
- Candidate bank: 32
- Selection-null permutations: 39
- Discovery patients: 480
- Confirmation groups: 160
- Wall clock: 2272.2 s

## Method

Replicates use the real contract validator, proposer bank, group-aware CV, discovery-only selection-aware null, single-candidate confirmation, power oracle, full adversary and gate. Effects are planted in a known latent score and realised on the generated cohort; the search never sees that score.

## Limitations

- Synthetic substrate: this table demonstrates the gate's operating characteristics; it is not validated instrument performance on real data.
- Synthetic-only calibration is not believed (spec §14): the simulation is written by the method's author. A real-data table is still owed.
- The full adversary runs in every replicate; 'Detected' includes its kills. The adversary's own false-alarm rate is reported below.
- Candidates come from the operator-family seed bank (no adaptive generations) to keep replicates affordable.
- Effects are planted in a known latent score; the search sees only the contract features, never the planted score.
- A refusal caused by the power oracle is counted as a non-detection, because that is what the pipeline would actually emit.
